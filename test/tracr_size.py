"""
Scaling experiment: detection cost vs. abstraction size for Tracr sort-rank programs.

For each SEQ_LEN in SEQ_LENS we:
  1. Compile a sort-rank RASP program into a Tracr transformer (the low-level model).
  2. Define a valid high-level model (rank_i = #{j≠i : token_j < token_i} for all i).
  3. Define a WRONG high-level model representing a specific wrong explanation hypothesis:
       rank_0 = seq_len - 1  when token_0 is the minimum-value token
               (correct formula)  otherwise
     This is wrong only for the "token_0 is minimum" input pattern, which
     occurs with probability exactly 1/seq_len for uniform token sampling.
  4. For each n in N_GRID, run CAE_down_nf/CAE_up_nf/CAE_down/CAE_up n_samples=n times.

Design rationale:
Wrong hypothesis with stochastic triggering:
  * valid condition:  score = 0.0 always (high-level model and low-level model always agree)
  * wrong condition:  score = 1/seq_len with prob 1/seq_len, else 0.0
    → power(n) = 1 − (1 − 1/seq_len)^n  (geometric distribution)
    → n for 95% power ≈ 3 × seq_len  (linear growth with program size)

Tracr precision:
Tracr's SelectorWidth sums ~seq_len soft attention weights, each with a small
error from the mlp_exactness sigmoid approximation.  For seq_len ≥ 12, the
default mlp_exactness=100 is insufficient: errors accumulate and roughly
10–30% of decoded rank values are wrong, giving valid-condition scores of 0.1–0.3.

Since the wrong-hypothesis signal is only 1/seq_len ≈ 8% for seq_len=12, a
valid noise floor of 0.1 completely buries it, making detection impossible.

Fix: we scale mlp_exactness = 30 × seq_len (300 for seq_len=10, 450 for 15).
A validation step after compilation measures empirical Tracr accuracy and warns
if it is below 99%.

The existing data for seq_len=2–10 (mlp_exactness=100 was sufficient) is
unaffected.  Only seq_len=12/15 need to be recompiled with higher exactness.
Delete or reset those entries in the JSON before restarting, or pass
--seq-lens 12 15 --reset-low_level_model to force recompilation.

JSON format:
    data[str(seq_len)][condition][metric][str(n)] = [score_run0, score_run1, ...]
condition ∈ {"valid", "wrong"}, metric ∈ {"CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"}.

Usage:
    python power_tracr_scaling.py                           # run forever
    python power_tracr_scaling.py --seq-lens 2 3 4 5 6     # subset
    python power_tracr_scaling.py --runs 50                 # stop after 50 rounds
    python power_tracr_scaling.py --seq-lens 12 15          # resume large seq_lens
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import load_system

_tracr_sys = load_system("08_tracr.py", "tracr_sys")

from causal_abstraction import (
    AbstractVariable, BottomUpSampler, CausalGraph, CoarseGrainingMap,
    EvaluationConfig, EvaluationEngine, LowLevelModel,
    MicroVariableSchema, RectSubspace, SystemState, TopDownSampler, ValueMap,
)
from causal_abstraction.schema import Variable
from causal_abstraction.tasks import StandardTasks

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir))
from runner import Condition

RESULTS_FILE = _test_dir / "results" / "power_tracr_scaling.json"

# Sequence lengths to test.  high-level model size = 2 * seq_len (token inputs + rank outputs).
# Compilation time grows with seq_len; allow 1–5 minutes per model for seq_len ≥ 10.
SEQ_LENS = [2, 3, 4, 5, 6, 8, 10, 12, 15]

# N_GRID covers the full expected detection range (analytic ≈ 3 × seq_len for 95%).
N_GRID = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75]

TARGET_RUNS = 100
CONDITIONS  = ["valid", "wrong"]
POWER_METRICS = frozenset({"CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"})


def _vocab_for(seq_len: int) -> set:
    """Use {1, …, seq_len+2}: large enough to avoid ties, small enough to compile fast."""
    return set(range(1, seq_len + 3))


def _mlp_exactness_for(seq_len: int) -> int:
    """
    Return the mlp_exactness needed to keep sort-rank accurate at this seq_len.

    Tracr's SelectorWidth (used for rank counting) sums ~seq_len soft attention
    weights. Each weight has error ε from the sigmoid approximation with temperature
    mlp_exactness. After summing, total error ≈ seq_len × ε. For correct rounding
    we need total error < 0.5, which requires mlp_exactness >> 2 × seq_len.

    Empirically: mlp_exactness=100 is sufficient for seq_len ≤ 10 but causes
    ~10–30% rank errors for seq_len ≥ 12. Using 30 × seq_len keeps the error
    well below 0.5 across the full range tested here.

    Higher values increase compilation time roughly linearly.
    """
    return max(100, seq_len * 30)


def _validate_low_level_model(low_level_model: "TracrScalingLowLevelModel", seq_len: int,
                  vocab: set, n_tests: int = 200) -> float:
    """
    Measure empirical sort-rank accuracy of the compiled Tracr model.

    Returns the fraction of random inputs where the low-level model output matches the
    exact sort-rank formula.  Prints a warning if accuracy < 99%.

    A valid score of 0 in the experiment requires accuracy ≈ 100%, because
    any low-level model error directly adds to the valid-condition hard-metric score.
    If accuracy is 90%, the valid floor is ~10%, which can swamp the
    wrong-hypothesis signal of 1/seq_len ≈ 7–8% for seq_len=12–15.
    """
    rng        = np.random.default_rng(seed=42)
    vocab_list = sorted(vocab)
    correct    = 0

    for _ in range(n_tests):
        tokens = [int(rng.choice(vocab_list)) for _ in range(seq_len)]
        expected = [
            sum(1 for j, t2 in enumerate(tokens) if j != i and t2 < t)
            for i, t in enumerate(tokens)
        ]
        # Feed as micro-values (label v → micro-value v−1)
        micro = np.array([[v - 1 for v in tokens]], dtype=float)
        result = low_level_model.forward_with_interventions(SystemState(), {"tokens": micro})
        got = [int(r) for r in result.values["ranks"][0].tolist()]
        if got == expected:
            correct += 1

    accuracy = correct / n_tests
    if accuracy >= 0.99:
        print(f"  Validation OK: {accuracy:.1%} correct ({n_tests} tests).", flush=True)
    else:
        print(
            f"  WARNING: Tracr accuracy = {accuracy:.1%} for seq_len={seq_len}.\n"
            f"  Valid-condition scores will be ~{1 - accuracy:.0%} instead of 0.\n"
            f"  The wrong-hypothesis signal is only 1/seq_len = {1/seq_len:.1%}.\n"
            f"  At this accuracy, detection may be impossible: the noise floor\n"
            f"  exceeds the signal. Increase mlp_exactness or reduce seq_len.",
            flush=True,
        )
    return accuracy


class TracrScalingLowLevelModel(LowLevelModel):
    """
    Tracr transformer wrapper with correct per-program vocab and integer-rounded outputs.

    The base TracrLowLevelModel in 08_tracr.py hardcodes len(VOCAB)=5 for clipping,
    which silently corrupts tokens for any larger vocab.  This class clips to the
    actual vocab bounds.

    Rank outputs are rounded to the nearest integer so they always fall cleanly
    inside RectSubspace bins [r-0.5, r+0.5].  Without rounding, Tracr's JAX
    outputs occasionally sit just outside a bin boundary (e.g. 1.9999 instead of
    2.0), causing UNMAPPED events and spurious 1.0 penalty scores.

    Micro-variables:
        "tokens"  shape (seq_len,)  float ≈ vocab_label − 1
        "ranks"   shape (seq_len,)  float = rounded integer rank
    """

    def __init__(self, compiled_model, seq_len: int, vocab: set, bos: str = "BOS"):
        self.compiled  = compiled_model
        self.seq_len   = seq_len
        self.vocab_min = min(vocab)
        self.vocab_max = max(vocab)
        self.bos       = bos

    def forward_with_interventions(self, input_state: SystemState,
                                   interventions: dict) -> SystemState:
        batch_size  = 1
        token_array = np.zeros((batch_size, self.seq_len), dtype=float)

        if "tokens" in interventions:
            spec = interventions["tokens"]
            if isinstance(spec, list):
                for pos_idx, val in spec:
                    if hasattr(val, "shape"):
                        batch_size = val.shape[0]
                    if token_array.shape[0] != batch_size:
                        token_array = np.zeros((batch_size, self.seq_len), dtype=float)
                    if pos_idx is None:
                        token_array = val.astype(float)
                    elif isinstance(pos_idx, int):
                        token_array[:, pos_idx] = val.squeeze(-1)
                    else:
                        token_array[:, pos_idx] = val.squeeze(-1)
            elif isinstance(spec, np.ndarray):
                token_array = spec.astype(float)
                batch_size  = token_array.shape[0]

        all_ranks = np.zeros((batch_size, self.seq_len), dtype=float)

        for b in range(batch_size):
            # Micro-value v−1 → vocab label v; clip to valid vocab range.
            vocab_labels = [
                int(np.clip(round(token_array[b, i]) + 1,
                            self.vocab_min, self.vocab_max))
                for i in range(self.seq_len)
            ]
            output    = self.compiled.apply([self.bos] + vocab_labels)
            decoded   = output.decoded[1:]   # strip BOS
            # Round to nearest integer: eliminates floating-point bin-boundary UNMAPPED.
            all_ranks[b] = np.round(np.array(decoded, dtype=float))

        return input_state.merge(SystemState(values={
            "tokens": token_array,
            "ranks":  all_ranks,
        }))


# Compile each Tracr model once per process (compilation can take several minutes
# for larger seq_lens).
_compiled_cache: Dict[int, object]          = {}
_low_level_model_cache:      Dict[int, TracrScalingLowLevelModel] = {}


def _get_low_level_model(seq_len: int) -> TracrScalingLowLevelModel:
    if seq_len not in _low_level_model_cache:
        vocab = _vocab_for(seq_len)
        print(f"  Compiling sort-rank (seq_len={seq_len}, "
              f"vocab={sorted(vocab)})...", flush=True)

        from tracr.rasp import rasp
        from tracr.compiler import compiling

        less_than = rasp.Select(
            rasp.tokens, rasp.tokens, rasp.Comparison.LT
        ).named("less_than")
        rank = rasp.SelectorWidth(less_than).named("rank")

        exactness = _mlp_exactness_for(seq_len)
        compiled = compiling.compile_rasp_to_model(
            rank,
            vocab=vocab,
            max_seq_len=seq_len,
            compiler_bos="BOS",
            mlp_exactness=exactness,
        )
        _compiled_cache[seq_len] = compiled
        low_level_model = TracrScalingLowLevelModel(compiled, seq_len, vocab)
        _low_level_model_cache[seq_len] = low_level_model
        print(f"  Done (seq_len={seq_len}, mlp_exactness={exactness}).", flush=True)
        _validate_low_level_model(low_level_model, seq_len, vocab)

    return _low_level_model_cache[seq_len]


def _build_maps(seq_len: int) -> Tuple[CoarseGrainingMap, ValueMap]:
    vocab = _vocab_for(seq_len)

    schema = MicroVariableSchema([
        Variable("tokens", shape=(seq_len,), dtype=int),
        Variable("ranks",  shape=(seq_len,), dtype=int),
    ])

    cg_mapping: Dict[str, list] = {}
    for i in range(seq_len):
        cg_mapping[f"token_{i}"] = [("tokens", i)]
        cg_mapping[f"rank_{i}"]  = [("ranks",  i)]

    cg_map = CoarseGrainingMap(schema, cg_mapping)

    # Token label v → micro-value ≈ v−1.
    tok_specs  = {v: RectSubspace((v - 1 - 0.5, v - 1 + 0.5)) for v in vocab}
    # Rank label r → micro-value = r (integer after rounding).
    rank_specs = {r: RectSubspace((r - 0.5, r + 0.5)) for r in range(seq_len)}

    var_specs: Dict[str, dict] = {}
    for i in range(seq_len):
        var_specs[f"token_{i}"] = tok_specs
        var_specs[f"rank_{i}"]  = rank_specs

    return cg_map, ValueMap(cg_map, var_specs)


def _build_high_level_model(seq_len: int, wrong: bool = False) -> CausalGraph:
    """
    Build sort-rank high-level model.

    valid:  rank_i = #{j ≠ i : token_j < token_i}  for all i   (correct)

    wrong:  rank_0 = seq_len − 1   if correct_rank_0 == 0     (wrong hypothesis)
            rank_0 = correct_rank_0  otherwise
            rank_i = #{j ≠ i : token_j < token_i}  for i > 0  (correct)

    The wrong hypothesis triggers when token_0 is the minimum-value token,
    which happens with probability ≈ 1/seq_len under uniform token sampling.
    When triggered: wrong_rank_0 = seq_len−1 (max) while low-level model gives 0 (min).
    This gives power(n) = 1 − (1 − 1/seq_len)^n, a clean geometric curve.

    The conceptual mistake encoded: "the minimum-value token at position 0
    always gets the highest rank", exactly the opposite of the correct LT rule.

    Choosing a wrong hypothesis with p ≈ 1/seq_len means:
      • Larger programs → lower detection probability per sample → more samples needed.
      • Required n ≈ 3 × seq_len for 95% power → linear growth.
    """
    vocab     = _vocab_for(seq_len)
    tok_dist  = {v: 1 / len(vocab) for v in vocab}
    rank_dist = {r: 1 / seq_len    for r in range(seq_len)}
    all_pos   = list(range(seq_len))

    high_level_model = CausalGraph()
    for i in range(seq_len):
        high_level_model.add_variable(
            AbstractVariable(f"token_{i}", distribution=tok_dist),
            equation=lambda **_: None,
            parents=[],
        )

    for i in range(seq_len):
        is_wrong_pos = (wrong and i == 0)

        def _make_eq(pos: int, apply_wrong: bool, sl: int = seq_len):
            def eq(**kwargs):
                token_i = kwargs[f"token_{pos}"]
                correct = sum(
                    1 for j in all_pos
                    if j != pos and kwargs[f"token_{j}"] < token_i
                )
                if apply_wrong and correct == 0:
                    # Wrong hypothesis: predict max rank when the correct rank is 0.
                    return sl - 1
                return correct
            return eq

        high_level_model.add_variable(
            AbstractVariable(f"rank_{i}", distribution=rank_dist),
            equation=_make_eq(i, is_wrong_pos),
            parents=[f"token_{j}" for j in range(seq_len)],
        )

    return high_level_model


def _build_tasks(builder, vm: ValueMap, seq_len: int) -> list:
    td = TopDownSampler(vm)
    bu = BottomUpSampler(vm)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def _compute_seed(seq_len: int, cond: str, run_index: int) -> int:
    return (run_index * len(CONDITIONS) * len(SEQ_LENS)
            + SEQ_LENS.index(seq_len) * len(CONDITIONS)
            + CONDITIONS.index(cond))


def build_condition(seq_len: int, cond: str, run_index: int) -> Condition:
    low_level_model    = _get_low_level_model(seq_len)
    cg, vm = _build_maps(seq_len)
    high_level_model    = _build_high_level_model(seq_len, wrong=(cond == "wrong"))

    seed = _compute_seed(seq_len, cond, run_index)
    cfg  = EvaluationConfig(metric="hard", seed=seed, n_jobs=1)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)

    interv_dom = [f"token_{i}" for i in range(seq_len)]

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, vm, seq_len),
        analytical=[],
        sampler=TopDownSampler(vm),
        task_kwargs=dict(
            batch_size=1,
            max_interventions=seq_len,
            intervention_domain=interv_dom,
        ),
    )


# Persistence helpers

def load_results(results_file: Path) -> Dict:
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            print(f"Loaded existing results from {results_file.name}")
            return data
        except Exception as e:
            print(f"Warning: could not load {results_file.name} ({e}). Starting fresh.")
    return {}


def save_results(data: Dict, results_file: Path) -> None:
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(results_file)


def _count(data: Dict, seq_len: int, cond: str, metric: str, n: int) -> int:
    return len(
        data.get(str(seq_len), {})
            .get(cond, {})
            .get(metric, {})
            .get(str(n), [])
    )


def _append(data: Dict, seq_len: int, cond: str,
            metric: str, n: int, score: float) -> None:
    stored = (None if score is None or (isinstance(score, float) and np.isnan(score))
              else float(score))
    (data.setdefault(str(seq_len), {})
         .setdefault(cond, {})
         .setdefault(metric, {})
         .setdefault(str(n), [])
         .append(stored))


def _cell_done(data: Dict, seq_len: int, cond: str, n: int, round_idx: int) -> bool:
    best = max(
        (_count(data, seq_len, cond, m, n) for m in POWER_METRICS), default=0)
    return best > round_idx


def _print_summary(data: Dict, seq_lens: List[int], n_grid: List[int],
                   rounds_done: int) -> None:
    print(f"\n  Summary after {rounds_done} rounds:")
    for sl in seq_lens:
        for cond in CONDITIONS:
            counts = {
                n: max((_count(data, sl, cond, m, n) for m in POWER_METRICS), default=0)
                for n in n_grid
            }
            row = "  ".join(f"n={n}:{counts[n]}" for n in n_grid)
            print(f"    seq={sl:>2} {cond:<8}  {row}")
    print()


def run_scaling_suite(
    seq_lens: List[int],
    n_grid: List[int],
    target_runs: int,
    results_file: Path,
    args=None,
) -> None:
    data   = load_results(results_file)
    target = (args.runs if args is not None and getattr(args, "runs", None) is not None
              else target_runs)

    print(f"\n{'=' * 65}")
    print(f"  Tracr sort-rank scaling experiment")
    print(f"  seq_lens={seq_lens}  n_grid={n_grid}  target={target}")
    print(f"  mlp_exactness per seq_len: "
          + ", ".join(f"{sl}→{_mlp_exactness_for(sl)}" for sl in seq_lens))
    print(f"  Wrong hypothesis: rank_0 = seq_len−1 when token_0 is minimum.")
    print(f"  Detection prob per sample = 1/seq_len  →  n_req ≈ 3×seq_len.")
    print(f"{'=' * 65}\n")

    print("Pre-compiling Tracr models (this may take several minutes)...")
    for sl in seq_lens:
        _get_low_level_model(sl)
    print()

    start_round = min(
        max((_count(data, sl, cond, m, n) for m in POWER_METRICS), default=0)
        for sl   in seq_lens
        for cond in CONDITIONS
        for n    in n_grid
    )
    if start_round > 0:
        print(f"  Resuming from round {start_round + 1}.\n")

    try:
        for round_idx in range(start_round, target):
            round_did_work = False

            for n in n_grid:
                for sl in seq_lens:
                    for cond in CONDITIONS:
                        if _cell_done(data, sl, cond, n, round_idx):
                            continue

                        round_did_work = True
                        cr    = build_condition(sl, cond, round_idx)
                        tasks = [t for t in cr.tasks if t.name in POWER_METRICS]
                        if not tasks:
                            continue

                        kwargs             = dict(cr.task_kwargs)
                        kwargs["n_samples"] = n
                        result             = cr.engine.run_tasks(tasks, **kwargs)

                        n_saved = 0
                        for name, res in result.items:
                            _append(data, sl, cond, name, n, float(res.score))
                            n_saved += 1

                        save_results(data, results_file)
                        print(
                            f"[Round {round_idx + 1:>4}/{target}]  "
                            f"n={n:>4}  seq_len={sl:>2}  {cond:<8}  "
                            f"({n_saved} metrics)  -> {results_file.name}",
                            flush=True,
                        )

            if not round_did_work:
                print(f"\nAll {target} rounds complete.")
                break

            if (round_idx + 1) % 10 == 0:
                _print_summary(data, seq_lens, n_grid, round_idx + 1)

    except KeyboardInterrupt:
        print(f"\nStopped. Results saved to {results_file}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tracr sort-rank scaling: detection cost vs. high-level model size")
    p.add_argument("--seq-lens", nargs="+", type=int, default=None,
                   help=f"Sequence lengths (default: {SEQ_LENS})")
    p.add_argument("--runs", type=int, default=None,
                   help="Number of rounds (default: run until Ctrl+C)")
    p.add_argument("--reset-low_level_model", action="store_true",
                   help="Clear compiled-model cache before starting "
                        "(forces recompilation with current mlp_exactness).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args     = _parse_args()
    seq_lens = args.seq_lens or SEQ_LENS
    if getattr(args, "reset_low_level_model", False):
        _compiled_cache.clear()
        _low_level_model_cache.clear()
        print("Compiled-model cache cleared. Models will be recompiled.")
    run_scaling_suite(
        seq_lens=seq_lens,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        results_file=RESULTS_FILE,
        args=args,
    )
"""
Example 8: Full causal abstraction experiment connecting:
  - High-level model E: RASP sort-rank program as a CausalGraph
  - Low-level model M: Tracr-compiled JAX transformer

Causal structure: token_0, token_1, token_2 -> rank_0, rank_1, rank_2
rank_i = #{j : token_j < token_i}   (zero-based rank)

Residual stream layout (from Step 1 output, 18 dims total):
  [0-3]    indices:0..3                       <- positional encoding (not mapped)
  [4]      one                                <- constant 1           (not mapped)
  [5-9]    rank_1:0..4                        <- one-hot rank output
  [10]     rank_1_selector_width_attn_output  <- intermediate (internal)
  [11-15]  tokens:1..5                        <- one-hot token encoding
  [16]     tokens:BOS                         <- not mapped
  [17]     tokens:compiler_pad                <- not mapped

Note: the "(internal)" / "(not mapped)" tags above describe tracr's raw residual
dimensions; they are not the library's Phi / internal-variable sets. Those residual
dims are never exposed as micro-variables, so for this system Phi = {} and there are
no declared internal variables (faithful and non-faithful CAE coincide).

Micro-variable design:
Rather than hooking into the raw 18-dim residual stream (which would require
JAX forward-hooks), we expose two higher-level micro-variables:

  tokens  shape=(SEQ_LEN,)  float array of token indices 0..4
                             (vocab label v maps to index v-1)
  ranks   shape=(SEQ_LEN,)  float array of rank values 0..4

The TracrLowLevelModel reads token indices from interventions, converts them
to vocab labels, runs the JAX model, and stores decoded ranks.
This is exact because Tracr guarantees lossless round-trip encoding.

Coarse-graining map:
  token_i  ->  tokens[i]   (scalar float ≈ v-1 for vocab label v)
  rank_i   ->  ranks[i]    (scalar float = r for rank r)

ValueMap:
  token_i:  label v (1..5) -> RectSubspace centered at v-1,  width 1
  rank_i:   label r (0..4) -> RectSubspace centered at r,    width 1
"""
import json
from dataclasses import asdict

import numpy as np
from typing import Dict, Any, Optional, List

from tracr.rasp import rasp
from tracr.compiler import compiling

from causal_abstraction import (
    CausalGraph,
    AbstractVariable,
    LowLevelModel,
    SystemState,
    MicroVariableSchema,
    CoarseGrainingMap,
    ValueMap,
    RectSubspace,
    Variable
)

# Constants
SEQ_LEN = 3
VOCAB   = {1, 2, 3, 4, 5}
BOS     = "BOS"


def build_compiled_model():
    less_than = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.LT
    ).named("less_than")
    rank = rasp.SelectorWidth(less_than).named("rank")

    return compiling.compile_rasp_to_model(
        rank,
        vocab=VOCAB,
        max_seq_len=SEQ_LEN,
        compiler_bos=BOS,
        mlp_exactness=100,
    )


# low-level model wrapping the JAX transformer
class TracrLowLevelModel(LowLevelModel):
    """
    Interventions on tokens are a list of (position_index, value_array) pairs.
    Each value_array has shape (batch, 1).

    Interventions on 'ranks' are intentionally not supported; ranks are always
    computed by running the transformer.
    """

    def __init__(self, compiled_model, seq_len: int):
        self.compiled = compiled_model
        self.seq_len  = seq_len

    def forward_with_interventions(self, input_state: SystemState, interventions: dict[str, Any]) -> SystemState:
        batch_size = 1

        # Reconstruct the token index array from interventions
        # Default: all tokens set to index 0 (vocab label 1)
        token_array = np.zeros((batch_size, self.seq_len), dtype=float)

        if "tokens" in interventions:
            spec = interventions["tokens"]

            if isinstance(spec, list):
                # List of (index, array_of_shape_(batch, 1)) tuples
                for pos_idx, val in spec:
                    # Infer batch size from first valid val
                    if hasattr(val, "shape"):
                        batch_size = val.shape[0]
                    if token_array.shape[0] != batch_size:
                        token_array = np.zeros((batch_size, self.seq_len), dtype=float)

                    if pos_idx is None:
                        # Full replacement
                        token_array = val.astype(float)
                    elif isinstance(pos_idx, int):
                        token_array[:, pos_idx] = val.squeeze(-1)
                    else:
                        token_array[:, pos_idx] = val.squeeze(-1)

            elif isinstance(spec, np.ndarray):
                token_array = spec.astype(float)
                batch_size  = token_array.shape[0]

        # Run the JAX transformer for each batch item
        all_ranks = np.zeros((batch_size, self.seq_len), dtype=float)

        for b in range(batch_size):
            # Convert float indices 0..4 -> vocab labels 1..5
            vocab_labels = [
                int(np.clip(round(token_array[b, i]), 0, len(VOCAB) - 1)) + 1
                for i in range(self.seq_len)
            ]

            input_seq = [BOS] + vocab_labels
            output    = self.compiled.apply(input_seq)

            decoded        = output.decoded[1:]   # strip BOS -> list of ints
            all_ranks[b]   = np.array(decoded, dtype=float)

        return SystemState(values={
            "tokens": token_array,   # (batch, SEQ_LEN)
            "ranks":  all_ranks,     # (batch, SEQ_LEN)
        })

# Correct rank equation: rank = #{j != i : token_j < token_i}
def _make_rank_equation(pos: int):
    def rank_eq(**kwargs):
        token_i = kwargs[f"token_{pos}"]
        return sum(
            1 for j in range(SEQ_LEN)
            if j != pos and kwargs[f"token_{j}"] < token_i
        )
    return rank_eq


# Failing rank equation: uses GT instead of LT -> reverses ranks
def _make_failing_rank_equation(pos: int):
    def rank_eq(**kwargs):
        token_i = kwargs[f"token_{pos}"]
        return sum(
            1 for j in range(SEQ_LEN)
            if j != pos and kwargs[f"token_{j}"] > token_i   # GT instead of LT
        )
    return rank_eq


def _build_shared_maps():
    schema = MicroVariableSchema([
        Variable("tokens", shape=(SEQ_LEN,), dtype=int),
        Variable("ranks",  shape=(SEQ_LEN,), dtype=int),
    ])
    cg_mapping: Dict[str, List] = {
        "token_0": [("tokens", 0)],
        "token_1": [("tokens", 1)],
        "token_2": [("tokens", 2)],
        "rank_0":  [("ranks", 0)],
        "rank_1":  [("ranks", 1)],
        "rank_2":  [("ranks", 2)],
    }
    cg_map = CoarseGrainingMap(schema, cg_mapping)

    _token_specs = {v: RectSubspace((v - 1 - 0.5, v - 1 + 0.5)) for v in range(1, 6)}
    _rank_specs  = {r: RectSubspace((r - 0.5, r + 0.5)) for r in range(5)}
    variable_specs = {
        "token_0": _token_specs, "token_1": _token_specs, "token_2": _token_specs,
        "rank_0":  _rank_specs,  "rank_1":  _rank_specs,  "rank_2":  _rank_specs,
    }
    value_map = ValueMap(cg_map, variable_specs)
    return schema, cg_map, value_map


def _build_high_level_model(rank_eq_factory):
    """Build a CausalGraph using the given rank equation factory."""
    high_level_model = CausalGraph()
    _tok_dist = {v: 1 / len(VOCAB) for v in VOCAB}
    _rank_dist = {r: 1 / 5 for r in range(5)}  # Ranks 0..4 are valid labels

    for _i in range(SEQ_LEN):
        high_level_model.add_variable(
            AbstractVariable(f"token_{_i}", distribution=_tok_dist),
            equation=lambda **_: None,
            parents=[],
        )
    for _i in range(SEQ_LEN):
        high_level_model.add_variable(
            AbstractVariable(f"rank_{_i}", distribution=_rank_dist),
            equation=rank_eq_factory(_i),
            parents=[f"token_{j}" for j in range(SEQ_LEN)],
        )
    return high_level_model


def run_full_suite(*args, **kwargs):
    """Deprecated shim. Run the evaluation suite from the command line instead:
    ``python test/08_tracr.py`` (built on ``test/runner.run_suite``)."""
    raise RuntimeError(
        "run_full_suite is not available here; run the Tracr evaluation "
        "suite directly with `python test/08_tracr.py`."
    )


if __name__ == "__main__":
    print("Compiling RASP program and running a sample sequence...")
    compiled_model = build_compiled_model()
    low_level_model = TracrLowLevelModel(compiled_model, SEQ_LEN)
    # Run a sample: tokens [3, 1, 4] -> expected ranks [1, 0, 2]
    import numpy as np
    state = low_level_model.forward_with_interventions(
        SystemState(values={}),
        {"tokens": np.array([[2.0, 0.0, 3.0]])},  # 0-indexed: 3->2, 1->0, 4->3
    )
    ranks = state.values.get("ranks", None)
    print(f"Tokens [3,1,4] -> ranks: {np.asarray(ranks).ravel() if ranks is not None else 'N/A'}")
    print("Expected: [1, 0, 2]")
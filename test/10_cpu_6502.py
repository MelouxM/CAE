"""
Full evaluation suite for the MOS 6502 Three-Level Abstraction.

Six conditions across three abstraction layers:
  valid_gate_isa          Gate (low-level model) vs ISA (high-level model)
  valid_transistor_gate   Transistor (low-level model) vs Gate (high-level model)
  valid_transistor_isa    Transistor (low-level model) vs ISA (high-level model)
  broken_gate_isa         Broken Gate (low-level model) vs ISA (high-level model)     - A[7] stuck-at-0
  broken_transistor_gate  Broken Transistor (low-level model) vs Gate (high-level model)
  broken_transistor_isa   Broken Transistor (low-level model) vs ISA (high-level model)

Crash isolation:
  Each condition runs in a forked child process.  The transistor bridge
  (perfect6502) has a latent heap corruption that can cause SIGSEGV/SIGABRT
  when glibc's malloc walks the metadata.  By forking, the parent's heap
  stays clean and crashes in children are caught and logged.

  The parent process loads only ISA + Gate bridges.  The transistor library
  is loaded only inside child processes that need it.

Resume logic:
  Conditions are processed least-data-first so that a crash in condition X
  doesn't permanently put X behind the others.
"""
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

from utils import load_system

_cpu = load_system("10_cpu_6502.py", "cpu_6502")

from causal_abstraction import (
    EvaluationConfig, EvaluationEngine,
    L2Metric, VarianceDecompositionMetric, ComplexityShiftMetric,
    IIAMetric, BCCMetric, IBLagrangianMetric, CIBLagrangianMetric,
    InfidelityMetric, ProbingMetric, RelationalFidelityMetric,
    SobolSensitivityMetric, ConditionalIndependenceMetric, MMDMetric,
    MacroscopicInvarianceMetric, MallowsCpMetric, RMSEMetric, SymbionMetric, RectSubspace, BottomUpSampler,
)
from causal_abstraction.primitives import SystemState
from causal_abstraction.tasks import StandardTasks
from utils import (
    collect_scores, load_results, save_results, n_runs, print_table,
)

RESULTS_FILE = Path(__file__).parent / "results" / "10_cpu_6502_results.json"
N_SAMPLES    = 200
OUTPUTS      = _cpu.OUTPUTS
INPUTS       = _cpu.INPUTS

# Registers whose values can be perturbed for infidelity measurement.
# Opcode/operand are excluded because perturbing them turns the instruction
# into a random multi-byte opcode. The ISA and Gate bridges have different
# memory layouts around the test instruction, so the "operand" bytes they
# read would differ → spurious infidelity.
REGISTER_INPUTS = ["A_in", "X_in", "Y_in", "S_in", "P_in"]

CONDITIONS = [
    "valid_gate_isa",
    "valid_transistor_gate",
    "valid_transistor_isa",
    "broken_gate_isa",
    "broken_transistor_gate",
    "broken_transistor_isa",
]

COMPARISONS = [
    ("valid_gate_isa",        "broken_gate_isa",
     "gate-isa: valid -> broken A[7]"),
    ("valid_transistor_gate", "broken_transistor_gate",
     "trans-gate: valid -> broken A[7]"),
    ("valid_transistor_isa",  "broken_transistor_isa",
     "trans-isa: valid -> broken A[7]"),
    ("valid_transistor_gate", "valid_gate_isa",
     "transistor -> gate level"),
]

# Conditions that require the transistor bridge
_TRANSISTOR_CONDS = {
    "valid_transistor_gate", "valid_transistor_isa",
    "broken_transistor_gate", "broken_transistor_isa",
}

# Shared objects - ISA and Gate loaded in parent, transistor in child only

print("Loading ISA and Gate simulators...")
_SCHEMA = _cpu.build_schema()
_CG     = _cpu.build_cg(_SCHEMA)
_VM     = _cpu.build_vm(_CG)

_ISA_SIM  = None
_GATE_SIM = None


def _isa():
    global _ISA_SIM
    if _ISA_SIM is None:
        _ISA_SIM = _cpu.ISASimulator()
    return _ISA_SIM


def _gate():
    global _GATE_SIM
    if _GATE_SIM is None:
        _GATE_SIM = _cpu.GateSimulator()
    return _GATE_SIM


def _transistor():
    """Create a TransistorSimulator.  Call only inside a forked child."""
    return _cpu.TransistorSimulator()


_IMPLIED_OPCODE_SPECS = {
    float(op): RectSubspace((float(op) - 0.5, float(op) + 0.5))
    for op, _, ilen in _cpu.TEST_OPCODES if ilen == 1
}
_BINARY_REG = {
    0: RectSubspace((-0.5, 0.5)),
    255: RectSubspace((254.5, 255.5)),
}
_PLAG_SAMPLE = {
    0x20: RectSubspace((0x1F + 0.5, 0x20 + 0.5)),  # only constant bit set
    0xA0: RectSubspace((0x9F + 0.5, 0xA0 + 0.5)),  # N flag set
}
_FULL_OUT = {0.0: RectSubspace((-0.5, 255.5))}

_COARSE_SYMBION_VM = _cpu.RegisterValueMap(_CG, {
    "opcode": _IMPLIED_OPCODE_SPECS,
    "operand": {0.0: RectSubspace((-0.5, 0.5))},  # always 0 for implied
    "A_in": _BINARY_REG,
    "X_in": _BINARY_REG,
    "Y_in": _BINARY_REG,
    "S_in": {0xFD: RectSubspace((0xFC + 0.5, 0xFD + 0.5))},  # fixed stack ptr
    "P_in": _PLAG_SAMPLE,
    "A_out": _FULL_OUT,
    "X_out": _FULL_OUT,
    "Y_out": _FULL_OUT,
    "S_out": _FULL_OUT,
    "P_out": _FULL_OUT,
})


class CPUSymbionMetric(SymbionMetric):
    """
    Symbion for the CPU 6502 using a coarse discrete encoding.

    The main RegisterValueMap has one label per register covering [0, 255],
    making exhaustive enumeration trivial and meaningless. This subclass
    substitutes a small finite value map: 2 representative register values
    (0 and 255) and implied-only opcodes, giving 272 combinations total.
    Labels correspond to actual register values so high-level model equations dispatch
    correctly.
    """

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        return super()._compute(
            high_level_model, low_level_model, _COARSE_SYMBION_VM, cg_map, sampler, n_samples, config
        )

# Eagerly load ISA + Gate so they're ready before forking
_isa()
_gate()
print("  ISA + Gate ready.\n")

# Quick verification (gate only - transistor verified in its first child)
_m, _t, _ = _cpu.verify_bridges(_isa(), _gate(), "ISA vs Gate", n_tests=100, seed=123)
if _m < _t:
    print(f"  WARNING: {_t - _m}/{_t} ISA-vs-Gate mismatches\n")

# Task and metric builders

def _build_tasks(builder, vm):
    s  = _cpu.InstructionSampler(vm)
    bu = BottomUpSampler(vm)
    return [
        StandardTasks.score(builder, sampler=s,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=s,  name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
        StandardTasks.observational_r2(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_mse( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_kl(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_jsd( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational(builder, RMSEMetric(), sampler=bu,
                                     output_vars=OUTPUTS, name="RMSE"),
        StandardTasks.observational(builder, L2Metric(), sampler=bu,
                                     output_vars=OUTPUTS, name="L2"),
        StandardTasks.observational(builder, MMDMetric(), sampler=bu,
                                     output_vars=OUTPUTS, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=bu,
                                     output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(), sampler=bu,
                                     output_vars=OUTPUTS, name="VarDecomp"),
    ]


def _build_analytical():
    return [
        IIAMetric(inner_metric="mse", output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=min(N_SAMPLES, 20), inner_metric="mse"),
        MacroscopicInvarianceMetric(n_pairs=min(N_SAMPLES, 10), inner_metric="mse"),
        RelationalFidelityMetric(output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        InfidelityMetric(perturb_vars=REGISTER_INPUTS, output_vars=OUTPUTS, delta_fraction=0.1),
        ProbingMetric(n_train=max(N_SAMPLES * 2, 80), inner_metric="nmse"),
        MallowsCpMetric(n_params=7, output_vars=OUTPUTS),
        CPUSymbionMetric(output_vars=OUTPUTS),
    ]


# Condition builder

def _build_and_run(cond, run_index, metric_filter=None):
    """
    Build the condition, run tasks + analytical, return scores dict.
    This is called inside a forked child process.
    """
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg  = EvaluationConfig(metric="mse", seed=seed)

    # Resolve high-level model and low-level model based on condition name
    needs_transistor = cond in _TRANSISTOR_CONDS
    is_broken = cond.startswith("broken_")

    # Determine the base low-level model
    if "transistor" in cond:
        base_low_level_model = _transistor()
    else:
        base_low_level_model = _gate()

    # Apply stuck-at-0 fault if broken
    low_level_model = _cpu.StuckA7Simulator(base_low_level_model) if is_broken else base_low_level_model

    # Determine high-level model (the "reference" simulator at the higher abstraction level)
    if cond.endswith("_isa"):
        high_level_model = _cpu.build_isa_high_level_model(_isa())
    elif cond.endswith("_gate"):
        high_level_model = _cpu.build_gate_high_level_model(_gate())
    else:
        raise ValueError(f"Unknown condition: {cond}")

    engine  = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    sampler = _cpu.InstructionSampler(_VM)

    task_kwargs = dict(
        n_samples=N_SAMPLES,
        batch_size=1,
        max_interventions=len(INPUTS),
        intervention_domain=INPUTS,
    )

    scores = {}

    # Build and filter tasks
    tasks = _build_tasks(engine.builder, _VM)
    analytical = _build_analytical()

    if metric_filter is not None:
        mf = set(metric_filter)
        tasks = [t for t in tasks if t.name in mf]
        analytical = [m for m in analytical if type(m).__name__ in mf]

    if tasks:
        task_results = engine.run_tasks(tasks, **task_kwargs)
        scores.update(collect_scores(task_results, {}))

    if analytical:
        a_results = engine.run_analytical_metrics(
            analytical, sampler=sampler, n_samples=N_SAMPLES)
        scores.update(collect_scores({}, a_results))

    return scores


# Fork-based crash isolation

def _run_condition_forked(cond, run_index, metric_filter=None):
    """
    Run a condition in a forked child process.
    Returns the scores dict, or None if the child crashed.
    """
    # Create temp file for results (before forking)
    fd, tmppath = tempfile.mkstemp(suffix=".json", prefix=f"6502_{cond}_")
    os.close(fd)

    pid = os.fork()

    if pid == 0:
        # Child process
        # Suppress tqdm's signal handlers
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        try:
            scores = _build_and_run(cond, run_index, metric_filter)

            serializable = {
                k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
                for k, v in scores.items()
            }

            with open(tmppath, "w") as f:
                json.dump(serializable, f)

            os._exit(0)
        except Exception as e:
            sys.stderr.write(f"  ERROR in {cond}: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            os._exit(1)
    else:
        # Parent process
        _, status = os.waitpid(pid, 0)

        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            # Child succeeded
            try:
                with open(tmppath) as f:
                    scores = json.load(f)
                return scores
            except Exception:
                return None
            finally:
                _safe_unlink(tmppath)

        else:
            # Child crashed or returned nonzero
            if os.WIFSIGNALED(status):
                sig = os.WTERMSIG(status)
                signame = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
                print(f"  CRASH: {cond} killed by {signame} - skipping this run")
            else:
                code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                print(f"  FAIL: {cond} exited with code {code} - skipping this run")

            _safe_unlink(tmppath)
            return None


def _safe_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


# Per-condition data counting

def _cond_count(data, cond):
    """How many complete runs does this condition have?"""
    cd = data.get(cond, {})
    if not cd:
        return 0
    lengths = [len(v) for v in cd.values() if isinstance(v, list)]
    return min(lengths) if lengths else 0


# Main loop

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MOS 6502 evaluation suite")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of runs per condition (default: infinite)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Only compute these metrics")
    parser.add_argument("--print-every", type=int, default=5,
                        help="Print table every N complete rounds")
    args = parser.parse_args()

    data = load_results(RESULTS_FILE, CONDITIONS)
    metric_filter = args.metrics

    # If re-running specific metrics, wipe those columns and set target
    if metric_filter:
        existing = n_runs(data, CONDITIONS)
        target_n = args.runs if args.runs is not None else existing
        if target_n == 0:
            print("No existing runs. Run without --metrics first.")
            return
        mf_set = set(metric_filter)
        for cond in CONDITIONS:
            if cond in data:
                for mk in list(data[cond].keys()):
                    if mk in mf_set:
                        data[cond][mk] = []
        save_results(data, RESULTS_FILE)
        print(f"Re-running {target_n} runs for: {metric_filter}")
    else:
        target_n = args.runs  # None = infinite

    round_num = 0
    print(f"\nStarting. {'Ctrl+C to stop.' if target_n is None else f'Target: {target_n} runs per condition.'}\n")

    try:
        while True:
            # Sort conditions by count (least data first)
            ordered = sorted(CONDITIONS, key=lambda c: _cond_count(data, c))

            # Check if all conditions have reached target
            if target_n is not None:
                min_count = _cond_count(data, ordered[0])
                if min_count >= target_n:
                    print(f"\nAll conditions have {target_n} runs. Done.")
                    break

            # Process conditions that still need runs
            any_processed = False
            for cond in ordered:
                cc = _cond_count(data, cond)
                if target_n is not None and cc >= target_n:
                    continue

                run_idx = cc  # use the condition's own count as its run index

                scores = _run_condition_forked(cond, run_idx, metric_filter)

                if scores is not None and scores:
                    if cond not in data:
                        data[cond] = {}
                    for metric, value in scores.items():
                        stored = None if value is None else float(value)
                        data[cond].setdefault(metric, []).append(stored)
                    save_results(data, RESULTS_FILE)
                    cc_new = _cond_count(data, cond)
                    print(f"  [{cond:<26s}] run {cc_new:>4d}  ->  {RESULTS_FILE.name}")
                    any_processed = True

                elif scores is not None:
                    # Empty scores (all NaN) - still count as processed
                    print(f"  [{cond:<26s}]          (no valid scores)")
                    any_processed = True

                # else: child crashed, already logged

            if not any_processed:
                print("No conditions were processed this round.")
                break

            round_num += 1
            if round_num % args.print_every == 0:
                cur_n = min(_cond_count(data, c) for c in CONDITIONS)
                print_table(data, cur_n, "MOS 6502 Three-Level Abstraction",
                            CONDITIONS, COMPARISONS)

    except KeyboardInterrupt:
        counts = {c: _cond_count(data, c) for c in CONDITIONS}
        print(f"\nStopped. Per-condition counts: {counts}")
        print(f"Results in {RESULTS_FILE}")

    # Final table
    cur_n = min(_cond_count(data, c) for c in CONDITIONS)
    if cur_n > 0:
        print_table(data, cur_n, "MOS 6502 Three-Level Abstraction",
                    CONDITIONS, COMPARISONS)


if __name__ == "__main__":
    main()
"""
Statistical power / convergence study for the CMOS half-adder.
Requires PySpice / ngspice.
"""
import sys
from pathlib import Path

from utils import load_system

try:
    _tr = load_system("02_transistor_circuit.py", "transistor_circuit")
except SystemExit:
    print("WARNING: transistor_circuit module exited (PySpice not installed?). Skipping.")
    sys.exit(0)

from causal_abstraction import (
    CausalGraph, CoarseGrainingMap, ValueMap, EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel, RectSubspace,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_02_transistor_circuit.json"
N_GRID       = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS  = 100
CONDITIONS   = ["valid", "fail", "noise"]

# Shared objects

_bit_map      = {0: RectSubspace((-0.5, 0.5)), 1: RectSubspace((4.5, 5.5))}
_HA_LOW_LEVEL_MODEL       = _tr.GenericSpiceModel(_tr.half_adder_topology, ["a", "b"])
_HA_INTERNALS = ["nand1_out", "nand2_out", "nand3_out",
                  "n_1", "n_2", "n_3", "n_4", "vdd", "0"]
_HA_CG = CoarseGrainingMap(
    _HA_LOW_LEVEL_MODEL.schema,
    {"a": ["a"], "b": ["b"], "sum": ["sum"], "carry": ["carry"]},
    internal_variables=_HA_INTERNALS,
)
_HA_VM = ValueMap(_HA_CG, {k: _bit_map for k in ["a", "b", "sum", "carry"]})


def _make_high_level_model(sum_fn, carry_fn):
    h = CausalGraph()
    d = {0: 0.5, 1: 0.5}
    h.add_variable("a",     lambda: None,  distribution=d)
    h.add_variable("b",     lambda: None,  distribution=d)
    h.add_variable("sum",   sum_fn,   parents=["a", "b"], distribution=d)
    h.add_variable("carry", carry_fn, parents=["a", "b"], distribution=d)
    return h


def _build_tasks(builder):
    td = TopDownSampler(_HA_VM)
    bu = BottomUpSampler(_HA_VM)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    if cond == "valid":
        high_level_model = _make_high_level_model(lambda a, b: a ^ b, lambda a, b: int(a and b))
        low_level_model = _HA_LOW_LEVEL_MODEL
    elif cond == "fail":
        high_level_model = _make_high_level_model(lambda a, b: int(a or b), lambda a, b: int(a and b))
        low_level_model = _HA_LOW_LEVEL_MODEL
    else:
        high_level_model = _make_high_level_model(lambda a, b: a ^ b, lambda a, b: int(a and b))
        low_level_model = NoisyLowLevelModel(_HA_LOW_LEVEL_MODEL, noise_std=1.5, noise_type="gaussian")
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _HA_VM, _HA_CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=BottomUpSampler(_HA_VM),
        task_kwargs=dict(batch_size=1, max_interventions=2),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Transistor circuit (CMOS half-adder) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

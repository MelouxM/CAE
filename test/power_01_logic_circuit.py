"""
Statistical power / convergence study for the 2-bit logic circuit.
"""
from pathlib import Path

from utils import load_system

_lc = load_system("01_logic_circuit.py", "logic_circuit")

from causal_abstraction import (
    MicroVariableSchema, EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_01_logic_circuit.json"
N_GRID       = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS  = 100
CONDITIONS   = ["valid", "fail", "inv_internal", "noise"]

# Shared objects

_gates, _all_wires = _lc.build_2bit_adder()
_SCHEMA   = MicroVariableSchema.from_names(_all_wires)
_BASE_LOW_LEVEL_MODEL = _lc.NetlistSimulator(_gates, _all_wires)
_CG, _VM  = _lc.build_cg_and_vm(_SCHEMA)

_HIGH_LEVEL_MODELS = {
    "valid":        _lc.build_valid_high_level_model(),
    "fail":         _lc.build_failing_high_level_model(),
    "inv_internal": _lc.build_inverted_internal_high_level_model(),
    "noise":        _lc.build_valid_high_level_model(),
}


def _build_tasks(builder, vm):
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


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _HIGH_LEVEL_MODELS[cond]
    low_level_model  = (NoisyLowLevelModel(_BASE_LOW_LEVEL_MODEL, noise_std=0.45, noise_type="gaussian")
            if cond == "noise" else _BASE_LOW_LEVEL_MODEL)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, _VM),
        analytical=[],
        sampler=BottomUpSampler(_VM),
        task_kwargs=dict(batch_size=10, max_interventions=2),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Logic circuit (2-bit adder) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

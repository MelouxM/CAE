"""
Statistical power / convergence study for Tracr compiled transformers.
"""
from pathlib import Path

from utils import load_system

_tracr = load_system("08_tracr.py", "tracr")

from causal_abstraction import (
    EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_08_tracr.json"
N_GRID       = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS  = 100
CONDITIONS   = ["valid", "fail", "noise"]
SEQ_LEN      = _tracr.SEQ_LEN
INTERV_DOM   = [f"token_{i}" for i in range(SEQ_LEN)]

# Shared objects (compile once)

print("Compiling RASP program (one-time)...")
_COMPILED   = _tracr.build_compiled_model()
print("Done.\n")
_LOW_LEVEL_MODEL        = _tracr.TracrLowLevelModel(_COMPILED, SEQ_LEN)
_, _CG, _VM = _tracr._build_shared_maps()
_HIGH_LEVEL_MODEL_VALID  = _tracr._build_high_level_model(_tracr._make_rank_equation)
_HIGH_LEVEL_MODEL_FAIL   = _tracr._build_high_level_model(_tracr._make_failing_rank_equation)


def _build_tasks(builder):
    s  = TopDownSampler(_VM)
    bu = BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=s,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=s,  name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _HIGH_LEVEL_MODEL_FAIL if cond == "fail" else _HIGH_LEVEL_MODEL_VALID
    low_level_model  = (NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.3, noise_type="gaussian")
            if cond == "noise" else _LOW_LEVEL_MODEL)
    cfg    = EvaluationConfig(metric="hard", seed=seed, n_jobs=1)
    engine = EvaluationEngine(high_level_model=high_level_model, low_level_model=low_level_model, value_map=_VM, cg_map=_CG, config=cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(batch_size=1, max_interventions=SEQ_LEN,
                         intervention_domain=INTERV_DOM),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Tracr (RASP sort-rank transformer) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

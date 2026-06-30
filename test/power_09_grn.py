"""
Statistical power / convergence study for the Segment Polarity GRN.
"""
import os
from pathlib import Path

from utils import load_system

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_grn = load_system("09_grn/grn.py", "grn")

from causal_abstraction import (
    EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_09_grn.json"
N_GRID       = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS  = 100
CONDITIONS   = ["valid", "wrong_map", "wrong_high_level_model", "noise"]
INTERV_VARS  = ["wg_src"]

# Shared objects

print("Loading segment polarity GRN model (one-time)...")
_GRN_DIR = os.path.join(_REPO_ROOT, "systems", "09_grn")
_MODEL   = _grn.SegmentPolarityModel.from_ginml(
    os.path.join(_GRN_DIR, "regulatoryGraph.ginml"))
_SCHEMA  = _grn._base_schema()
_LOW_LEVEL_MODEL     = _grn.SegmentPolarityLowLevelModel(_MODEL)
_NOISY   = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.4, noise_type="gaussian")
print("Done.\n")

_CG_VALID, _VM_VALID = _grn.build_valid_cg_vm(_SCHEMA)
_CG_WRONG, _VM_WRONG = _grn.build_wrong_cg_vm(_SCHEMA)
_HIGH_LEVEL_MODEL_VALID    = _grn.build_valid_high_level_model()
_HIGH_LEVEL_MODEL_REVERSED = _grn.build_reversed_high_level_model()


def _build_tasks(builder, vm):
    s  = TopDownSampler(vm)
    bu = BottomUpSampler(vm)
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
    high_level_model, low_level_model, cg, vm = {
        "valid":     (_HIGH_LEVEL_MODEL_VALID,    _LOW_LEVEL_MODEL,   _CG_VALID, _VM_VALID),
        "wrong_map": (_HIGH_LEVEL_MODEL_VALID,    _LOW_LEVEL_MODEL,   _CG_WRONG, _VM_WRONG),
        "wrong_high_level_model": (_HIGH_LEVEL_MODEL_REVERSED, _LOW_LEVEL_MODEL,   _CG_VALID, _VM_VALID),
        "noise":     (_HIGH_LEVEL_MODEL_VALID,    _NOISY, _CG_VALID, _VM_VALID),
    }[cond]
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, vm),
        analytical=[],
        sampler=TopDownSampler(vm),
        task_kwargs=dict(batch_size=1, max_interventions=1,
                         intervention_domain=INTERV_VARS),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="GRN Segment Polarity (Wg->Fz) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

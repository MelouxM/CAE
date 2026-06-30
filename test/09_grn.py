"""
Full evaluation suite for the Segment Polarity Gene Regulatory Network.
Four conditions: valid, wrong_map, wrong_high_level_model, noise.
"""
import os
from pathlib import Path

from utils import load_system

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_grn = load_system("09_grn/grn.py", "grn")

from causal_abstraction import (
    EvaluationConfig, EvaluationEngine, TopDownSampler, BottomUpSampler,
    NoisyLowLevelModel,
    L2Metric, VarianceDecompositionMetric, MacroscopicInvarianceMetric,
    ComplexityShiftMetric, IIAMetric, BCCMetric, MallowsCpMetric,
    IBLagrangianMetric, CIBLagrangianMetric, InfidelityMetric, SymbionMetric,
    ProbingMetric, RelationalFidelityMetric, SobolSensitivityMetric,
    ConditionalIndependenceMetric, MMDMetric, DCCMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "09_grn_results.json"
N_SAMPLES    = 200
CONDITIONS   = ["valid", "wrong_map", "wrong_high_level_model", "noise"]
OUTPUTS      = ["fz_tgt"]
INTERV_VARS  = ["wg_src"]
COMPARISONS  = [
    ("valid", "wrong_map", "valid -> wrong_map"),
    ("valid", "wrong_high_level_model", "valid -> wrong_high_level_model"),
    ("valid", "noise",     "valid -> noise"),
]

print("Loading segment polarity GRN model (one-time)...")
_GRN_DIR = os.path.join(_REPO_ROOT, 'systems', '09_grn')
_MODEL   = _grn.SegmentPolarityModel.from_ginml(os.path.join(_GRN_DIR, 'regulatoryGraph.ginml'))
_SCHEMA  = _grn._base_schema()
_LOW_LEVEL_MODEL     = _grn.SegmentPolarityLowLevelModel(_MODEL)
_NOISY   = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.4, noise_type='gaussian')
print("Done.\n")

_CG_VALID, _VM_VALID = _grn.build_valid_cg_vm(_SCHEMA)
_CG_WRONG, _VM_WRONG = _grn.build_wrong_cg_vm(_SCHEMA)
_HIGH_LEVEL_MODEL_VALID   = _grn.build_valid_high_level_model()
_HIGH_LEVEL_MODEL_REVERSED = _grn.build_reversed_high_level_model()


def _build_tasks(builder, vm):
    s, bu = TopDownSampler(vm), BottomUpSampler(vm)
    return [
        StandardTasks.score(builder, sampler=s,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=s,  name="CAE_down", include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up", include_faithfulness=True),
        StandardTasks.observational_r2(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_mse( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_kl(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_jsd( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational(builder, L2Metric(),    sampler=bu, output_vars=OUTPUTS, name="L2"),
        StandardTasks.observational(builder, MMDMetric(),   sampler=bu, output_vars=OUTPUTS, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=bu, output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),   sampler=bu, output_vars=OUTPUTS, name="VarDecomp"),
    ]


def _build_analytical():
    return [
        IIAMetric(inner_metric='mse', output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        MacroscopicInvarianceMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        MallowsCpMetric(),
        InfidelityMetric(output_vars=OUTPUTS),
        SymbionMetric(output_vars=OUTPUTS),
        ProbingMetric(n_train=N_SAMPLES, inner_metric='mse'),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        DCCMetric(n_pairs=20, inner_metric='mse'),    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)

    high_level_model, low_level_model, cg, vm = {
        "valid":     (_HIGH_LEVEL_MODEL_VALID,    _LOW_LEVEL_MODEL,   _CG_VALID, _VM_VALID),
        "wrong_map": (_HIGH_LEVEL_MODEL_VALID,    _LOW_LEVEL_MODEL,   _CG_WRONG, _VM_WRONG),
        "wrong_high_level_model": (_HIGH_LEVEL_MODEL_REVERSED, _LOW_LEVEL_MODEL,   _CG_VALID, _VM_VALID),
        "noise":     (_HIGH_LEVEL_MODEL_VALID,    _NOISY, _CG_VALID, _VM_VALID),
    }[cond]

    cfg    = EvaluationConfig(metric='mse', seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, vm),
        analytical=_build_analytical(),
        sampler=TopDownSampler(vm),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1,
                         max_interventions=1, intervention_domain=INTERV_VARS),
    )


if __name__ == '__main__':
    run_suite(
        title="GRN Segment Polarity (Wg->Fz)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
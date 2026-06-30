"""
Full evaluation suite for the Tracr Compiled Transformers system.
Three conditions: valid, fail, noise.
"""
from pathlib import Path

from utils import load_system

_tracr = load_system("08_tracr.py", "tracr")

from causal_abstraction import (
    EvaluationConfig, EvaluationEngine, TopDownSampler, BottomUpSampler,
    NoisyLowLevelModel,
    L2Metric, VarianceDecompositionMetric, MacroscopicInvarianceMetric,
    ComplexityShiftMetric, IIAMetric, BCCMetric, MallowsCpMetric,
    IBLagrangianMetric, CIBLagrangianMetric, ProbingMetric, InfidelityMetric,
    SymbionMetric, RelationalFidelityMetric, SobolSensitivityMetric,
    ConditionalIndependenceMetric, MMDMetric, DCCMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "08_tracr_results.json"
N_SAMPLES    = 100
CONDITIONS   = ["valid", "fail", "noise"]
SEQ_LEN      = _tracr.SEQ_LEN
RANK_VARS    = [f"rank_{i}" for i in range(SEQ_LEN)]
INTERV_DOM   = [f"token_{i}" for i in range(SEQ_LEN)]
OUTPUTS      = RANK_VARS
COMPARISONS  = [("valid", "fail", "valid -> fail"), ("valid", "noise", "valid -> noise")]

print("Compiling RASP program (one-time)...")
_COMPILED = _tracr.build_compiled_model()
print("Done.\n")
_LOW_LEVEL_MODEL = _tracr.TracrLowLevelModel(_COMPILED, SEQ_LEN)
_, _CG, _VM = _tracr._build_shared_maps()
_HIGH_LEVEL_MODEL_VALID = _tracr._build_high_level_model(_tracr._make_rank_equation)
_HIGH_LEVEL_MODEL_FAIL  = _tracr._build_high_level_model(_tracr._make_failing_rank_equation)


def _build_tasks(builder):
    s, bu = TopDownSampler(_VM), BottomUpSampler(_VM)
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
        IIAMetric(inner_metric='hard', output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric='hard'),
        MacroscopicInvarianceMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        MallowsCpMetric(output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        SymbionMetric(output_vars=OUTPUTS),
        ProbingMetric(n_train=N_SAMPLES, inner_metric='hard'),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        DCCMetric(n_pairs=20, inner_metric='hard'),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _HIGH_LEVEL_MODEL_FAIL if cond == "fail" else _HIGH_LEVEL_MODEL_VALID
    low_level_model  = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.3, noise_type='gaussian') if cond == "noise" else _LOW_LEVEL_MODEL

    cfg    = EvaluationConfig(metric='hard', seed=seed, n_jobs=1)
    engine = EvaluationEngine(high_level_model=high_level_model, low_level_model=low_level_model, value_map=_VM, cg_map=_CG, config=cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=_build_analytical(),
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1,
                         max_interventions=SEQ_LEN, intervention_domain=INTERV_DOM),
    )


if __name__ == '__main__':
    run_suite(
        title="Tracr (RASP sort-rank transformer)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
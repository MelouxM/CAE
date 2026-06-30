"""
Full evaluation suite for the logic circuit (2-bit adder) system.

Four conditions: valid, fail, inv_internal, noise.
"""
from pathlib import Path

from utils import load_system

_lc = load_system("01_logic_circuit.py", "logic_circuit")

from causal_abstraction import (
    MicroVariableSchema, EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
    VarianceDecompositionMetric, CIBLagrangianMetric,
    IBLagrangianMetric, ComplexityShiftMetric,
    SobolSensitivityMetric, IIAMetric, BCCMetric,
    ProbingMetric, InfidelityMetric, SymbionMetric, MMDMetric,
    ConditionalIndependenceMetric, RelationalFidelityMetric,
    MacroscopicInvarianceMetric, DCCMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "01_logic_circuit_results.json"
N_SAMPLES    = 1000
CONDITIONS   = ["valid", "fail", "inv_internal", "noise"]
OUTPUTS      = ["Result_Sum", "Result_Carry"]
COMPARISONS  = [
    ("valid", "fail",         "valid -> fail"),
    ("valid", "inv_internal", "valid -> inv_internal"),
    ("valid", "noise",        "valid -> noise"),
]

# Shared objects (built once)
_gates, _all_wires = _lc.build_2bit_adder()
_SCHEMA  = MicroVariableSchema.from_names(_all_wires)
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
        StandardTasks.score(builder, sampler=td,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu,  name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down", include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up", include_faithfulness=True),
        StandardTasks.observational_r2(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_mse( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_kl(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_jsd( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational(builder, MMDMetric(),                     sampler=bu, output_vars=OUTPUTS, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=bu, output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),   sampler=bu, output_vars=OUTPUTS, name="VarDecomp"),
    ]


def _build_analytical():
    return [
        IIAMetric(inner_metric='mse', output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        ProbingMetric(n_train=N_SAMPLES, inner_metric='mse'),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        SymbionMetric(output_vars=OUTPUTS),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        MacroscopicInvarianceMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        DCCMetric(n_pairs=50, inner_metric='hard'),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _HIGH_LEVEL_MODELS[cond]
    low_level_model  = NoisyLowLevelModel(_BASE_LOW_LEVEL_MODEL, noise_std=0.4, noise_type='gaussian') if cond == "noise" else _BASE_LOW_LEVEL_MODEL

    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, _VM),
        analytical=_build_analytical(),
        sampler=BottomUpSampler(_VM),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=10, max_interventions=2),
    )


if __name__ == '__main__':
    run_suite(
        title="Logic circuit",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=10,
    )
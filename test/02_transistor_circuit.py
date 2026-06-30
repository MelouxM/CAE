"""
Full evaluation suite for the Transistor Circuit (CMOS half-adder).
Three conditions: valid, fail, noise.
"""
from pathlib import Path

from utils import load_system

_tr = load_system("02_transistor_circuit.py", "transistor_circuit")

from causal_abstraction import (
    CausalGraph, CoarseGrainingMap, ValueMap, EvaluationConfig, EvaluationEngine,
    TopDownSampler, BottomUpSampler, RectSubspace, NoisyLowLevelModel,
    VarianceDecompositionMetric, CIBLagrangianMetric,
    IBLagrangianMetric, ComplexityShiftMetric,
    SobolSensitivityMetric, IIAMetric, BCCMetric,
    ProbingMetric, InfidelityMetric, SymbionMetric, MMDMetric,
    ConditionalIndependenceMetric, RelationalFidelityMetric, MacroscopicInvarianceMetric,
    DCCMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "02_transistor_circuit_results.json"
N_SAMPLES    = 250
CONDITIONS   = ["valid", "fail", "noise"]
OUTPUTS      = ["sum", "carry"]
COMPARISONS  = [
    ("valid", "fail",  "valid -> fail"),
    ("valid", "noise", "valid -> noise"),
]

# Shared
bit_map = {0: RectSubspace((-0.5, 0.5)), 1: RectSubspace((4.5, 5.5))}
HA_LOW_LEVEL_MODEL  = _tr.GenericSpiceModel(_tr.half_adder_topology, ['a', 'b'])
ha_internals = [
    "nand1_out", "nand2_out", "nand3_out",
    "n_1", "n_2", "n_3", "n_4", "vdd", "0",
]
HA_CG = CoarseGrainingMap(
    HA_LOW_LEVEL_MODEL.schema,
    {"a": ["a"], "b": ["b"], "sum": ["sum"], "carry": ["carry"]},
    internal_variables=ha_internals,
)
HA_VM = ValueMap(HA_CG, {k: bit_map for k in ["a", "b", "sum", "carry"]})


def _make_high_level_model(sum_fn, carry_fn):
    h = CausalGraph()
    h.add_variable("a",     lambda: None, distribution={0: 0.5, 1: 0.5})
    h.add_variable("b",     lambda: None, distribution={0: 0.5, 1: 0.5})
    h.add_variable("sum",   sum_fn,   parents=["a", "b"], distribution={0: 0.5, 1: 0.5})
    h.add_variable("carry", carry_fn, parents=["a", "b"], distribution={0: 0.5, 1: 0.5})
    return h


def _build_tasks(builder):
    td = TopDownSampler(HA_VM)
    bu = BottomUpSampler(HA_VM)
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
        ProbingMetric(inner_metric='mse'),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        SymbionMetric(output_vars=OUTPUTS),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        MacroscopicInvarianceMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        DCCMetric(n_pairs=5, inner_metric='mse'),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)

    if cond == "valid":
        high_level_model = _make_high_level_model(lambda a, b: a ^ b, lambda a, b: int(a and b))
        low_level_model = HA_LOW_LEVEL_MODEL
    elif cond == "fail":
        high_level_model = _make_high_level_model(lambda a, b: int(a or b), lambda a, b: int(a and b))
        low_level_model = HA_LOW_LEVEL_MODEL
    else:  # noise
        high_level_model = _make_high_level_model(lambda a, b: a ^ b, lambda a, b: int(a and b))
        low_level_model = NoisyLowLevelModel(HA_LOW_LEVEL_MODEL, noise_std=1.5, noise_type='gaussian')

    cfg    = EvaluationConfig(metric='mse', seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, HA_VM, HA_CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=_build_analytical(),
        sampler=BottomUpSampler(HA_VM),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1, max_interventions=2),
    )


if __name__ == '__main__':
    run_suite(
        title="Transistor (CMOS half-adder)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=10,
    )
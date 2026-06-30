"""
Full evaluation suite for the Ising Model.
Three conditions: valid, fail, noise.
"""
from pathlib import Path

from utils import load_system

_ising = load_system("07_ising_model.py", "ising_model")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, ContinuousValueMap,
    EvaluationEngine, EvaluationConfig, TopDownSampler, BottomUpSampler,
    RectSubspace, FullSubspace, NoisyLowLevelModel,
    L2Metric, VarianceDecompositionMetric, MacroscopicInvarianceMetric,
    ComplexityShiftMetric, IIAMetric, BCCMetric, MallowsCpMetric,
    IBLagrangianMetric, CIBLagrangianMetric, InfidelityMetric, ProbingMetric,
    RelationalFidelityMetric, SobolSensitivityMetric, ConditionalIndependenceMetric,
    MMDMetric, RMSEMetric,
)
from causal_abstraction.analytical_metrics import (
    StructuralDeviationMetric, CausalSensitivityIndexMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "07_ising_model_results.json"
N_SAMPLES    = 30
CONDITIONS   = ["valid", "fail", "noise"]
OUTPUTS      = ["PredictedMagnetization"]
COMPARISONS  = [("valid", "fail", "valid -> fail"), ("valid", "noise", "valid -> noise")]

GRID_SIDE    = 8
HIGH_LEVEL_MODEL_SWEEPS   = (500, 200)

_D_TEMP  = RectSubspace((0.5, 4.0))
_D_FIELD = RectSubspace((-0.5, 0.5))
_D_MAG   = FullSubspace(1)
_SCHEMA  = MicroVariableSchema.from_names(["Temperature", "ExternalField", "PredictedMagnetization"])
_CG      = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM      = ContinuousValueMap(_CG, {
    "Temperature": {0: _D_TEMP}, "ExternalField": {0: _D_FIELD},
    "PredictedMagnetization": {0: _D_MAG},
})
_LOW_LEVEL_MODEL = _ising.MolecularDynamicsIsingModel(grid_side=GRID_SIDE, vibrational_temp=0.0)
_LOW_LEVEL_MODEL.params['equil_sweeps'] = HIGH_LEVEL_MODEL_SWEEPS[0]
_LOW_LEVEL_MODEL.params['measure_sweeps'] = HIGH_LEVEL_MODEL_SWEEPS[1]


def _make_high_level_model(J):
    h = CausalGraph()
    h.add_variable("Temperature", lambda: None, domain=_D_TEMP)
    h.add_variable("ExternalField", lambda: None, domain=_D_FIELD)
    def predict_mag(Temperature, ExternalField):
        seed = int(round(Temperature, 8) * 1e8) + int(round(ExternalField + 10, 8) * 1e8)
        return _ising._run_rigid_simulation(GRID_SIDE, Temperature, ExternalField, J,
                                            HIGH_LEVEL_MODEL_SWEEPS[0], HIGH_LEVEL_MODEL_SWEEPS[1], seed)
    h.add_variable("PredictedMagnetization", predict_mag,
                   parents=["Temperature", "ExternalField"], domain=_D_MAG)
    return h


def _build_tasks(builder):
    s, bu = TopDownSampler(_VM), BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=s, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=s, name="CAE_down", include_faithfulness=True),
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
        StandardTasks.observational(builder, RMSEMetric(),                    sampler=bu, output_vars=OUTPUTS, name="RMSE"),
    ]


def _build_analytical():
    mk = lambda p: _make_high_level_model(p["J"])
    return [
        IIAMetric(inner_metric='mse', output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        MacroscopicInvarianceMetric(n_pairs=min(N_SAMPLES, 5), inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        MallowsCpMetric(n_params=1, output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        ProbingMetric(n_train=max(N_SAMPLES, 80), inner_metric='mse'),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        StructuralDeviationMetric(param_names=["J"], nominal_params={"J": 1.0},
                                  make_high_level_model=mk, inner_metric="mse"),
        CausalSensitivityIndexMetric(param_names=["J"], nominal_params={"J": 1.0},
                                     make_high_level_model=mk, inner_metric="mse"),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _make_high_level_model(J=2.0 if cond == "fail" else 1.0)
    low_level_model  = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.05, noise_type='gaussian') if cond == "noise" else _LOW_LEVEL_MODEL

    cfg    = EvaluationConfig(metric='mse', seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=_build_analytical(),
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1, max_interventions=2,
                         intervention_domain=["Temperature", "ExternalField"]),
    )


if __name__ == '__main__':
    run_suite(
        title="Ising Model (rigid vs MD)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
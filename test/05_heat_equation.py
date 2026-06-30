"""
Full evaluation suite for the 1D Heat Equation (Brownian particle diffusion).
Three conditions: valid, fail, noise.
"""
from pathlib import Path

from utils import load_system

_h = load_system("05_heat_equation.py", "heat_equation")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, FullSubspace,
    EvaluationEngine, EvaluationConfig, NoisyLowLevelModel,
    BottomUpSampler,
    L2Metric, TrajectoryMSEMetric, DTWMetric, TemporalAutocorrelationMetric,
    SpectralMetric, VarianceDecompositionMetric, MacroscopicInvarianceMetric,
    ComplexityShiftMetric, IIAMetric, BCCMetric, MallowsCpMetric,
    IBLagrangianMetric, CIBLagrangianMetric, InfidelityMetric, ProbingMetric,
    RelationalFidelityMetric, SobolSensitivityMetric, ConditionalIndependenceMetric, MMDMetric,
)
from causal_abstraction.analytical_metrics import (
    StructuralDeviationMetric, CausalSensitivityIndexMetric, DCCMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "05_heat_equation_results.json"
N_SAMPLES    = 30
CONDITIONS   = ["valid", "fail", "noise"]
OUTPUTS      = ["T_final"]
COMPARISONS  = [("valid", "fail", "valid -> fail"), ("valid", "noise", "valid -> noise")]

L, ALPHA, T_MAX, STEPS, BINS, N_PART = 1.0, 0.1, 0.2, 200, 50, 1000
DT = T_MAX / STEPS

_LOW_LEVEL_MODEL = _h.BrownianParticleSystem(n_particles=N_PART, n_steps=STEPS, diff_coeff=ALPHA, box_len=L, dt=DT)
_SCHEMA = MicroVariableSchema.from_names(["T_initial", "T_final"])
_CG = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM = _h.BinningValueMap(_CG, {"T_initial": {0: FullSubspace(BINS)}, "T_final": {0: FullSubspace(BINS)}},
                         n_bins=BINS, box_len=L, n_particles=N_PART)
_SAMPLER = _h.SmoothProfileSampler(_VM, BINS)


def _make_high_level_model(diff_coeff):
    solver = _h.HeatEquationSolver(n_bins=BINS, diff_coeff=diff_coeff, box_len=L, dt=DT, n_steps=STEPS)
    h = CausalGraph()
    h.add_variable("T_initial", lambda: None, domain=FullSubspace(BINS))
    h.add_variable("T_final", lambda T_initial: solver.solve(T_initial),
                   parents=["T_initial"], domain=FullSubspace(BINS))
    return h


def _build_tasks(builder):
    s  = _SAMPLER
    bu = BottomUpSampler(_VM)
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
        StandardTasks.observational(builder, L2Metric(),          sampler=bu, output_vars=OUTPUTS, name="L2"),
        StandardTasks.observational(builder, TrajectoryMSEMetric(), sampler=bu, output_vars=OUTPUTS, name="TrajMSE"),
        StandardTasks.observational(builder, DTWMetric(normalize_length=True), sampler=bu, output_vars=OUTPUTS, name="DTW"),
        StandardTasks.observational(builder, TemporalAutocorrelationMetric(),  sampler=bu, output_vars=OUTPUTS, name="Autocorr"),
        StandardTasks.observational(builder, SpectralMetric(normalize_psd=True), sampler=bu, output_vars=OUTPUTS, name="Spectral"),
        StandardTasks.observational(builder, MMDMetric(),                     sampler=bu, output_vars=OUTPUTS, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=bu, output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),   sampler=bu, output_vars=OUTPUTS, name="VarDecomp"),
    ]


def _build_analytical():
    mk = lambda p: _make_high_level_model(p["ALPHA"])
    return [
        IIAMetric(inner_metric='mse', output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        DCCMetric(n_pairs=N_SAMPLES, inner_metric='mse'),
        MacroscopicInvarianceMetric(n_pairs=min(N_SAMPLES, 5), inner_metric='mse'),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric='mse'),
        MallowsCpMetric(n_params=1, output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        ProbingMetric(n_train=max(N_SAMPLES, 80), inner_metric='mse'),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        StructuralDeviationMetric(param_names=["ALPHA"], nominal_params={"ALPHA": ALPHA}, make_high_level_model=mk, inner_metric="mse"),
        CausalSensitivityIndexMetric(param_names=["ALPHA"], nominal_params={"ALPHA": ALPHA}, make_high_level_model=mk, inner_metric="mse"),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _make_high_level_model(ALPHA if cond != "fail" else 0.0)
    low_level_model  = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.15, noise_type='gaussian') if cond == "noise" else _LOW_LEVEL_MODEL

    cfg    = EvaluationConfig(metric='mse', seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=_build_analytical(),
        sampler=_SAMPLER,
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1, max_interventions=1,
                         intervention_domain=["T_initial"]),
    )


if __name__ == '__main__':
    run_suite(
        title="Heat Equation 1D (Brownian diffusion)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
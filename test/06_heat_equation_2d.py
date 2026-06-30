"""
Full evaluation suite for the 2D Heat Equation (phonon lattice).
Three conditions: valid, fail, noise.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from utils import load_system

_h2d = load_system("06_heat_equation_2d.py", "heat_equation_2d")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap,
    EvaluationEngine, EvaluationConfig, TopDownSampler, BottomUpSampler,
    RectSubspace, FullSubspace, NoisyLowLevelModel,
    L2Metric, TrajectoryMSEMetric, DTWMetric, TemporalAutocorrelationMetric,
    SpectralMetric, VarianceDecompositionMetric, MacroscopicInvarianceMetric,
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

RESULTS_FILE = Path(__file__).parent / "results" / "06_heat_equation_2d_results.json"
N_SAMPLES    = 5
CONDITIONS   = ["valid", "fail", "noise"]
OUTPUTS      = ["final_temp_map"]
COMPARISONS  = [("valid", "fail", "valid -> fail"), ("valid", "noise", "valid -> noise")]

GRID_SIZE, SIM_TIME, P_SCATTER, N_AVG = 16, 5.0, 0.2, 10
MAP_SIGMA = max(0.1, GRID_SIZE * 0.05)

_SRC_X, _SRC_Y = RectSubspace((0.3, 0.7)), RectSubspace((0.3, 0.7))
_SRC_E = RectSubspace((500.0, 1000.0))
_OUT_T = FullSubspace(GRID_SIZE * GRID_SIZE)
_SCHEMA = MicroVariableSchema.from_names(["source_x", "source_y", "source_E", "final_temp_map"])
_CG = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM = _h2d.SpatialValueMap(_CG,
    {"source_x": {0: _SRC_X}, "source_y": {0: _SRC_Y},
     "source_E": {0: _SRC_E}, "final_temp_map": {0: _OUT_T}},
    (GRID_SIZE, GRID_SIZE))

_LOW_LEVEL_MODEL = _h2d.PhononLatticeModel(grid_h=GRID_SIZE, grid_w=GRID_SIZE,
                                p_scatter=P_SCATTER, sim_time=SIM_TIME, n_averages=N_AVG)

print("Calibrating 2D heat equation alpha (one-time)...")
_ALPHA = _h2d.calibrate_alpha_opt(GRID_SIZE, P_SCATTER, SIM_TIME, MAP_SIGMA, n_calib_averages=N_AVG)
print(f"Calibrated alpha={_ALPHA:.6f}\n")


def _make_high_level_model(alpha):
    h = CausalGraph()
    h.add_variable("source_x", lambda: None, domain=_SRC_X)
    h.add_variable("source_y", lambda: None, domain=_SRC_Y)
    h.add_variable("source_E", lambda: None, domain=_SRC_E)
    def predict(source_x, source_y, source_E):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        grid[int(source_y * GRID_SIZE), int(source_x * GRID_SIZE)] = 1.0
        grid = gaussian_filter(grid, sigma=MAP_SIGMA)
        final = _h2d.solve_heat_pde_neumann(grid, alpha, dt=0.2, steps=int(SIM_TIME / 0.2))
        t = np.sum(final)
        return final / t if t > 1e-9 else final
    h.add_variable("final_temp_map", predict,
                   parents=["source_x", "source_y", "source_E"], domain=_OUT_T)
    return h


def _build_tasks(builder):
    td = TopDownSampler(_VM)
    bu = BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down", include_faithfulness=True),
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
        StandardTasks.observational(builder, RMSEMetric(),                    sampler=bu, output_vars=OUTPUTS, name="RMSE"),
    ]


def _build_analytical():
    mk = lambda p: _make_high_level_model(p["alpha"])
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
        StructuralDeviationMetric(param_names=["alpha"], nominal_params={"alpha": _ALPHA},
                                  make_high_level_model=mk, inner_metric="mse"),
        CausalSensitivityIndexMetric(param_names=["alpha"], nominal_params={"alpha": _ALPHA},
                                     make_high_level_model=mk, inner_metric="mse"),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed  = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    alpha = _ALPHA * 10.0 if cond == "fail" else _ALPHA
    high_level_model   = _make_high_level_model(alpha)
    low_level_model   = NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=50.0, noise_type='gaussian') if cond == "noise" else _LOW_LEVEL_MODEL

    cfg    = EvaluationConfig(metric='mse', seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=_build_analytical(),
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1, max_interventions=3,
                         intervention_domain=["source_x", "source_y"]),
    )


if __name__ == '__main__':
    run_suite(
        title="Heat Equation 2D (phonon lattice)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
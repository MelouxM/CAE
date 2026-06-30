"""
Full evaluation suite for the Predator-Prey (Lotka-Volterra vs ABM) system.
Seven conditions: valid, fail_alpha, spatial, stochastic, aging, noise, complex.
"""
from pathlib import Path

import numpy as np

from utils import load_system

_pp = load_system("04_predator_prey.py", "predator_prey")

from causal_abstraction import (
    BCCMetric,
    BottomUpSampler,
    CausalGraph,
    CoarseGrainingMap,
    ContinuousValueMap,
    EvaluationConfig,
    EvaluationEngine,
    FullSubspace,
    IIAMetric,
    IBLagrangianMetric,
    CIBLagrangianMetric,
    InfidelityMetric,
    ComplexityShiftMetric,
    ConditionalIndependenceMetric,
    MacroscopicInvarianceMetric,
    MicroVariableSchema,
    MMDMetric,
    NoisyLowLevelModel,
    ProbingMetric,
    RectSubspace,
    RelationalFidelityMetric,
    SobolSensitivityMetric,
    TopDownSampler,
    VarianceDecompositionMetric,
    L2Metric,
)
from causal_abstraction.analytical_metrics import (
    AnalyticalMetric,
    CausalSensitivityIndexMetric,
    MallowsCpMetric,
    StructuralDeviationMetric,
)
from causal_abstraction.metrics import (
    RMSEMetric,
    SpectralMetric,
    TemporalAutocorrelationMetric,
    DTWMetric,
    TrajectoryMSEMetric,
)
from causal_abstraction.primitives import SystemState
from causal_abstraction.schema import Variable
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite


RESULTS_FILE  = Path(__file__).parent / "results" / "04_predator_prey_results.json"
N_SAMPLES     = 50
N_TRAJ_EPOCHS = 5
CONDITIONS    = ["valid", "fail_alpha", "spatial", "stochastic", "aging", "noise", "complex"]
OUTPUTS       = ["final_populations"]
COMPARISONS   = [
    ("valid", "fail_alpha", "valid -> wrong alpha"),
    ("valid", "spatial",    "valid -> spatial"),
    ("valid", "stochastic", "valid -> stochastic"),
    ("valid", "aging",      "valid -> aging"),
    ("valid", "noise",      "valid -> noise"),
    ("valid", "complex",    "valid -> complex"),
]

_BASE_CONFIG = {
    "prey_reproduce_prob":      0.1,
    "predation_prob":           0.001,
    "predator_reproduce_prob":  0.5,
    "predator_starvation_prob": 0.05,
    "num_steps":                50,
    "grid_size":                20,
    "prey_age_death_prob":      0.02,
    "predator_age_death_prob":  0.04,
}

_IDEAL_CONFIG = dict(
    _BASE_CONFIG,
    spatial=False,
    stochastic_reproduction=False,
    agent_aging=False,
)
_LOW_LEVEL_MODEL_IDEAL = _pp.AgentBasedModel(_IDEAL_CONFIG)

_SEARCH_SPACE = {
    "alpha": np.linspace(0.08, 0.12, 3),
    "beta":  np.linspace(0.0008, 0.0012, 3),
    "delta": np.linspace(0.0004, 0.0006, 3),
    "gamma": np.linspace(0.04, 0.06, 3),
}
_HIGH_LEVEL_MODEL_PARAMS = _pp.calibrate_high_level_model(_LOW_LEVEL_MODEL_IDEAL, _SEARCH_SPACE)

_PREY_DOM = RectSubspace((50, 250))
_PRED_DOM = RectSubspace((20, 100))
_OUT_DOM  = FullSubspace(2)

_SCHEMA = MicroVariableSchema(
    [Variable(x) for x in ["prey_t", "predator_t", "final_populations"]]
)
_CG = CoarseGrainingMap(
    _SCHEMA,
    {k: [k] for k in _SCHEMA.variable_names},
)
_VM = ContinuousValueMap(_CG, {
    "prey_t":            {0: _PREY_DOM},
    "predator_t":        {0: _PRED_DOM},
    "final_populations": {0: _OUT_DOM},
})


def _make_high_level_model(params):
    g = CausalGraph()
    g.add_variable("prey_t", lambda: None, domain=_PREY_DOM)
    g.add_variable("predator_t", lambda: None, domain=_PRED_DOM)
    g.add_variable(
        "final_populations",
        lambda prey_t, predator_t: _pp.solve_lotka_volterra(prey_t, predator_t, params),
        parents=["prey_t", "predator_t"],
        domain=_OUT_DOM,
    )
    return g


def _make_low_level_model(spatial, stochastic, aging):
    config = dict(
        _BASE_CONFIG,
        spatial=spatial,
        stochastic_reproduction=stochastic,
        agent_aging=aging,
    )
    return _pp.AgentBasedModel(config)


# Trajectory helpers

def _lv_trajectory(prey_0, pred_0, params, n_epochs):
    traj = [[prey_0, pred_0]]
    prey, pred = float(prey_0), float(pred_0)
    for _ in range(n_epochs):
        out = _pp.solve_lotka_volterra(prey, pred, params)
        prey, pred = float(out[0]), float(out[1])
        traj.append([prey, pred])
    return np.array(traj, dtype=float)


def _abm_trajectory(low_level_model, prey_0, pred_0, n_epochs):
    traj  = [[prey_0, pred_0]]
    state = SystemState(
        values={"prey_t": np.array([prey_0]), "predator_t": np.array([pred_0])}
    )
    for _ in range(n_epochs):
        if hasattr(low_level_model, "step"):
            state = low_level_model.step(state)
        else:
            state = low_level_model.forward_with_interventions(
                SystemState(),
                {
                    "prey_t":     state.values["prey_t"],
                    "predator_t": state.values["predator_t"],
                },
            )
            pops = np.asarray(state.values["final_populations"]).ravel()
            state.values["prey_t"]     = np.array([float(pops[0])])
            state.values["predator_t"] = np.array([float(pops[1]) if len(pops) > 1 else 0.0])

        pops = np.asarray(state.values["final_populations"]).ravel()
        traj.append([
            float(pops[0]),
            float(pops[1]) if len(pops) > 1 else 0.0,
        ])
    return np.array(traj, dtype=float)


# Base class for trajectory-based metrics

class _PPTemporalMetric(AnalyticalMetric):
    def __init__(self, high_level_model_params, n_epochs=N_TRAJ_EPOCHS, normalize=True):
        super().__init__(normalize=normalize)
        self.high_level_model_params  = high_level_model_params
        self.n_epochs    = n_epochs
        self._ref_scale  = None

    def _score_trajectories(self, lv, abm):
        s = self._smooth(abm, self.smooth_window)
        if len(s) < 2:
            return float('nan')
        X_dot = np.diff(s, axis=0)
        X_lib = s[:-1]
        alive = (X_lib[:, 0] > 1.0) & (X_lib[:, 1] > 1.0)
        if not np.any(alive):
            return float('nan')
        X_dot = X_dot[alive]
        X_lib = X_lib[alive]
        Theta = np.column_stack([X_lib[:, 0], X_lib[:, 1], X_lib[:, 0] * X_lib[:, 1]])
        alpha = self.high_level_model_params['alpha']
        beta = self.high_level_model_params['beta']
        delta = self.high_level_model_params['delta']
        gamma = self.high_level_model_params['gamma']
        xi = np.array([[alpha, 0.0, -beta], [0.0, -gamma, delta]]).T
        X_dot_pred = Theta @ xi
        residual = X_dot - X_dot_pred
        # Store ref scale for normalization: variance of actual low-level model derivatives
        self._ref_scale = float(np.mean(np.var(X_dot, axis=0))) + 1e-9
        return float(np.mean(residual ** 2))

    def _compute_ref_scale(self, lv):
        # lv is the high-level model trajectory, but we want the low-level model derivative variance
        # This is computed inside _score_trajectories; override normalize instead
        return 1.0  # defer normalization to _normalize_result

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        rng = np.random.default_rng(config.seed if config else None)
        scores, refs = [], []
        for _ in range(n_samples):
            p0  = float(rng.uniform(50, 250))
            pr0 = float(rng.uniform(20, 100))
            lv  = _lv_trajectory(p0, pr0, self.high_level_model_params, self.n_epochs)
            abm = _abm_trajectory(low_level_model, p0, pr0, self.n_epochs)
            refs.append(self._compute_ref_scale(lv))
            try:
                s = self._score_trajectories(lv, abm)
                if not np.isnan(s):
                    scores.append(s)
            except Exception:
                pass
        self._ref_scale = float(np.mean(refs)) if refs else None
        return float(np.mean(scores)) if scores else float("nan")


# Concrete trajectory metrics

class TrajectoryMSEPP(_PPTemporalMetric):
    def _score_trajectories(self, lv, abm):
        return TrajectoryMSEMetric(normalize=False)._measure(lv, abm)

    def _normalize_result(self, r):
        if np.isnan(r):
            return r
        return float(np.tanh(r / (self._ref_scale or 1.0)))


class DTWMetricPP(_PPTemporalMetric):
    def _compute_ref_scale(self, lv):
        return float(np.mean(np.linalg.norm(np.diff(lv, axis=0), axis=1))) + 1e-9

    def _score_trajectories(self, lv, abm):
        return DTWMetric(normalize_length=True, normalize=False)._measure(lv, abm)

    def _normalize_result(self, r):
        if np.isnan(r):
            return r
        return float(np.tanh(r / (self._ref_scale or 1.0)))


class AutocorrelationPP(_PPTemporalMetric):
    def _compute_ref_scale(self, lv):
        return 1.0

    def _score_trajectories(self, lv, abm):
        n = max(2, self.n_epochs // 2)
        m = TemporalAutocorrelationMetric(n_lags=n, normalize=False)
        prey_score = m._measure(lv[:, 0], abm[:, 0])
        pred_score = m._measure(lv[:, 1], abm[:, 1])
        return float(np.mean([prey_score, pred_score]))

    def _normalize_result(self, r):
        if np.isnan(r):
            return r
        return float(np.tanh(r))


class SpectralPP(_PPTemporalMetric):
    def _compute_ref_scale(self, lv):
        return 1.0

    def _score_trajectories(self, lv, abm):
        m = SpectralMetric(normalize_psd=True, normalize=False)
        prey_score = m._measure(lv[:, 0], abm[:, 0])
        pred_score = m._measure(lv[:, 1], abm[:, 1])
        return float(np.mean([prey_score, pred_score]))

    def _normalize_result(self, r):
        if np.isnan(r):
            return r
        return float(np.tanh(r))


class SINDyValidationPP(_PPTemporalMetric):
    """
    SINDy-style validation: measures how well the LV high-level model predicts the ABM's
    macro-step trajectory changes.

    For each consecutive pair of smoothed trajectory points (X[t], X[t+1]):
      observed change: X_dot[t] = X[t+1] - X[t]
      LV prediction:   X_dot_pred[t] = solve_lv(X[t]) - X[t]

    score = ||X_dot - X_dot_pred||_F / ||X_dot||_F

    This is normalized to [0, 1) via tanh. Score near 0 = ABM follows LV well.
    """

    def __init__(self, high_level_model_params, n_epochs=N_TRAJ_EPOCHS,
                 smooth_window=3, normalize=True):
        super().__init__(high_level_model_params, n_epochs, normalize)
        self.smooth_window = smooth_window

    @staticmethod
    def _smooth(t, w):
        if w <= 1 or len(t) < w:
            return t
        k = np.ones(w) / w
        return np.stack(
            [np.convolve(t[:, c], k, mode='valid')
             for c in range(t.shape[1])], axis=1)

    def _compute_ref_scale(self, lv):
        return 1.0  # normalization handled in _normalize_result

    def _score_trajectories(self, lv, abm):
        # Don't smooth: we compare full-epoch changes to full-epoch LV predictions,
        # not ODE derivatives. Smoothing averaged non-states (e.g. mean of ABM[0..2])
        # into LV starting points, breaking the comparison and inflating valid-case error.
        if len(abm) < 2:
            return float('nan')

        X     = abm[:-1]               # actual ABM starting states, (n_epochs, 2)
        X_dot = np.diff(abm, axis=0)   # actual epoch changes, (n_epochs, 2)

        alive = (X[:, 0] > 1.0) & (X[:, 1] > 1.0)
        if not np.any(alive):
            return float('nan')

        X_dot = X_dot[alive]
        X     = X[alive]

        X_dot_pred = np.zeros_like(X_dot)
        for i in range(len(X)):
            lv_out = _pp.solve_lotka_volterra(
                float(X[i, 0]), float(X[i, 1]), self.high_level_model_params
            )
            X_dot_pred[i, 0] = float(lv_out[0]) - float(X[i, 0])
            X_dot_pred[i, 1] = float(lv_out[1]) - float(X[i, 1])

        residual = X_dot - X_dot_pred
        denom = float(np.linalg.norm(X_dot, 'fro'))
        if denom < 1e-10:
            return float('nan')

        return float(np.linalg.norm(residual, 'fro')) / denom


# Task and metric builders

def _build_tasks(builder, vm):
    td = TopDownSampler(vm)
    bu = BottomUpSampler(vm)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down", include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up", include_faithfulness=True),
        StandardTasks.observational_r2(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_mse( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_kl(  builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational_jsd( builder, sampler=bu, output_vars=OUTPUTS),
        StandardTasks.observational(builder, RMSEMetric(), sampler=bu, output_vars=OUTPUTS, name="RMSE"),
        StandardTasks.observational(builder, MMDMetric(),  sampler=bu, output_vars=OUTPUTS, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=bu, output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),   sampler=bu, output_vars=OUTPUTS, name="VarDecomp"),
        StandardTasks.observational(builder, L2Metric(), sampler=bu, output_vars=OUTPUTS, name="L2"),
    ]


def _build_analytical(cond_params):
    nominal = {k: cond_params[k] for k in ("alpha", "beta", "delta", "gamma")}

    def make_high_level_model(p):
        return _make_high_level_model(dict(cond_params, **p))

    return [
        IIAMetric(inner_metric="l2", output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric="l2"),
        MacroscopicInvarianceMetric(n_pairs=min(N_SAMPLES, 10), inner_metric="l2"),
        RelationalFidelityMetric(output_vars=OUTPUTS, n_pairs=200),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        ComplexityShiftMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        InfidelityMetric(output_vars=OUTPUTS),
        ProbingMetric(n_train=max(N_SAMPLES * 4, 80), inner_metric="l2"),
        MallowsCpMetric(n_params=4, output_vars=OUTPUTS),
        StructuralDeviationMetric(
            param_names=list(nominal),
            nominal_params=nominal,
            make_high_level_model=make_high_level_model,
            inner_metric="mse",
        ),
        CausalSensitivityIndexMetric(
            param_names=list(nominal),
            nominal_params=nominal,
            make_high_level_model=make_high_level_model,
            inner_metric="mse",
        ),
        TrajectoryMSEPP(cond_params,  n_epochs=N_TRAJ_EPOCHS),
        DTWMetricPP(cond_params,      n_epochs=N_TRAJ_EPOCHS),
        AutocorrelationPP(cond_params, n_epochs=N_TRAJ_EPOCHS),
        SpectralPP(cond_params,       n_epochs=N_TRAJ_EPOCHS),
        SINDyValidationPP(cond_params, n_epochs=N_TRAJ_EPOCHS),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    if cond == "fail_alpha":
        cond_params = dict(_HIGH_LEVEL_MODEL_PARAMS, alpha=_HIGH_LEVEL_MODEL_PARAMS["alpha"] * 1.5)
        low_level_model = _LOW_LEVEL_MODEL_IDEAL
    else:
        cond_params = _HIGH_LEVEL_MODEL_PARAMS
        low_level_model = {
            "valid":      _LOW_LEVEL_MODEL_IDEAL,
            "spatial":    _make_low_level_model(True, False, False),
            "stochastic": _make_low_level_model(False, True, False),
            "aging":      _make_low_level_model(False, False, True),
            "noise":      NoisyLowLevelModel(_LOW_LEVEL_MODEL_IDEAL, noise_std=5.0, noise_type="gaussian"),
            "complex":    _make_low_level_model(True, True, True),
        }[cond]

    high_level_model    = _make_high_level_model(cond_params)
    seed   = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)

    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, _VM),
        analytical=_build_analytical(cond_params),
        sampler=BottomUpSampler(_VM),
        task_kwargs=dict(
            n_samples=N_SAMPLES,
            batch_size=1,
            max_interventions=2,
            intervention_domain=["prey_t", "predator_t"],
        ),
    )


if __name__ == "__main__":
    run_suite(
        title="Predator-Prey (LV vs ABM)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=10,
    )
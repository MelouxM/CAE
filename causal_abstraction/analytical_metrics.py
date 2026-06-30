"""
Metrics that analyse properties of the high-level model E and low-level
model M without comparing two CausalPaths. Unlike EvaluationMetric, these metrics
compute properties directly from the model components.

All subclasses implement:

    compute(high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None)
        -> float
"""

from __future__ import annotations

import itertools
import zlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

BOUNDED_METRICS = {'hard', 'jsd', 'mmd', 'hsic'}

def bound_and_normalize(result: float, inner_metric: str):
    """Clip bounded inner metrics to [0, 1]; squash others via ``tanh(|result|)``.

    Args:
        result: The raw metric value.
        inner_metric: The inner metric name; if it is in ``BOUNDED_METRICS`` the
            result is clipped, otherwise it is squashed.

    Returns:
        The value mapped into [0, 1] (NaN propagates).
    """
    if np.isnan(result):
        return float('nan')
    if inner_metric in BOUNDED_METRICS:
        return float(np.clip(result, 0.0, 1.0))
    return float(np.tanh(abs(result)))


class AnalyticalMetric(ABC):
    """
    Abstract base class for non-path-based analytical metrics.

    Subclasses implement ``_compute(...)`` which returns a raw float score. The
    public ``compute()`` method calls ``_compute()`` and, when normalize=True,
    passes the result through ``_normalize_result()`` which maps it to [0, 1]
    (0 = perfect, 1 = maximal error) via ``tanh(|score|)``.
    Subclasses whose score is already in [0, 1] should override
    ``_normalize_result()`` to return the result unchanged.
    """

    def __init__(self, normalize: bool = True):
        self._normalize = normalize

    # Public API
    def compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples: int, config=None) -> float:
        """
        Compute the metric, optionally normalizing the 'score' to [0, 1].

        Args:
            high_level_model: The high-level model E (CausalGraph).
            low_level_model: The low-level model M (LowLevelModel).
            value_map: The value map tau (ValueMap).
            cg_map: The coarse-graining map a (CoarseGrainingMap).
            sampler: The intervention sampler.
            n_samples: Number of samples for the internal experiment.
            config: Optional EvaluationConfig.

        Returns:
            A float (or another JSON-serializable scalar).
        """
        result = self._compute(high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config)
        if self._normalize:
            return self._normalize_result(result)
        return result

    # Subclass contract
    @abstractmethod
    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples: int, config=None) -> float:
        """Raw metric computation (no normalization)."""
        raise NotImplementedError

    def _normalize_result(self, result: float) -> float:
        """
        Map the score to [0, 1] via ``tanh(|score|)``.
        Overridden in subclasses whose score is already in [0, 1].
        """
        if isinstance(result, float):
            if np.isnan(result):
                pass  # propagate NaN
            elif np.isinf(result):
                return 1.0
            else:
                return float(np.tanh(abs(result)))
        return result


# Helper: get a fresh EvaluationConfig for inner runs

def _inner_config(config, metric='mse'):
    from .config import EvaluationConfig

    seed = config.seed if config is not None else None
    return EvaluationConfig(metric=metric, n_jobs=1, seed=seed)


# Structural deviation
class StructuralDeviationMetric(AnalyticalMetric):
    """
    Structural deviation from PDE analysis.

    Measures how sensitive the CAE_up is to an infinitesimal change in each high-level model parameter.
    For each parameter p_i with nominal value p0_i we compute:

        deviation_i = |CAE_up(p0 + δ·p0) − CAE_up(p0)| / (|CAE_up(p0)| + ε)

    Args:
        param_names: Names of the high-level model parameters to perturb.
        nominal_params: Nominal parameter values {name: float}.
        make_high_level_model: Factory rebuilding the high-level model from an
            updated parameter dict.
        delta: Relative perturbation size (default 0.01, i.e. 1%).
        inner_metric: Metric string used for the inner CAE_up experiment.
        normalize: If True, map the score to [0, 1].
    """

    def __init__(
        self,
        param_names: List[str],
        nominal_params: Dict[str, float],
        make_high_level_model: Callable,
        delta: float = 0.01,
        inner_metric: str = 'mse',
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.param_names = param_names
        self.nominal_params = nominal_params
        self.make_high_level_model = make_high_level_model
        self.delta = delta
        self.inner_metric = inner_metric

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .engine import EvaluationEngine

        cfg = _inner_config(config, self.inner_metric)

        base_engine = EvaluationEngine(high_level_model, low_level_model, value_map, cg_map, cfg)
        base_score = base_engine.run_experiment(sampler, n_samples=n_samples, batch_size=1).score

        deviations: List[float] = []
        for pname in self.param_names:
            perturbed = dict(self.nominal_params)
            perturbed[pname] = perturbed[pname] * (1.0 + self.delta)
            p_high_level_model = self.make_high_level_model(perturbed)

            p_engine = EvaluationEngine(p_high_level_model, low_level_model, value_map, cg_map, cfg)
            p_score = p_engine.run_experiment(sampler, n_samples=n_samples, batch_size=1).score

            deviations.append(abs(p_score - base_score) / (abs(base_score) + 1e-12))

        return float(np.mean(deviations)) if deviations else float('nan')


# Causal sensitivity index
class CausalSensitivityIndexMetric(AnalyticalMetric):
    """
    Causal Sensitivity Index from PDE analysis.

    Measures the contribution of each high-level model parameter by zeroing it out and measuring the resulting degradation in CAE_up:

        sensitivity_i = |CAE_up(p0) − CAE_up(p0 with p_i = 0)| / (|CAE_up(p0)| + ε)

    Args:
        param_names: Names of the high-level model parameters to zero out.
        nominal_params: Nominal parameter values {name: float}.
        make_high_level_model: Factory rebuilding the high-level model from an
            updated parameter dict.
        inner_metric: Metric string used for the inner CAE_up experiment.
        normalize: If True, map the score to [0, 1].
    """

    def __init__(
        self,
        param_names: List[str],
        nominal_params: Dict[str, float],
        make_high_level_model: Callable,
        inner_metric: str = 'mse',
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.param_names = param_names
        self.nominal_params = nominal_params
        self.make_high_level_model = make_high_level_model
        self.inner_metric = inner_metric

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .engine import EvaluationEngine

        cfg = _inner_config(config, self.inner_metric)

        base_engine = EvaluationEngine(high_level_model, low_level_model, value_map, cg_map, cfg)
        base_score = base_engine.run_experiment(sampler, n_samples=n_samples, batch_size=1).score

        indices: List[float] = []
        for pname in self.param_names:
            zeroed = dict(self.nominal_params)
            zeroed[pname] = 0.0
            z_high_level_model = self.make_high_level_model(zeroed)

            z_engine = EvaluationEngine(z_high_level_model, low_level_model, value_map, cg_map, cfg)
            z_score = z_engine.run_experiment(sampler, n_samples=n_samples, batch_size=1).score

            indices.append(abs(z_score - base_score) / (abs(base_score) + 1e-12))

        return float(np.mean(indices)) if indices else float('nan')


# Mallows' Cp
class MallowsCpMetric(AnalyticalMetric):
    """
    Mallows' Cp model-selection criterion.

    Collects (E prediction, M output) pairs, fits a linear proxy model
    (intercept + per-dimension slope), and computes:

        Cp = RSS / σ² − N + 2p

    where p is the number of fitted proxy parameters (slope + intercept per
    output dimension, i.e. p = 2 * number of scored output variables unless
    n_params is given explicitly), sigma^2 is the mean variance of M's outputs
    across output dimensions (a reference noise floor; or ``sigma2_ref`` if
    given), and N is the number of observations.

    Normalisation: tanh(max(0, Cp - p) / max(N, 1))
      - 0.0 when Cp <= p  (no underfitting penalty; unbiased or overfit)
      - -> 1.0 as Cp >> p (underfitting)
      - Special case RSS ≈ 0 (perfect fit) -> 0.0
    """

    def __init__(self, n_params: Optional[int] = None,
                 sigma2_ref: Optional[float] = None,
                 output_vars: Optional[List[str]] = None,
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.n_params = n_params
        self.sigma2_ref = sigma2_ref
        self.output_vars = output_vars
        self._p_est = 1
        self._n_est = 1

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .sampling import TopDownSampler

        cfg = _inner_config(config, 'mse')
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        td = TopDownSampler(value_map)
        rng = np.random.default_rng(cfg.seed)

        root_vars = [
            v for name, v in high_level_model.variables.items()
            if not high_level_model._nodes.get(name, {}).get('parents')
        ]

        path_high_level_model = builder.build_path_standard_high_level_model()
        path_low_level_model = builder.build_path_standard_low_level_model()

        # Collect per-variable lists independently to avoid shape mixing
        high_level_model_preds: Dict[str, List[np.ndarray]] = {}
        low_level_model_outs: Dict[str, List[np.ndarray]] = {}

        for _ in range(n_samples):
            spec = td.sample_intervention(root_vars, batch_size=1, force_all=True, rng=rng)
            res_a = path_high_level_model.execute(spec, rng=rng)
            res_b = path_low_level_model.execute(spec, rng=rng)

            common = set(res_a.keys()) & set(res_b.keys())
            if self.output_vars is not None:
                common = common & set(self.output_vars)
            for vname in common:
                try:
                    a = np.asarray(res_a[vname], dtype=float).ravel()
                    b = np.asarray(res_b[vname], dtype=float).ravel()
                    if a.shape != b.shape or np.any(np.isnan(a)) or np.any(np.isnan(b)):
                        continue
                    high_level_model_preds.setdefault(vname, []).append(a)
                    low_level_model_outs.setdefault(vname, []).append(b)
                except Exception:
                    pass

        if not high_level_model_preds:
            return float('nan')

        if self.n_params is not None:
            p = self.n_params
        else:
            if self.output_vars is not None:
                n_scored = len(self.output_vars)
            else:
                root_names = {
                    name for name in high_level_model.variables
                    if not high_level_model._nodes.get(name, {}).get('parents')
                }
                n_scored = sum(1 for name in high_level_model.variables if name not in root_names)
            p = max(1, n_scored) * 2
        self._p_est = p

        cp_values: List[float] = []

        for vname in high_level_model_preds:
            X_list = high_level_model_preds[vname]
            y_list = low_level_model_outs[vname]
            if len(X_list) < max(10 * p, 20):
                continue

            X_mat = np.stack(X_list)  # (N, d_var) - all same shape, safe to stack
            y_mat = np.stack(y_list)  # (N, d_var)
            N = len(X_mat)
            self._n_est = N

            X_flat = X_mat.reshape(N, -1)
            y_flat = y_mat.reshape(N, -1)

            if self.sigma2_ref is not None:
                sigma2 = self.sigma2_ref
            else:
                sigma2 = float(np.mean(np.var(y_flat, axis=0)))
                if sigma2 < 1e-12:
                    cp_values.append(0.0)
                    continue

            Xmat = np.column_stack([np.ones(N), X_flat])
            try:
                beta, _, _, _ = np.linalg.lstsq(Xmat, y_flat, rcond=None)
                y_hat = Xmat @ beta
                rss = float(np.sum((y_flat - y_hat) ** 2))
            except Exception:
                continue

            if np.isnan(rss):
                continue
            if rss < 1e-12:
                cp_values.append(0.0)
                continue

            cp_values.append(float(rss / sigma2 - N + 2 * p))

        if not cp_values:
            return float('nan')

        return float(np.mean(cp_values))

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        p = float(self._p_est)
        N = float(self._n_est)
        return float(np.tanh(max(0.0, result - p) / max(N, 1.0)))


# Lagrangian
class IBLagrangianMetric(AnalyticalMetric):
    """
    Regular Information Bottleneck Lagrangian.

        L_IB^β[q_{T|X}] := I(X; T) − β · I(T; Y)

    where X = micro-state, T = abstract label, Y = low-level model output (observed
    passively, no interventions), and β ≥ 0.

    * I(X; T)  estimated via binned mutual information between grounded
               micro-values and their abstract labels.
    * I(T; Y)  estimated via binned mutual information between the per-sample
               abstract label and the per-sample abstracted low-level output
               (the two are aligned 1:1, one entry per sample), under the
               observational (no-intervention) distribution.

    Contrast with CIBLagrangianMetric which uses I_c(Y | do(T)), an
    interventional quantity approximated by the CAE_up score. The regular IB
    uses passive co-occurrence statistics, so a spurious correlate of Y that
    is not causally downstream of T can inflate I(T; Y) here but not I_c.

    Args:
        beta: Trade-off weight β ≥ 0 (default 1.0).
        n_bins: Histogram bins for MI estimation (default 20).
        inner_metric: Metric string used for the inner experiment (default 'mse').
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, beta: float = 1.0, n_bins: int = 20, inner_metric: str = 'mse',
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.beta = beta
        self.n_bins = n_bins
        self.inner_metric = inner_metric

    @staticmethod
    def _histogram_entropy(x: np.ndarray, n_bins: int) -> float:
        """Estimate H(X) in nats via histogram."""
        counts, _ = np.histogram(x, bins=n_bins)
        p = counts.astype(float) + 1e-10
        p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _binned_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        """Estimate I(X; Y) via joint histogram."""
        x = x.ravel()
        y = y.ravel()
        n = min(len(x), len(y))
        H_xy, _, _ = np.histogram2d(x[:n], y[:n], bins=n_bins)
        H_xy = H_xy + 1e-10
        p_xy = H_xy / H_xy.sum()
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            mi = float(np.nansum(p_xy * np.log(p_xy / (p_x * p_y + 1e-10) + 1e-10)))
        return max(0.0, mi)

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .sampling import TopDownSampler
        from .primitives import UNMAPPED

        cfg = _inner_config(config, self.inner_metric)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())
        td = TopDownSampler(value_map)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)

        micro_vals: List[float] = []
        abstract_vals: List[float] = []        # per (sample, variable), paired with micro_vals for I(X; T)
        output_vals: List[float] = []          # per-sample abstracted low-level outputs (observational)
        abstract_y_vals: List[float] = []      # per-sample abstract label, aligned 1:1 with output_vals for I(T; Y)

        for _ in range(n_samples):
            spec = td.sample_intervention(all_vars, batch_size=1, rng=rng)

            # Collect (X, T) pairs - same as CIB
            sample_abstract: List[float] = []
            for vname, entry in spec.items():
                labels = entry.get('labels')
                if labels is None:
                    continue
                lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
                try:
                    mv = value_map.ground(vname, lbl, rng=rng)
                    a_val = (
                        float(lbl) if np.isscalar(lbl)
                        else float(np.mean(np.asarray(lbl, dtype=float)))
                    )
                    abstract_vals.append(a_val)
                    micro_vals.append(float(np.mean(np.asarray(mv, dtype=float))))
                    sample_abstract.append(a_val)
                except Exception:
                    pass

            # Collect Y: run low-level model passively and abstract output
            try:
                low_level_model_ints = builder._step_ground_or_passthrough(spec, rng=rng)
                low_level_model_state = builder._step_low_level_model_execute(low_level_model_ints, rng=rng)
                low_level_model_abs = builder._step_abstract(low_level_model_state, rng=rng)

                # Summarize all output variable values into one scalar per sample
                out_scalars = []
                for vname, val in low_level_model_abs.items():
                    if val is UNMAPPED or val is None:
                        continue
                    if isinstance(val, list):
                        val = val[0] if len(val) == 1 else val
                    try:
                        out_scalars.append(
                            float(val) if np.isscalar(val)
                            else float(np.mean(np.asarray(val, dtype=float)))
                        )
                    except Exception:
                        pass
                # Append Y and the per-sample abstract label T together so the two
                # lists stay index-aligned (one entry per sample) for I(T; Y).
                if out_scalars and sample_abstract:
                    output_vals.append(float(np.mean(out_scalars)))
                    abstract_y_vals.append(float(np.mean(sample_abstract)))
            except Exception:
                pass

        # I(X; T)
        i_xt = 0.0
        if len(micro_vals) >= 4:
            i_xt = self._binned_mi(
                np.array(micro_vals), np.array(abstract_vals), self.n_bins
            )

        # I(T; Y) - observational, using the per-sample abstract label and the
        # per-sample abstracted low-level output (aligned 1:1 by construction).
        i_ty = 0.0
        if len(abstract_y_vals) >= 4:
            i_ty = self._binned_mi(
                np.array(abstract_y_vals),
                np.array(output_vals),
                self.n_bins,
            )

        self._h_x_est = self._histogram_entropy(
            np.array(micro_vals), self.n_bins
        ) if len(micro_vals) >= 4 else 1.0

        self._h_y_est = self._histogram_entropy(
            np.array(output_vals), self.n_bins
        ) if len(output_vals) >= 4 else 1.0

        return float(i_xt - self.beta * i_ty)

    def _normalize_result(self, result: float) -> float:
        if isinstance(result, float) and not np.isnan(result):
            h_x = getattr(self, '_h_x_est', 1.0)
            h_y = getattr(self, '_h_y_est', 1.0)
            denom = h_x + self.beta * h_y
            if denom < 1e-12:
                return 0.0
            result = float(np.clip(
                (result + self.beta * h_y) / denom,
                0.0, 1.0
            ))
        return result


# Causal information bottleneck (CIB) Lagrangian
class CIBLagrangianMetric(IBLagrangianMetric):
    """
    Causal Information Bottleneck Lagrangian.

        L_CIB^β[q_{T|X}] := I(X; T) − β · I_c(Y | do(T))

    where X = micro-state, T = abstract label, Y = low-level model output, and β ≥ 0.

    * I(X; T)          estimated via binned mutual information.
    * I_c(Y | do(T))   the interventional relevance of T for Y, estimated
                       directly from histograms as max(0, H(Y) − H_c(Y | do(T))).
                       H(Y) is the histogram entropy of the abstracted low-level
                       outputs sampled under direct (top-down) interventions on T;
                       H_c(Y | do(T)) is the sample-weighted mean of the per-bin
                       output entropies obtained by binning the abstract labels
                       into n_bins cells. No inner CAE_up experiment is run.

    The optimal abstraction satisfies L_CIB^β = 0 at the Pareto frontier.
    A lower L_CIB (more negative) indicates the abstraction is both
    minimal (small I(X;T)) and interventionally sufficient (large I_c).

    Args:
        beta: Trade-off weight β ≥ 0 (default 1.0).
        n_bins: Histogram bins for MI/entropy estimation and for binning the
            abstract labels into discrete T cells (default 20).
        inner_metric: Metric string for the inner EvaluationConfig that seeds the
            RNG and configures the DiagramBuilder grounding/abstraction passes
            (default 'hard'); it does NOT select or run a CAE_up experiment.
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, beta: float = 1.0, n_bins: int = 20, inner_metric: str = 'hard',
                 normalize: bool = True):
        super().__init__(beta, n_bins, inner_metric, normalize)

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .sampling import TopDownSampler
        from .paths import DiagramBuilder
        from .primitives import UNMAPPED

        cfg = _inner_config(config, self.inner_metric)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())
        td = TopDownSampler(value_map)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)

        # Collect (x_micro, t_abstract, y_abstract) triples
        micro_vals = []  # scalar summary of micro input x
        abstract_vals = []  # abstract label t = τ(x)
        y_do_vals = []  # abstract low-level model output y under do(X=x)

        for _ in range(n_samples):
            spec = td.sample_intervention(all_vars, batch_size=1, force_all=True, rng=rng)

            # Collect (X, T) pairs
            sample_micro = []
            sample_abstract = []
            for vname, entry in spec.items():
                labels = entry.get('labels')
                if labels is None:
                    continue
                lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
                try:
                    mv = value_map.ground(vname, lbl, rng=rng)
                    sample_abstract.append(
                        float(lbl) if np.isscalar(lbl)
                        else float(np.mean(np.asarray(lbl, dtype=float)))
                    )
                    sample_micro.append(float(np.mean(np.asarray(mv, dtype=float))))
                except Exception:
                    pass

            if not sample_micro:
                continue

            # Run low-level model under do(X=x)
            try:
                low_level_model_ints = builder._step_ground_or_passthrough(spec, rng=rng)
                low_level_model_state = builder._step_low_level_model_execute(low_level_model_ints, rng=rng)
                low_level_model_abs = builder._step_abstract(low_level_model_state, rng=rng)
            except Exception:
                continue

            # Summarize y into a scalar
            out_scalars = []
            for vname, val in low_level_model_abs.items():
                if val is UNMAPPED or val is None:
                    continue
                if isinstance(val, list):
                    val = val[0] if len(val) == 1 else val
                try:
                    out_scalars.append(
                        float(val) if np.isscalar(val)
                        else float(np.mean(np.asarray(val, dtype=float)))
                    )
                except Exception:
                    pass

            if not out_scalars:
                continue

            micro_vals.append(float(np.mean(sample_micro)))
            abstract_vals.append(float(np.mean(sample_abstract)))
            y_do_vals.append(float(np.mean(out_scalars)))

        if len(micro_vals) < 4:
            return float('nan')

        micro_arr = np.array(micro_vals)
        abstract_arr = np.array(abstract_vals)
        y_arr = np.array(y_do_vals)

        # I(X; T) via binned MI (same as IBLagrangianMetric)
        i_xt = self._binned_mi(micro_arr, abstract_arr, self.n_bins)

        # I_c(Y | do(T)) = H(Y) - H_c(Y | do(T))
        # H(Y): marginal entropy of Y under the intervention distribution
        h_y = self._histogram_entropy(y_arr, self.n_bins)

        # H_c(Y | do(T)): average entropy of p(Y | do(T=t))
        # Bin abstract labels to define discrete "T values"
        t_bins = np.digitize(
            abstract_arr,
            np.linspace(abstract_arr.min() - 1e-9, abstract_arr.max() + 1e-9, self.n_bins + 1)
        )

        h_c = 0.0
        total_weight = 0.0
        for b in np.unique(t_bins):
            mask = (t_bins == b)
            n_t = mask.sum()
            if n_t < 2:
                continue
            # p(Y | do(T=t)) is the empirical distribution of y for samples where τ(x) falls in bin b
            # (uniform prior p*(x) means p*(x|t) ∝ q(t|x), this is just the empirical conditional (τ deterministic))
            h_y_given_t = self._histogram_entropy(y_arr[mask], min(self.n_bins, n_t))
            weight = float(n_t)
            h_c += weight * h_y_given_t
            total_weight += weight

        if total_weight > 0:
            h_c /= total_weight

        i_c = max(0.0, h_y - h_c)

        # Cache for normalization
        self._h_x_est = self._histogram_entropy(micro_arr, self.n_bins)
        self._i_c_est = i_c

        return float(i_xt - self.beta * i_c)

    def _normalize_result(self, result: float) -> float:
        if isinstance(result, float) and not np.isnan(result):
            h_x = getattr(self, '_h_x_est', 1.0)
            denom = h_x + self.beta
            if denom < 1e-12:
                return 0.0
            result = float(np.clip((result + self.beta) / denom, 0.0, 1.0))
        return result


# Macroscopic invariance
class MacroscopicInvarianceMetric(AnalyticalMetric):
    """
    Macroscopic Invariance from renormalization-group analysis.

    Tests whether the low-level model output distribution is invariant to which
    micro-state within an abstract cell is used.

    For each abstract variable v and each label l, we sample ``n_pairs``
    pairs of distinct micro-states (x1, x2) that both map to label l under
    the current ValueMap.  We run the low-level model on each and measure the divergence
    of the abstracted outputs.

    High invariance (low divergence) = the output depends only on the
    abstract label, not on the specific micro-realisation -> macroscopic
    invariance holds.

    Args:
        n_pairs: Number of within-cell pairs to test per label (default 10).
        inner_metric: Metric used to compare the two outputs (default 'mse').
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, n_pairs: int = 10, inner_metric: str = 'mse',
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.n_pairs = n_pairs
        self.inner_metric = inner_metric

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .metrics import get_metric

        cfg = _inner_config(config, self.inner_metric)
        metric = get_metric(cfg)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)

        results: List[Any] = []

        for vname, label_to_subspace in value_map.specs.items():
            var_scores = []
            selectors = cg_map.get_micro_vars(vname)
            if not selectors:
                continue

            for label, subspace in label_to_subspace.items():
                pair_scores = []
                for _ in range(self.n_pairs):
                    try:
                        # Sample two micro-states from the same abstract cell
                        mv1 = subspace.sample((1, subspace.dim), rng=rng)
                        mv2 = subspace.sample((1, subspace.dim), rng=rng)

                        # Build intervention specs (single-variable intervention)
                        spec1 = {vname: {'labels': [label], 'micro_values': mv1}}
                        spec2 = {vname: {'labels': [label], 'micro_values': mv2}}

                        # Run low-level model on both and abstract
                        low_level_model_int1 = builder._step_ground_or_passthrough(spec1, rng=rng)
                        low_level_model_int2 = builder._step_ground_or_passthrough(spec2, rng=rng)
                        state1 = builder._step_low_level_model_execute(low_level_model_int1, rng=rng)
                        state2 = builder._step_low_level_model_execute(low_level_model_int2, rng=rng)
                        abs1 = builder._step_abstract(state1, rng=rng)
                        abs2 = builder._step_abstract(state2, rng=rng)

                        # Score the output agreement for each high-level model output variable
                        for out_var in high_level_model.variables:
                            if out_var in abs1 and out_var in abs2:
                                v1 = np.asarray(abs1[out_var], dtype=float)
                                v2 = np.asarray(abs2[out_var], dtype=float)
                                pair_scores.append(metric.measure(v1, v2))
                    except Exception:
                        pass

                if pair_scores:
                    var_scores.extend(pair_scores)

            if var_scores:
                results.append(float(np.mean(var_scores)))

        return float(np.mean(results)) if results else float('nan')

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float(np.clip(result, 0.0, 1.0))


# Complexity shift
class ComplexityShiftMetric(AnalyticalMetric):
    """
    Algorithmic Complexity Shift from algorithmic information dynamics.

    Uses zlib compression as a proxy for Kolmogorov complexity.
    For each sampled input x and a Gaussian-perturbed version x+eps,
    computes the change in compressed output size induced by the
    perturbation, independently for the high-level model E and low-level model M:

        shift_E = (K(E(tau(x+eps))) - K(E(tau(x)))) / K(input)
        shift_M = (K(tau(M(x+eps))) - K(tau(M(x)))) / K(input)

    The metric returns |mean(shift_E) - mean(shift_M)|: how much the two models
    disagree on the complexity change induced by a small perturbation.

    Args:
        output_vars: Output variables to score (None = all non-root variables).
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, output_vars: Optional[List[str]] = None, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.output_vars = output_vars

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .sampling import TopDownSampler
        from .primitives import UNMAPPED

        cfg = _inner_config(config, 'mse')
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        td = TopDownSampler(value_map)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())

        # Perturbation std: small relative to typical micro-value scale
        epsilon = 0.01

        high_level_model_shifts, low_level_model_shifts = [], []

        for _ in range(n_samples):
            spec = td.sample_intervention(all_vars, batch_size=1, rng=rng)

            # Ground spec to micro-values (original input x)
            try:
                x = builder._step_ground_or_passthrough(spec, rng=rng)
            except Exception:
                continue

            # Perturbed input x + eps
            x_eps = {}
            for k, v in x.items():
                if isinstance(v, np.ndarray):
                    x_eps[k] = v + rng.normal(0, epsilon, size=v.shape)
                else:
                    x_eps[k] = v

            input_bytes = repr([(k, v.tolist() if isinstance(v, np.ndarray)
            else v) for k, v in sorted(x.items())]).encode()
            input_size = len(zlib.compress(input_bytes, level=9))
            if input_size == 0:
                continue

            # high-level model complexity shift: K(output_perturbed) - K(output_original)
            try:
                high_level_model_out_orig = builder._step_high_level_model_predict(spec, rng=rng)
                # Build perturbed spec by re-abstracting x_eps
                from .primitives import SystemState as _SS
                abs_xeps = builder._step_abstract(_SS(values=x_eps), rng=rng)
                perturbed_spec = {}
                for k in spec:
                    if k in abs_xeps and abs_xeps[k] is not UNMAPPED:
                        perturbed_spec[k] = {'labels': abs_xeps[k]
                        if isinstance(abs_xeps[k], list)
                        else [abs_xeps[k]],
                                             'micro_values': None}
                    else:
                        perturbed_spec[k] = spec[k]
                high_level_model_out_eps = builder._step_high_level_model_predict(perturbed_spec, rng=rng)

                if self.output_vars is not None:
                    high_level_model_out_orig = {k: v for k, v in high_level_model_out_orig.items()
                                    if k in self.output_vars}
                    high_level_model_out_eps = {k: v for k, v in high_level_model_out_eps.items()
                                   if k in self.output_vars}

                k_orig = len(zlib.compress(
                    repr(list(high_level_model_out_orig.values())).encode(), level=9))
                k_eps = len(zlib.compress(
                    repr(list(high_level_model_out_eps.values())).encode(), level=9))
                high_level_model_shifts.append((k_eps - k_orig) / (input_size + 1e-9))
            except Exception:
                pass

            # low-level model complexity shift: K(abstract_output_perturbed) - K(abstract_output_original)
            try:
                low_level_model_ints_orig = x
                low_level_model_state_orig = builder._step_low_level_model_execute(low_level_model_ints_orig, rng=rng)
                abstract_orig = builder._step_abstract(low_level_model_state_orig, rng=rng)

                low_level_model_state_eps = builder._step_low_level_model_execute(x_eps, rng=rng)
                abstract_eps = builder._step_abstract(low_level_model_state_eps, rng=rng)

                if self.output_vars is not None:
                    abstract_orig = {k: v for k, v in abstract_orig.items()
                                     if k in self.output_vars}
                    abstract_eps = {k: v for k, v in abstract_eps.items()
                                    if k in self.output_vars}

                k_orig = len(zlib.compress(
                    repr(list(abstract_orig.values())).encode(), level=9))
                k_eps = len(zlib.compress(
                    repr(list(abstract_eps.values())).encode(), level=9))
                low_level_model_shifts.append((k_eps - k_orig) / (input_size + 1e-9))
            except Exception:
                pass

        if not high_level_model_shifts or not low_level_model_shifts:
            return float('nan')

        return float(abs(np.mean(high_level_model_shifts) - np.mean(low_level_model_shifts)))

# Sobol sensitivity indices
class SobolSensitivityMetric(AnalyticalMetric):
    """
    Sobol First-Order Variance-Based Sensitivity Indices for the high-level model.

    Estimates S_i = Var_{X_i}[E_{X_{~i}}(Y | X_i)] / Var(Y) using the
    Saltelli estimator:

        S_i ≈ (1/N) Σ_j f(A)_j · (f(AB_i)_j − f(B)_j) / Var(Y)

    where A and B are independent (N × k) sample matrices, and AB_i equals
    B with the i-th column replaced by A[:,i].

    The output variable Y is taken as the mean over all high-level model output values
    (a scalar summary).

    Args:
        n_samples: If not None, overrides the n_samples argument to compute().
        output_vars: Output variables to score (None = all).
        normalize: If True, map the score to [0, 1].
    """
    _MIN_SAMPLES_PER_INPUT = 100

    def __init__(self, n_samples: Optional[int] = None, output_vars: Optional[List[str]] = None,
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self._n_samples = n_samples
        self.output_vars = output_vars

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .sampling import TopDownSampler
        from .paths import DiagramBuilder
        from .primitives import UNMAPPED

        N = self._n_samples or n_samples
        cfg = _inner_config(config, 'mse')
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())
        var_names = [v.name for v in all_vars]
        td = TopDownSampler(value_map)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)

        k = len(list(high_level_model.variables))  # number of input dimensions
        min_required = self._MIN_SAMPLES_PER_INPUT * k
        if N < min_required:
            import warnings
            warnings.warn(
                f"SobolSensitivityMetric: n_samples={N} is below the recommended "
                f"minimum of {min_required} ({self._MIN_SAMPLES_PER_INPUT}×k={k}). ",
                UserWarning,
            )

        def _sample_row() -> List[float]:
            row = []
            for v in all_vars:
                spec = td.sample_intervention([v], batch_size=1, rng=rng)
                entry = spec.get(v.name, {})
                labels = entry.get('labels', [0])
                lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
                try:
                    row.append(float(lbl) if np.isscalar(lbl)
                               else float(np.mean(np.asarray(lbl, dtype=float))))
                except Exception:
                    row.append(0.0)
            return row

        _score_vars = self.output_vars

        def _eval_high_level_model(label_row: List[float]) -> float:
            inputs = {vname: lbl for vname, lbl in zip(var_names, label_row)}
            pred = high_level_model.predict(inputs)
            vals = []
            for vn, v in pred.items():
                if _score_vars is not None and vn not in _score_vars:
                    continue
                if v is not UNMAPPED and v is not None:
                    try:
                        vals.append(float(np.mean(np.asarray(v, dtype=float))))
                    except Exception:
                        pass
            return float(np.mean(vals)) if vals else 0.0

        def _eval_low_level_model(label_row: List[float]) -> float:
            # Build a spec from the abstract label row, ground it, run low-level model, abstract
            spec = {
                vname: {'labels': [lbl], 'micro_values': None}
                for vname, lbl in zip(var_names, label_row)
            }
            try:
                low_level_model_ints = builder._step_ground_or_passthrough(spec, rng=rng)
                low_level_model_state = builder._step_low_level_model_execute(low_level_model_ints, rng=rng)
                low_level_model_abs = builder._step_abstract(low_level_model_state, rng=rng)
            except Exception:
                return 0.0

            vals = []
            for vn, val in low_level_model_abs.items():
                if _score_vars is not None and vn not in _score_vars:
                    continue
                if val is UNMAPPED or val is None:
                    continue
                if isinstance(val, list):
                    val = val[0] if len(val) == 1 else val
                try:
                    vals.append(float(np.mean(np.asarray(val, dtype=float))))
                except Exception:
                    pass
            return float(np.mean(vals)) if vals else 0.0

        # Shared Saltelli matrices - same rows for both models so indices are comparable
        A = [_sample_row() for _ in range(N)]
        B = [_sample_row() for _ in range(N)]

        fA_high_level_model = np.array([_eval_high_level_model(r) for r in A])
        fB_high_level_model = np.array([_eval_high_level_model(r) for r in B])
        fA_low_level_model = np.array([_eval_low_level_model(r) for r in A])
        fB_low_level_model = np.array([_eval_low_level_model(r) for r in B])

        var_high_level_model = float(np.var(np.concatenate([fA_high_level_model, fB_high_level_model])))
        var_low_level_model = float(np.var(np.concatenate([fA_low_level_model, fB_low_level_model])))

        disagreements: List[float] = []
        self._sobol_indices: Dict[str, Dict[str, float]] = {}

        for i, vname in enumerate(var_names):
            # Build AB_i: B with column i replaced by A's column i
            AB_i = [list(b) for b in B]
            for j in range(N):
                AB_i[j][i] = A[j][i]

            fABi_high_level_model = np.array([_eval_high_level_model(r) for r in AB_i])
            fABi_low_level_model = np.array([_eval_low_level_model(r) for r in AB_i])

            S_high_level_model = (float(np.mean(fA_high_level_model * (fABi_high_level_model - fB_high_level_model))) / (var_high_level_model + 1e-12))
            S_low_level_model = (float(np.mean(fA_low_level_model * (fABi_low_level_model - fB_low_level_model))) / (var_low_level_model + 1e-12))

            S_high_level_model = float(np.clip(S_high_level_model, 0.0, 1.0))
            S_low_level_model = float(np.clip(S_low_level_model, 0.0, 1.0))

            self._sobol_indices[vname] = {'high_level_model': S_high_level_model, 'low_level_model': S_low_level_model}
            disagreements.append(abs(S_high_level_model - S_low_level_model))

        return float(np.mean(disagreements)) if disagreements else float('nan')

    def _normalize_result(self, result: float) -> float:
        return result


# Interchange intervention accuracy (IIA)
class IIAMetric(AnalyticalMetric):
    """
    Interchange intervention accuracy (Geiger et al., 2022).

    For each variable v and each pair of states (x, x'):
      - x_interchange = x but with the micro-variables of v replaced by those from x'
      - Path A: high-level model prediction with abstract label of v set to abstract(x')[v],
                all other inputs from x
      - Path B: low-level model run on x_interchange, then abstracted

    IIA score = average disagreement (abstraction error) between Path A and Path B
    outputs: 0 = perfect agreement, higher = worse. The value is inverted relative
    to the literature, which reports IIA as an accuracy, to match this library's
    error orientation (lower-is-better).

    For discrete ValueMaps: x and x' are sampled via BottomUpSampler, which
    produces concrete micro-values with proper abstract labels.

    For ContinuousValueMaps: uses the sampler to get concrete label values
    for both x and x'. The interchange replaces v's micro-value in x with
    v's micro-value from x'. Since ground() is identity, the micro-value IS
    the label, so the interchange is exactly replacing the population value.

    Args:
        inner_metric: Metric used to compare Path A and Path B outputs.
        output_vars: High-level model output variables to score (None = the
            output/sink variables, i.e. nodes with no children).
        n_pairs: Number of (x, x') pairs per variable (None = use n_samples from
            compute()).
        normalize: If True, map the score to [0, 1].
    """

    def __init__(
        self,
        inner_metric: str = 'hard',
        output_vars: Optional[List[str]] = None,
        n_pairs: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.inner_metric = inner_metric
        self.output_vars  = output_vars
        self.n_pairs      = n_pairs

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .metrics import get_metric
        from .sampling import BottomUpSampler
        from .valuemap import ContinuousValueMap
        from .primitives import UNMAPPED

        cfg = _inner_config(config, self.inner_metric)
        metric = get_metric(cfg)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)

        n_pairs = self.n_pairs if self.n_pairs is not None else n_samples
        is_continuous = isinstance(value_map, ContinuousValueMap)
        all_vars = list(high_level_model.variables.values())
        var_names = [v.name for v in all_vars]

        root_names = {
            name for name in high_level_model.variables
            if not high_level_model._nodes.get(name, {}).get('parents')
        }
        # Default to output (sink) variables only (nodes with no children),
        # matching the paper's "IIA checks consistency only at output variables".
        leaf_names = {
            name for name in var_names
            if not high_level_model._adjacency.get(name)
        }
        score_vars = self.output_vars or [
            n for n in var_names if n in leaf_names and n not in root_names
        ]

        if is_continuous:
            pair_sampler = sampler
        else:
            pair_sampler = BottomUpSampler(value_map)

        var_scores: Dict[str, List[float]] = {name: [] for name in var_names}

        for _ in range(n_pairs):
            root_vars = [v for v in all_vars if v.name in root_names]

            spec_b = pair_sampler.sample_intervention(root_vars, batch_size=1, force_all=True, rng=rng)
            spec_s = pair_sampler.sample_intervention(root_vars, batch_size=1, force_all=True, rng=rng)

            # Run high-level model on source
            try:
                high_level_model_pred_s = builder._step_high_level_model_predict(spec_s, rng=rng)
            except Exception:
                continue

            # Run low-level model on source -> full micro-state
            try:
                low_level_model_ints_s = builder._step_ground_or_passthrough(spec_s, rng=rng)
                low_level_model_state_s = builder._step_low_level_model_execute(low_level_model_ints_s, rng=rng)
            except Exception:
                continue

            # Ground base inputs for low-level model counterfactual
            try:
                low_level_model_ints_b = builder._step_ground_or_passthrough(spec_b, rng=rng)
            except Exception:
                continue

            # Interchange
            for v in all_vars:
                vname = v.name

                # Skip terminal outputs
                has_children = bool(high_level_model._adjacency.get(vname, []))
                if vname not in root_names and not has_children:
                    continue

                # high-level model: Force v = high_level_model_source[v], everything else from base
                high_level_model_v_s = high_level_model_pred_s.get(vname)
                if high_level_model_v_s is None:
                    continue
                if isinstance(high_level_model_v_s, list) and any(x is UNMAPPED for x in high_level_model_v_s):
                    continue
                if high_level_model_v_s is UNMAPPED:
                    continue

                high_level_model_input_cf = {}
                for n, entry in spec_b.items():
                    if n not in high_level_model.variables:
                        continue
                    lbl_list = entry.get('labels', [])
                    high_level_model_input_cf[n] = (
                        lbl_list[0]
                        if isinstance(lbl_list, (list, np.ndarray)) and len(lbl_list) > 0
                        else lbl_list
                    )

                high_level_model_v_val = high_level_model_v_s[0] if isinstance(high_level_model_v_s, list) else high_level_model_v_s
                high_level_model_input_cf[vname] = builder._coerce_label(high_level_model_v_val)

                try:
                    high_level_model_cf = high_level_model.predict(high_level_model_input_cf)
                except Exception:
                    continue

                # low-level model: Force v's micro-variables = raw values from source low-level model run
                selectors = cg_map.get_micro_vars(vname)
                if not selectors:
                    continue

                micro_patch = {}
                all_found = True
                for sel in selectors:
                    if sel.variable not in low_level_model_state_s.values:
                        all_found = False
                        break
                    full_val = low_level_model_state_s.values[sel.variable]
                    if sel.index is not None:
                        base = low_level_model_ints_b.get(sel.variable)
                        # Reconstruct full array when grounding produced (index, value) tuples
                        if isinstance(base, list) and base and isinstance(base[0], tuple):
                            shape = cg_map.schema.get_shape(sel.variable)
                            batch_size_b = next(
                                (v.shape[0] for _, v in base if isinstance(v, np.ndarray)), 1
                            )
                            canvas = np.zeros((batch_size_b, *shape), dtype=float)
                            for idx, val in base:
                                if idx is None:
                                    canvas[:] = val.reshape(canvas.shape)
                                elif isinstance(idx, int):
                                    canvas[:, idx:idx + 1] = val
                                else:
                                    canvas[:, idx] = val
                            base = canvas
                        if not isinstance(base, np.ndarray):
                            all_found = False
                            break
                        patched = base.copy()
                        if isinstance(sel.index, int):
                            patched[:, sel.index:sel.index + 1] = full_val[:, sel.index:sel.index + 1]
                        else:
                            patched[:, sel.index] = full_val[:, sel.index]
                        micro_patch[sel.variable] = patched
                    else:
                        micro_patch[sel.variable] = full_val

                if not all_found:
                    continue

                # Patch base interventions with source micro-values for v
                low_level_model_cf_ints = dict(low_level_model_ints_b)
                low_level_model_cf_ints.update(micro_patch)

                try:
                    low_level_model_state_cf = builder._step_low_level_model_execute(low_level_model_cf_ints, rng=rng)
                    low_level_model_abs_cf = builder._step_abstract(low_level_model_state_cf, rng=rng)
                except Exception:
                    continue

                # Score
                for sv in score_vars:
                    if sv not in high_level_model_cf or sv not in low_level_model_abs_cf:
                        continue
                    a = high_level_model_cf[sv]
                    b_val = low_level_model_abs_cf[sv]
                    if a is UNMAPPED or b_val is UNMAPPED:
                        continue
                    a_arr = _to_float_array(a)
                    b_arr = _to_float_array(b_val)
                    if a_arr is None or b_arr is None or a_arr.shape != b_arr.shape:
                        continue
                    try:
                        var_scores[vname].append(float(metric.measure(a_arr, b_arr)))
                    except Exception:
                        pass

        scored = [float(np.mean(s)) for s in var_scores.values() if s]
        return float(np.mean(scored)) if scored else float('nan')

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float(np.clip(result, 0.0, 1.0))


# Behavioral/dynamic causal consistency (BCC/DCC)
class BCCMetric(AnalyticalMetric):
    """
    Behavioral Causal Consistency (BCC) from NC-MCM.

    For every pair of micro-states (x1, x2) that share the same abstract label,
    the low-level model outputs when run from x1 and x2 must agree after abstraction.

    Unlike macroscopic invariance, which perturbs a single variable in isolation
    while leaving the rest of the state unspecified, BCC tests each variable
    within a complete system state. Each iteration draws a full joint assignment
    of labels to all mapped input (root) variables of E, grounds the whole state to
    a micro-realization, then re-grounds it to a second micro-realization of the
    same complete label vector; the abstracted low-level outputs (read from the
    non-root behavioral variables) of the two runs must agree. Pairs whose
    grounding is the identity (ideal ABM, where re-grounding reproduces the same
    micro-state) are trivially consistent and skipped so they don't inflate the
    mean with uninformative zeros. BCC scores > 0 only when grounding genuinely
    produces distinct micro-states for the same complete label vector (e.g. a
    spatial ABM: same population counts, different spatial layout).

    For discrete value maps the joint label assignment is drawn directly from each
    variable's label set; for continuous value maps it is drawn via the sampler.

    Args:
        n_pairs: Number of complete-system-state pairs to test.
        inner_metric: Metric used to compare the two abstracted outputs.
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, n_pairs: int = 10, inner_metric: str = 'mse',
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.n_pairs = n_pairs
        self.inner_metric = inner_metric

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .metrics import get_metric
        from .valuemap import ContinuousValueMap

        cfg = _inner_config(config, self.inner_metric)
        metric = get_metric(cfg)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)

        is_continuous = isinstance(value_map, ContinuousValueMap)
        all_high_level_model_vars = list(high_level_model.variables.values())

        # The complete system state is the joint assignment over the mapped input
        # (root) variables of E; behavior is read from the remaining (non-root)
        # output variables. Intervening on the whole input state (rather than a
        # single variable) is what distinguishes BCC from macroscopic invariance.
        root_names = [v.name for v in high_level_model.get_roots() if v.name in value_map.specs]
        if not root_names:
            # No mapped root variables: fall back to all mapped variables so the
            # complete-state pair remains well-defined.
            root_names = list(value_map.specs.keys())
        root_set = set(root_names)

        var_scores: List[float] = []

        for _ in range(self.n_pairs):
            try:
                # Draw a complete macro-state: one label per mapped root variable.
                labels: Dict[str, Any] = {}
                if is_continuous:
                    spec = sampler.sample_intervention(
                        all_high_level_model_vars, batch_size=1, force_all=True, rng=rng
                    )
                    for vn in root_names:
                        lbls = spec.get(vn, {}).get('labels', [])
                        if not len(lbls):
                            raise ValueError(f"sampler returned no label for {vn}")
                        labels[vn] = lbls[0] if isinstance(lbls, (list, np.ndarray)) else lbls
                else:
                    for vn in root_names:
                        lbl_choices = list(value_map.specs[vn].keys())
                        if not lbl_choices:
                            raise ValueError(f"no labels for {vn}")
                        labels[vn] = lbl_choices[int(rng.integers(len(lbl_choices)))]

                # Ground the complete state twice (two micro-realizations, same labels).
                spec1: Dict[str, Any] = {}
                spec2: Dict[str, Any] = {}
                identity = True
                for vn in root_names:
                    mv1 = value_map.ground(vn, labels[vn], rng=rng)
                    mv2 = value_map.ground(vn, labels[vn], rng=rng)
                    if not np.allclose(np.asarray(mv1, dtype=float), np.asarray(mv2, dtype=float)):
                        identity = False
                    spec1[vn] = {'labels': [labels[vn]], 'micro_values': mv1}
                    spec2[vn] = {'labels': [labels[vn]], 'micro_values': mv2}

                if identity:
                    # Every cell re-grounds to the same micro-state -> trivial, skip.
                    continue

                ints1 = builder._step_ground_or_passthrough(spec1, rng=rng)
                ints2 = builder._step_ground_or_passthrough(spec2, rng=rng)
                st1   = builder._step_low_level_model_execute(ints1, rng=rng)
                st2   = builder._step_low_level_model_execute(ints2, rng=rng)
                abs1  = builder._step_abstract(st1, rng=rng)
                abs2  = builder._step_abstract(st2, rng=rng)

                for out_var in high_level_model.variables:
                    if out_var in root_set:
                        continue  # score behavior on non-root outputs, not the clamped inputs
                    if out_var not in abs1 or out_var not in abs2:
                        continue
                    a1 = _to_float_array(abs1[out_var])
                    a2 = _to_float_array(abs2[out_var])
                    if a1 is None or a2 is None or a1.shape != a2.shape:
                        continue
                    var_scores.append(metric.measure(a1, a2))

            except Exception:
                pass

        return float(np.mean(var_scores)) if var_scores else float('nan')

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float(np.clip(result, 0.0, 1.0))


class DCCMetric(AnalyticalMetric):
    """
    Dynamic Causal Consistency (DCC) from NC-MCM.

    Two micro-states x1, x2 sharing abstract label c = τ(x1) = τ(x2) at time t
    must satisfy τ(M.step(x1)) = τ(M.step(x2)) - they must land in the
    same abstract cell after one simulation step.

    Contrast with BCC: BCC compares behavioral outputs using inner_metric
    (continuous comparison). DCC compares abstract next-state labels using
    SubspaceCheckMetric (hard label match), because DCC is a property of the
    transition structure, not output magnitude.

    Uses low_level_model.step() when available (preferred). Falls back to
    forward_with_interventions for low-level models that don't implement step().

    Only tests root (input) variables, since intervening on output variables
    has no meaning for the next-state check.

    For identity groundings (ContinuousValueMap, ideal ABM), ground() returns
    the same micro-value for the same label, so both micro-realizations are
    identical and DCC is trivially satisfied - those pairs are skipped and
    NaN is returned if no non-trivial pairs exist.

    Args:
        n_pairs: Number of within-cell pairs per variable.
        inner_metric: Kept for API compatibility; DCC always uses
            SubspaceCheckMetric for the next-state label comparison.
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, n_pairs: int = 10, inner_metric: str = 'hard',
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.n_pairs = n_pairs
        self.inner_metric = inner_metric

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .metrics import SubspaceCheckMetric
        from .valuemap import ContinuousValueMap
        from .primitives import SystemState as _SS

        cfg = _inner_config(config, self.inner_metric)
        check = SubspaceCheckMetric(normalize=False)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)

        is_continuous = isinstance(value_map, ContinuousValueMap)
        all_high_level_model_vars = list(high_level_model.variables.values())

        root_names = {
            name for name in high_level_model.variables
            if not high_level_model._nodes.get(name, {}).get('parents')
        }

        results: List[float] = []

        for vname in value_map.specs:
            if vname not in root_names:
                continue

            var_scores: List[float] = []

            if is_continuous:
                for _ in range(self.n_pairs):
                    try:
                        spec = sampler.sample_intervention(
                            all_high_level_model_vars, batch_size=1, force_all=True, rng=rng
                        )
                        entry = spec.get(vname, {})
                        lbls  = entry.get('labels', [])
                        if not len(lbls):
                            continue
                        lbl = lbls[0] if isinstance(lbls, (list, np.ndarray)) else lbls

                        mv1 = value_map.ground(vname, lbl, rng=rng)
                        mv2 = value_map.ground(vname, lbl, rng=rng)

                        if np.allclose(mv1, mv2):
                            # Identity grounding - trivially consistent, skip
                            continue

                        # Single-variable intervention scope, matching the discrete
                        # branch below.
                        spec1 = {vname: {'labels': [lbl], 'micro_values': mv1}}
                        spec2 = {vname: {'labels': [lbl], 'micro_values': mv2}}

                        ints1 = builder._step_ground_or_passthrough(spec1, rng=rng)
                        ints2 = builder._step_ground_or_passthrough(spec2, rng=rng)

                        if hasattr(low_level_model, 'step'):
                            st1 = low_level_model.step(_SS(values=ints1))
                            st2 = low_level_model.step(_SS(values=ints2))
                        else:
                            st1 = builder._step_low_level_model_execute(ints1, rng=rng)
                            st2 = builder._step_low_level_model_execute(ints2, rng=rng)

                        next1 = builder._step_abstract(st1, rng=rng)
                        next2 = builder._step_abstract(st2, rng=rng)

                        for out_var in high_level_model.variables:
                            if out_var not in next1 or out_var not in next2:
                                continue
                            a1 = _to_float_array(next1[out_var])
                            a2 = _to_float_array(next2[out_var])
                            if a1 is None or a2 is None or a1.shape != a2.shape:
                                continue
                            var_scores.append(check.measure(a1, a2))

                    except Exception:
                        pass

            else:
                for label, subspace in value_map.specs[vname].items():
                    for _ in range(self.n_pairs):
                        try:
                            mv1 = subspace.sample((1, subspace.dim), rng=rng)
                            mv2 = subspace.sample((1, subspace.dim), rng=rng)

                            spec1 = {vname: {'labels': [label], 'micro_values': mv1}}
                            spec2 = {vname: {'labels': [label], 'micro_values': mv2}}

                            ints1 = builder._step_ground_or_passthrough(spec1, rng=rng)
                            ints2 = builder._step_ground_or_passthrough(spec2, rng=rng)

                            if hasattr(low_level_model, 'step'):
                                st1 = low_level_model.step(_SS(values=ints1))
                                st2 = low_level_model.step(_SS(values=ints2))
                            else:
                                st1 = builder._step_low_level_model_execute(ints1, rng=rng)
                                st2 = builder._step_low_level_model_execute(ints2, rng=rng)

                            next1 = builder._step_abstract(st1, rng=rng)
                            next2 = builder._step_abstract(st2, rng=rng)

                            for out_var in high_level_model.variables:
                                if out_var not in next1 or out_var not in next2:
                                    continue
                                a1 = _to_float_array(next1[out_var])
                                a2 = _to_float_array(next2[out_var])
                                if a1 is None or a2 is None or a1.shape != a2.shape:
                                    continue
                                var_scores.append(check.measure(a1, a2))

                        except Exception:
                            pass

            if var_scores:
                results.append(float(np.mean(var_scores)))

        return float(np.mean(results)) if results else float('nan')

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float(np.clip(result, 0.0, 1.0))


# Probing
class ProbingMetric(AnalyticalMetric):
    """
    Linear probing (mechanistic interpretability).

    Trains a linear classifier/regressor on intermediate representations of
    the low-level model to predict the abstract labels of each high-level model variable.  High
    accuracy indicates that the abstract information is linearly accessible
    in the low-level representation.

    This metric is most informative for neural low-level models (where intermediate
    layer activations exist). For non-neural low-level models (physics simulations,
    logic circuits), probing is applied to the full output state.

    Args:
        probe_var: Name of the low-level model variable (micro-variable) to use as
            the probe input. None = concatenate all micro-variables.
        n_train: Number of training samples for the probe (default 200).
        inner_metric: Metric used to evaluate probe predictions (default 'mse').
        normalize: If True, map the score to [0, 1].
    """

    def __init__(
        self,
        probe_var: Optional[str] = None,
        n_train: int = 200,
        inner_metric: str = 'mse',
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.probe_var = probe_var
        self.n_train = n_train
        self.inner_metric = inner_metric

    @staticmethod
    def _fit_linear_probe(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit a least-squares linear probe and return the weight matrix."""
        X = np.column_stack([X_train, np.ones(len(X_train))])
        W, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
        return W

    @staticmethod
    def _predict_probe(W: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        X = np.column_stack([X_test, np.ones(len(X_test))])
        return X @ W

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .sampling import TopDownSampler
        from .metrics import get_metric, SubspaceCheckMetric

        cfg = _inner_config(config, self.inner_metric)
        metric = get_metric(cfg, normalize=False)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())
        td = TopDownSampler(value_map)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)

        n_test = min(50, max(5, n_samples))
        n_total = max(self.n_train + n_test, n_samples)

        # Collect (micro-state, abstract labels) pairs.
        X_data: List[np.ndarray] = []
        Y_data: Dict[str, List] = {v.name: [] for v in all_vars}

        for _ in range(n_total):
            spec = td.sample_intervention(all_vars, batch_size=1, force_all=True, rng=rng)
            try:
                ints = builder._step_ground_or_passthrough(spec, rng=rng)
                state = builder._step_low_level_model_execute(ints, rng=rng)
            except Exception:
                continue

            # Extract probe features from low-level model state
            if self.probe_var is not None:
                if self.probe_var not in state.values:
                    continue
                feat = np.asarray(state.values[self.probe_var], dtype=float).ravel()
            else:
                parts = []
                for val in state.values.values():
                    if isinstance(val, np.ndarray):
                        parts.append(val.ravel())
                if not parts:
                    continue
                feat = np.concatenate(parts)

            X_data.append(feat)
            for v in all_vars:
                lbl = spec.get(v.name, {}).get('labels')
                if lbl is not None:
                    lbl_val = lbl[0] if isinstance(lbl, (list, np.ndarray)) else lbl
                    Y_data[v.name].append(float(lbl_val) if np.isscalar(lbl_val) else float(np.mean(np.asarray(lbl_val, dtype=float))))
                else:
                    Y_data[v.name].append(np.nan)

        n_unique_labels = len(set(
            v for vals in Y_data.values() for v in vals if not np.isnan(v)
        ))
        if n_unique_labels <= 1:
            import warnings
            warnings.warn(
                "ProbingMetric: all samples have the same abstract label. "
                "The probe will trivially succeed (score≈0). "
                "This is expected for ContinuousValueMap with a single label.",
                UserWarning,
            )

        if len(X_data) < self.n_train + 2:
            return float('nan')

        X = np.stack(X_data)
        n_train = min(self.n_train, len(X) - 2)

        results: List[Any] = []
        for vname, y_list in Y_data.items():
            y = np.array(y_list)
            valid = ~np.isnan(y)
            if valid.sum() < 4:
                results.append(float('nan'))
                continue

            y_clean = y[valid]
            X_clean = X[valid]
            nt = min(n_train, len(X_clean) - 2)
            xt, xv = X_clean[:nt], X_clean[nt:]
            yt, yv = y_clean[:nt], y_clean[nt:]

            if len(xt) < 2 or len(xv) < 1:
                continue

            try:
                W = self._fit_linear_probe(xt, yt.reshape(-1, 1))
                y_pred = self._predict_probe(W, xv).ravel()
                # For exact-match metrics round continuous predictions to nearest
                # integer so that the discrete label comparison is meaningful.
                if isinstance(metric, SubspaceCheckMetric):
                    y_pred = np.round(y_pred)
                score = metric.measure(
                    yv.reshape(1, -1) if yv.ndim == 1 else yv,
                    y_pred.reshape(1, -1) if y_pred.ndim == 1 else y_pred,
                )
                results.append(float(score))
            except Exception:
                pass

        return float(np.mean(results)) if results else float('nan')


# Helper for InfidelityMetric
def _label_to_float(val: Any) -> Optional[float]:
    """Convert an abstract label (or list-of-one-label) to a scalar float.
    Returns None if the value is UNMAPPED or cannot be converted."""
    from .primitives import UNMAPPED

    if val is None or val is UNMAPPED:
        return None
    if isinstance(val, list):
        if not val:
            return None
        val = val[0]
    if val is UNMAPPED:
        return None
    try:
        arr = np.asarray(val, dtype=float).ravel()
        if len(arr) == 0:
            return None
        # Single element: exact scalar. Multi-element (e.g. final_populations): mean.
        return float(arr[0]) if len(arr) == 1 else float(arr.mean())
    except (TypeError, ValueError):
        return None

def _to_float_array(val: Any) -> Optional[np.ndarray]:
    """Convert any label/value to a 1-D float array, preserving all dimensions."""
    from .primitives import UNMAPPED
    if val is None or val is UNMAPPED:
        return None
    if isinstance(val, list):
        if not val:
            return None
        val = val[0] if len(val) == 1 else np.asarray(val)
    if val is UNMAPPED:
        return None
    try:
        arr = np.asarray(val, dtype=float).ravel()
        return arr if len(arr) > 0 else None
    except (TypeError, ValueError):
        return None

# Infidelity
class InfidelityMetric(AnalyticalMetric):
    """
    Attribution-based Infidelity for causal abstraction.

    For each sampled abstract input x and each input variable i, computes
    finite-difference attributions for both high-level model and low-level model:

        phi_i(E, x) = (E(x; x_i -> x_i + delta_i) - E(x)) / |delta_i|
        phi_i(M, x) = (tau(M(tau^{-1}(x; x_i -> x_i + delta_i)))
                       - tau(M(tau^{-1}(x)))) / |delta_i|

    The metric reports E_x[mean_i ||phi_i(E,x) - phi_i(M,x)||^2].

    For continuous labels, delta_i = max(delta_fraction * |x_i|, min_delta).
    For discrete (integer) labels, delta_i = 1.

    Args:
        delta_fraction: Relative perturbation size for continuous labels (default 0.1).
        min_delta: Minimum absolute delta for continuous labels (default 1e-3).
        output_vars: High-level model variable names to treat as outputs
            (None = all non-root variables).
        perturb_vars: Abstract input variable names to perturb (None = all root variables).
        normalize: If True, map the score to [0, 1].
    """

    def __init__(self, delta_fraction: float = 0.1,
                 min_delta: float = 1e-3,
                 output_vars: Optional[List[str]] = None,
                 perturb_vars: Optional[List[str]] = None,
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.delta_fraction = delta_fraction
        self.min_delta = min_delta
        self.output_vars = output_vars
        self.perturb_vars = perturb_vars

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler,
                 n_samples: int, config=None) -> float:
        from .paths import DiagramBuilder

        cfg = _inner_config(config, 'mse')
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())

        root_names = {
            name for name in high_level_model.variables
            if not high_level_model._nodes.get(name, {}).get('parents')
        }
        input_vars = [v for v in all_vars if v.name in root_names]
        if self.perturb_vars is not None:
            input_vars = [v for v in input_vars
                          if v.name in self.perturb_vars]

        output_vars = self.output_vars if self.output_vars is not None else [
            n for n in high_level_model.variables if n not in root_names
        ]

        if not input_vars or not output_vars:
            return float('nan')

        squared_diffs: List[float] = []

        for _ in range(n_samples):
            spec = sampler.sample_intervention(
                all_vars, batch_size=1, force_all=True, rng=rng)

            # Run original
            try:
                high_level_model_orig = builder._step_high_level_model_predict(spec, rng=rng)
                low_level_model_ints_orig = builder._step_ground_or_passthrough(
                    spec, rng=rng)
                low_level_model_state_orig = builder._step_low_level_model_execute(
                    low_level_model_ints_orig, rng=rng)
                abs_orig = builder._step_abstract(low_level_model_state_orig, rng=rng)
            except Exception:
                continue

            var_diffs: List[float] = []

            for var in input_vars:
                vname = var.name
                if vname not in spec:
                    continue

                orig_labels = spec[vname].get('labels', [])
                if not orig_labels:
                    continue
                orig_label = orig_labels[0]

                # Compute delta based on label type
                if isinstance(orig_label, (bool, np.bool_)):
                    delta = 1.0
                elif isinstance(orig_label, (int, np.integer)):
                    delta = 1.0
                elif isinstance(orig_label, np.ndarray) and orig_label.ndim == 0:
                    if np.issubdtype(orig_label.dtype, np.integer):
                        delta = 1.0
                    else:
                        delta = max(self.delta_fraction * abs(float(orig_label)), self.min_delta)
                elif isinstance(orig_label, np.ndarray):
                    delta = max(self.delta_fraction * float(np.linalg.norm(orig_label)), self.min_delta)
                else:
                    try:
                        delta = max(self.delta_fraction * abs(float(orig_label)), self.min_delta)
                    except Exception:
                        delta = self.min_delta

                # Build perturbed label
                try:
                    if isinstance(orig_label, np.ndarray):
                        if orig_label.ndim == 0 and np.issubdtype(orig_label.dtype, np.integer):
                            pert_label = (orig_label + 1).astype(orig_label.dtype)
                        else:
                            pert_label = orig_label + delta
                    elif isinstance(orig_label, (int, np.integer)):
                        pert_label = int(orig_label) + 1
                    else:
                        pert_label = float(orig_label) + delta
                except Exception:
                    continue

                # Build perturbed spec: only variable i changes
                pert_spec = {
                    k: ({'labels': [pert_label], 'micro_values': None}
                        if k == vname else v)
                    for k, v in spec.items()
                }

                # Run perturbed
                try:
                    high_level_model_pert = builder._step_high_level_model_predict(pert_spec, rng=rng)
                    low_level_model_ints_pert = builder._step_ground_or_passthrough(
                        pert_spec, rng=rng)
                    low_level_model_state_pert = builder._step_low_level_model_execute(
                        low_level_model_ints_pert, rng=rng)
                    abs_pert = builder._step_abstract(
                        low_level_model_state_pert, rng=rng)
                except Exception:
                    continue

                # Attribution disagreement per output variable
                out_diffs: List[float] = []
                for out_var in output_vars:
                    high_level_model_o = _label_to_float(high_level_model_orig.get(out_var))
                    high_level_model_p = _label_to_float(high_level_model_pert.get(out_var))
                    low_level_model_o = _label_to_float(abs_orig.get(out_var))
                    low_level_model_p = _label_to_float(abs_pert.get(out_var))

                    if any(v is None for v in [high_level_model_o, high_level_model_p, low_level_model_o, low_level_model_p]):
                        continue

                    phi_high_level_model = (high_level_model_p - high_level_model_o) / delta
                    phi_low_level_model = (low_level_model_p - low_level_model_o) / delta
                    out_diffs.append((phi_high_level_model - phi_low_level_model) ** 2)

                if out_diffs:
                    var_diffs.append(float(np.mean(out_diffs)))

            if var_diffs:
                squared_diffs.append(float(np.mean(var_diffs)))

        return float(np.mean(squared_diffs)) if squared_diffs else float('nan')


# Symbion
class SymbionMetric(AnalyticalMetric):
    """
    Symbion / concolic coverage testing for discrete systems with finite
    abstract input spaces.

    Exhaustively enumerates every combination of root-variable abstract
    labels, runs both high-level model and low-level model, and reports the fraction of input
    combinations on which they disagree on any output variable.
    """

    def __init__(self, inner_metric: str = 'hard', output_vars: Optional[List[str]] = None,
                 normalize: bool = True):
        super().__init__(normalize=normalize)
        self.inner_metric = inner_metric
        self.output_vars = output_vars

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder
        from .primitives import UNMAPPED
        from .metrics import SubspaceCheckMetric

        cfg = _inner_config(config, self.inner_metric)
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)

        metric = SubspaceCheckMetric()

        root_names = [
            name for name in high_level_model.variables
            if not high_level_model._nodes.get(name, {}).get('parents')
        ]

        if self.output_vars is not None:
            output_names = self.output_vars
        else:
            output_names = [
                name for name in high_level_model.variables
                if name not in root_names
            ]

        # Collect ordered label lists per root variable
        label_lists: List[List[Any]] = []
        for vname in root_names:
            if vname in value_map.specs:
                label_lists.append(list(value_map.specs[vname].keys()))
            else:
                label_lists.append([])

        all_combos = list(itertools.product(*label_lists))
        n_tested = len(all_combos)

        if n_tested == 0:
            return float('nan')

        n_failures = 0  # number of combos with at least one output disagreement

        for combo in all_combos:
            spec: Dict[str, Any] = {
                vname: {'labels': [label], 'micro_values': None}
                for vname, label in zip(root_names, combo)
            }

            try:
                high_level_model_out = builder._step_high_level_model_predict(spec, rng=rng)
            except Exception:
                continue

            try:
                low_level_model_ints = builder._step_ground_or_passthrough(spec, rng=rng)
                low_level_model_state = builder._step_low_level_model_execute(low_level_model_ints, rng=rng)
                low_level_model_abs = builder._step_abstract(low_level_model_state, rng=rng)
            except Exception:
                continue

            combo_failed = False
            for out_var in output_names:
                if out_var not in high_level_model_out or out_var not in low_level_model_abs:
                    continue

                a = high_level_model_out[out_var]
                b = low_level_model_abs[out_var]

                if a is UNMAPPED or b is UNMAPPED:
                    continue

                # Normalize to a flat float array of shape (1,)
                def _to_array(v):
                    if isinstance(v, list):
                        v = v[0] if len(v) >= 1 else v
                    if v is UNMAPPED:
                        return None
                    try:
                        # Compare all output components (covers vector outputs).
                        return np.asarray(v, dtype=float).ravel()
                    except Exception:
                        return None

                a_arr = _to_array(a)
                b_arr = _to_array(b)

                if a_arr is None or b_arr is None:
                    continue

                try:
                    score = metric.measure(a_arr, b_arr)
                    if score > 0.0:
                        combo_failed = True
                        break  # one disagreement is enough to flag this combo
                except Exception:
                    pass

            if combo_failed:
                n_failures += 1

        return float(n_failures) / float(n_tested)

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float(np.clip(result, 0.0, 1.0))


# Relational fidelity
class RelationalFidelityMetric(AnalyticalMetric):
    """
    Relational Fidelity: Pearson correlation between Δ_E and Δ_M.

    For each pair of independently sampled inputs (spec1, spec2):
        Δ_E[var][d] = E(spec2)[var][d] − E(spec1)[var][d]
        Δ_M[var][d] = tau(M(spec2))[var][d] − tau(M(spec1))[var][d]

    Pearson correlation is computed independently per output dimension d, then
    averaged across dimensions and variables. This preserves all channels of
    vector-valued outputs (e.g. [prey, predator] for PP) without collapsing.

    Args:
        output_vars: Output variables to score (None = all variables, including
            roots; suites pass an explicit output list). Root channels are set
            identically in E and M, so leaving them in biases the score toward 0.
        n_pairs: Number of input pairs (None = use n_samples from compute()).
        normalize: If True, map the score to [0, 1].
    """

    def __init__(
        self,
        output_vars: Optional[List[str]] = None,
        n_pairs: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.output_vars = output_vars
        self.n_pairs = n_pairs  # None = use whatever n_samples is passed to compute()

    def _compute(self, high_level_model, low_level_model, value_map, cg_map, sampler, n_samples, config=None):
        from .paths import DiagramBuilder

        effective_n = self.n_pairs if self.n_pairs is not None else n_samples
        if effective_n < 50:
            import warnings
            warnings.warn(
                f"RelationalFidelityMetric: {effective_n} pairs is below the "
                "recommended minimum of 50 for reliable Pearson correlation.",
                UserWarning,
            )

        cfg = _inner_config(config, 'mse')
        builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, cfg)
        rng = np.random.default_rng(cfg.seed)
        all_vars = list(high_level_model.variables.values())
        output_vars = self.output_vars or list(high_level_model.variables.keys())

        accum: Dict[str, Dict[str, List[np.ndarray]]] = {
            v: {'high_level_model': [], 'low_level_model': []} for v in output_vars
        }

        for _ in range(effective_n):
            spec1 = sampler.sample_intervention(all_vars, batch_size=1, force_all=True, rng=rng)
            spec2 = sampler.sample_intervention(all_vars, batch_size=1, force_all=True, rng=rng)

            try:
                high_level_model_out1 = builder._step_high_level_model_predict(spec1, rng=rng)
                high_level_model_out2 = builder._step_high_level_model_predict(spec2, rng=rng)
            except Exception:
                continue

            try:
                ints1 = builder._step_ground_or_passthrough(spec1, rng=rng)
                st1   = builder._step_low_level_model_execute(ints1, rng=rng)
                abs1  = builder._step_abstract(st1, rng=rng)

                ints2 = builder._step_ground_or_passthrough(spec2, rng=rng)
                st2   = builder._step_low_level_model_execute(ints2, rng=rng)
                abs2  = builder._step_abstract(st2, rng=rng)
            except Exception:
                continue

            for var in output_vars:
                h1 = _to_float_array(high_level_model_out1.get(var))
                h2 = _to_float_array(high_level_model_out2.get(var))
                l1 = _to_float_array(abs1.get(var))
                l2 = _to_float_array(abs2.get(var))

                if any(v is None for v in [h1, h2, l1, l2]):
                    continue
                if h1.shape != h2.shape or l1.shape != l2.shape or h1.shape != l1.shape:
                    continue

                accum[var]['high_level_model'].append(h2 - h1)
                accum[var]['low_level_model'].append(l2 - l1)

        all_corrs: List[float] = []
        for var in output_vars:
            dh_list = accum[var]['high_level_model']
            dl_list = accum[var]['low_level_model']
            if len(dh_list) < 4:
                continue
            dh = np.stack(dh_list)  # (n_pairs, d)
            dl = np.stack(dl_list)  # (n_pairs, d)
            for d in range(dh.shape[1]):
                corr = np.corrcoef(dh[:, d], dl[:, d])[0, 1]
                if not np.isnan(corr):
                    all_corrs.append(float(corr))

        return float(np.mean(all_corrs)) if all_corrs else float('nan')

    def _normalize_result(self, result: float) -> float:
        if np.isnan(result):
            return float('nan')
        return float((1.0 - result) / 2.0)

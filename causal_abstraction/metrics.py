"""
Defines the dissimilarity metrics D for evaluating causal abstractions, supporting
both point-wise comparisons and distributional divergences.

Every metric accepts ``normalize=False`` in its constructor. When ``normalize=True``
most metrics map ``measure()`` into [0, 1], where 0 = perfect and 1 = maximal error,
via an affine transform for bounded metrics or ``tanh`` for unbounded ones. This is a
per-subclass convention, not a guarantee enforced by the base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from .config import EvaluationConfig
from .primitives import ProbabilityDistribution, EmpiricalDistribution
from .spaces import Subspace
from .spaces import UnionSubspace
import logging as _logging

_metrics_logger = _logging.getLogger(__name__)


class EvaluationMetric(ABC):
    """
    Abstract base class for measuring the distance/divergence between two outcomes.

    Subclasses implement ``_measure(y_true, y_pred) -> float``.  The public
    ``measure()`` method calls ``_measure()`` and, when ``normalize=True``,
    passes the result through ``_normalize_score()`` to obtain a value in
    [0, 1] where 0 = perfect and 1 = maximal error.
    """
    best_value: float = 0.0
    lower_is_better: bool = True

    def __init__(self, normalize: bool = True):
        self._normalize = normalize

    # Public API
    def measure(self, y_true: Any, y_pred: Any) -> float:
        """Compute the metric, optionally normalizing to [0, 1]."""
        y_true, y_pred = self._coerce_shapes(y_true, y_pred)
        score = self._measure(y_true, y_pred)
        if self._normalize:
            return self._normalize_score(score)
        return score

    @staticmethod
    def _coerce_shapes(y_true, y_pred):
        """
        Best-effort shape alignment before scoring.

        If both inputs are plain ndarrays with compatible element counts
        but different shapes (e.g. (5,) vs (5,1)), reshape the simpler
        one to match the other.  This prevents spurious inf scores from
        shape != shape guards in _measure().
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            return y_true, y_pred
        if y_true.shape == y_pred.shape:
            return y_true, y_pred
        # Only coerce if element counts match
        if y_true.size != y_pred.size:
            return y_true, y_pred
        # Reshape the lower-dimensional one to match the higher-dimensional one
        if y_true.ndim < y_pred.ndim:
            try:
                y_true = y_true.reshape(y_pred.shape)
            except ValueError:
                pass
        elif y_pred.ndim < y_true.ndim:
            try:
                y_pred = y_pred.reshape(y_true.shape)
            except ValueError:
                pass
        return y_true, y_pred

    # Subclass contract
    @abstractmethod
    def _measure(self, y_true: Any, y_pred: Any) -> float:
        """Raw metric computation (no normalization)."""
        pass

    def _normalize_score(self, score: float) -> float:
        """
        Map score to [0, 1]. Default: ``tanh(score)`` for non-negative,
        lower-is-better metrics whose range is [0, ∞).
        Overridden in subclasses that need different behavior.
        """
        if np.isnan(score):
            return float('nan')
        if np.isinf(score):
            return 1.0 if score > 0 else 0.0
        return float(np.tanh(score))


class PrecisionMetric:
    """
    Measures the specificity of the abstraction mapping.
    Score = 1 - (Volume(Subspace) / Volume(LabeledUnion)), where LabeledUnion is
    the summed volume of the variable's labeled subspaces (passed in as
    ``total_volume`` by the engine), not the full micro domain.
    """

    def __init__(self, normalize: bool = True):
        self._normalize = normalize  # kept for API consistency; already [0,1]

    def measure(self, subspace: Subspace, total_volume: float = 1.0) -> float:
        """Measure how specific ``subspace`` is within the total domain.

        Args:
            subspace: The micro-subspace associated with a label.
            total_volume: Normalization volume. The engine passes the summed
                volume of the variable's labeled subspaces, not the full domain.

        Returns:
            A precision score in [0, 1]: 1 = maximally specific, 0 = fills the domain.
        """
        return self._measure(subspace, total_volume)

    def _measure(self, subspace: Subspace, total_volume: float = 1.0) -> float:
        if isinstance(subspace, UnionSubspace):
            import warnings
            warnings.warn("PrecisionMetric may be inaccurate for UnionSubspace due to overlap.", UserWarning)

        vol = subspace.volume()

        if vol == 0:
            return 1.0  # Point mass
        if np.isinf(vol):
            return 0.0
        if np.isinf(total_volume):
            return 1.0  # Finite subspace in infinite space

        ratio = vol / total_volume
        return 1.0 - ratio


class SubspaceCheckMetric(EvaluationMetric):
    """Exact match scoring. Returns fraction of mismatches."""

    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.mode()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.mode()

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            # A shape mismatch is a total failure to match, not a perfect match.
            # Return the maximal mismatch fraction (1.0 = worst), mirroring the
            # worst-score convention MSEMetric uses (inf) on a shape mismatch,
            # instead of 0.0 (= perfect).
            return 1.0

        matches = (y_true == y_pred)
        return 1.0 - float(np.mean(matches))

    def _normalize_score(self, score: float) -> float:
        return score


class L2Metric(EvaluationMetric):
    """
    Relative squared error: mean over samples of ``||y_true - y_pred||^2 / ||y_true||^2``.

    Despite the "L2" label this is not a Euclidean distance (there is no square root);
    it is an NMSE-like relative squared error. On the 1-D concatenated engine path it
    collapses to a single global ratio ``sum(diff^2) / sum(y_true^2)``.
    """

    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.as_array()

        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.as_array()

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Standardize shapes
        if y_true.ndim > 1:
            y_true = y_true.reshape(y_true.shape[0], -1)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        elif y_true.ndim == 0:
            y_true = y_true.reshape(1, 1)
            y_pred = y_pred.reshape(1, 1)

        diff = y_true - y_pred
        error_norm_sq = np.sum(diff ** 2, axis=-1)  # ||diff||^2
        base_norm_sq = np.sum(y_true ** 2, axis=-1)  # ||y_true||^2
        normalized_error = error_norm_sq / (base_norm_sq + 1e-9)
        return float(np.mean(normalized_error))

    def _normalize_score(self, score: float) -> float:
        return float(np.tanh(score))


class JSDivergenceMetric(EvaluationMetric):
    """
    Measures divergence between probability distributions.
    Continuous samples are binned into histograms.
    """
    best_value = 0.0
    lower_is_better = True

    def __init__(self, n_bins: int = 20, n_samples: int = 1000, normalize: bool = True,
                 is_distribution: bool = False):
        super().__init__(normalize=normalize)
        self.n_bins = n_bins
        self.n_samples = n_samples
        # When True, inputs are treated as categorical probability vectors and
        # compared element-wise; otherwise they are sample arrays binned into a
        # histogram.
        self.is_distribution = is_distribution

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, EmpiricalDistribution):
            y_true = y_true.sample(self.n_samples)
        if isinstance(y_pred, EmpiricalDistribution):
            y_pred = y_pred.sample(self.n_samples)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.size < 4 and not self.is_distribution:
            _metrics_logger.debug("JSDivergenceMetric: need ≥4 samples for histogram, got %d. Returning NaN.", y_true.size)
            return float("nan")

        # Discrete probs
        if self.is_distribution:
            p = y_true
            q = y_pred
        # Continuous samples -> histogram
        else:
            v1 = y_true.flatten().astype(float)
            v2 = y_pred.flatten().astype(float)

            n = len(v1) + len(v2)
            adaptive_bins = min(self.n_bins, max(2, int(np.ceil(np.log2(n) + 1))))

            # Shared min-data guard with KLDivergenceMetric (≥5 pooled samples/bin).
            if n < 5 * adaptive_bins:
                return float('nan')  # Not enough data

            min_val = min(v1.min(), v2.min())
            max_val = max(v1.max(), v2.max())

            if np.isclose(min_val, max_val):
                return 0.0

            bins = np.linspace(min_val, max_val, adaptive_bins + 1)
            p, _ = np.histogram(v1, bins=bins, density=True)
            q, _ = np.histogram(v2, bins=bins, density=True)

            p = p / (np.sum(p) + 1e-9)
            q = q / (np.sum(q) + 1e-9)

        with np.errstate(divide="ignore"):
            # scipy's jensenshannon returns the JS distance (the square root of
            # the divergence); square it to report the Jensen-Shannon divergence
            # this metric's name claims. With base 2 the divergence stays in [0, 1].
            js_distance = jensenshannon(p, q, base=2.0)

        if np.isnan(js_distance):
            return 1.0
        return float(js_distance ** 2)

    def _normalize_score(self, score: float) -> float:
        return score


class KLDivergenceMetric(EvaluationMetric):
    """
    Measures KL-divergence between probability distributions.

    y_true is treated as P (actual/target distribution).
    y_pred is treated as Q (predicted/approximate distribution).
    """
    def __init__(self, n_bins: int = 20, n_samples: int = 1000,
                 epsilon: float = 1e-10, normalize: bool = True,
                 is_distribution: bool = False):
        super().__init__(normalize=normalize)
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.epsilon = epsilon
        # When True, inputs are treated as discrete probability vectors (see JSDivergenceMetric).
        self.is_distribution = is_distribution

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if hasattr(y_true, 'sample'):
            y_true = y_true.sample(self.n_samples)
        if hasattr(y_pred, 'sample'):
            y_pred = y_pred.sample(self.n_samples)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.size < 4 and not self.is_distribution:
            _metrics_logger.debug("KLDivergenceMetric: need ≥4 samples for histogram, got %d. Returning NaN.", y_true.size)
            return float("nan")

        # If inputs are already discrete probabilities
        if self.is_distribution:
            p = y_true
            q = y_pred
        else:  # Convert to hist
            v1 = y_true.flatten().astype(float)
            v2 = y_pred.flatten().astype(float)

            min_val = min(v1.min(), v2.min())
            max_val = max(v1.max(), v2.max())

            # Sturges' rule, capped at requested n_bins
            n = len(v1) + len(v2)
            adaptive_bins = min(self.n_bins, max(2, int(np.ceil(np.log2(n) + 1))))

            if n < 5 * adaptive_bins:
                return float('nan')  # Not enough data

            # Edge case: All values are identical (point mass)
            if np.isclose(min_val, max_val):
                return 0.0 if np.isclose(v1.mean(), v2.mean()) else float('inf')

            bins = np.linspace(min_val, max_val, adaptive_bins + 1)

            p, _ = np.histogram(v1, bins=bins, density=True)
            q, _ = np.histogram(v2, bins=bins, density=True)

        # Stability/smoothing + renormalization
        p = p.astype(float) + self.epsilon
        p = p / p.sum()
        q = q.astype(float) + self.epsilon
        q = q / q.sum()

        kl_div = entropy(pk=p, qk=q, base=2.0)

        if np.isnan(kl_div):
            return float('inf')

        return float(kl_div)


class R2Metric(EvaluationMetric):
    """
    R² coefficient of determination.  Range: (−∞, 1] where 1 = perfect.

    When normalize=True: maps to [0, 1] via ``clip((1 − R²) / 2, 0, 1)``.
    """
    lower_is_better = False
    best_value = 1.0

    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.as_array()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.as_array()

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if y_true.shape != y_pred.shape:
            return float("nan")

        if len(y_true) < 2:
            _metrics_logger.debug(
                "R2Metric: needs >=2 samples (got %d). R2 is undefined for single-sample "
                "pairwise comparisons (e.g. IIA/BCC inner metric); use 'mse' or 'l2' there.",
                len(y_true),
            )
            return float("nan")

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0

        return float(1.0 - ss_res / ss_tot)

    def _normalize_score(self, score: float) -> float:
        if np.isnan(score):
            return float('nan')
        return float(np.clip((1.0 - score) / 2.0, 0.0, 1.0))


class MSEMetric(EvaluationMetric):
    """Mean Squared Error"""
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.mode()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.mode()

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            return float("inf")

        return float(np.mean((y_true - y_pred) ** 2))


class RMSEMetric(MSEMetric):
    """Root Mean Squared Error"""
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true, y_pred) -> float:
        return np.sqrt(super()._measure(y_true, y_pred))


class NMSEMetric(MSEMetric):
    """Normalized Mean Squared Error (MSE / Var)"""
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true, y_pred) -> float:
        y_arr = np.asarray(y_true, dtype=float).ravel()
        if len(y_arr) < 2:
            _metrics_logger.debug("NMSEMetric: need ≥2 samples, got %d. Returning NaN.", len(y_arr))
            return float("nan")
        mse = super()._measure(y_true, y_pred)
        var = np.var(y_true)
        if var < 1e-12:
            return 0.0 if np.allclose(y_true, y_pred) else float('inf')
        return mse / var


class MMDMetric(EvaluationMetric):
    """
    Maximum Mean Discrepancy with RBF kernel. Returns the unbiased MMD² estimate.
    """
    best_value = 0.0
    lower_is_better = True
    def __init__(self, bandwidth: Optional[float] = None, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.bandwidth = bandwidth

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        sq_dist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return np.exp(-sq_dist / (2.0 * sigma ** 2))

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.as_array()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.as_array()

        X = np.asarray(y_true, dtype=float)
        Y = np.asarray(y_pred, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]

        n, m = len(X), len(Y)
        if n < 2 or m < 2:
            return 0.0

        if self.bandwidth is None:
            all_data = np.vstack([X, Y])
            pairwise = np.sum((all_data[:, None] - all_data[None]) ** 2, axis=-1)
            upper = pairwise[np.triu_indices_from(pairwise, k=1)]
            median_sq = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
            sigma = float(np.sqrt(median_sq / 2.0))
        else:
            sigma = float(self.bandwidth)

        K_xx = self._rbf_kernel(X, X, sigma)
        K_yy = self._rbf_kernel(Y, Y, sigma)
        K_xy = self._rbf_kernel(X, Y, sigma)

        mmd2 = (K_xx.sum() - K_xx.diagonal().sum()) / (n * (n - 1))
        mmd2 += (K_yy.sum() - K_yy.diagonal().sum()) / (m * (m - 1))
        mmd2 -= 2.0 * float(np.mean(K_xy))
        return float(max(0.0, mmd2))


class TrajectoryMSEMetric(EvaluationMetric):
    """
    MSE along a shared time axis between two trajectories.

    y_true and y_pred should be array-like of shape (T,) or (T, d).
    If the lengths differ, the common prefix is used.
    """
    best_value = 0.0
    lower_is_better = True
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        s1 = np.asarray(y_true, dtype=float)
        s2 = np.asarray(y_pred, dtype=float)
        min_len = min(len(s1), len(s2))
        return float(np.mean((s1[:min_len] - s2[:min_len]) ** 2))


class DTWMetric(EvaluationMetric):
    """
    Dynamic Time Warping distance between two time series.

    y_true and y_pred can be 1-D (T,) or 2-D (T, d) arrays.

    Args:
        normalize_length: If True, divide the raw DTW distance by (|s1| + |s2|)
            to account for sequence length.
        normalize: If True, map the result to [0, 1].
    """
    best_value = 0.0
    lower_is_better = True

    def __init__(self, normalize_length: bool = True, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.normalize_length = normalize_length

    @staticmethod
    def _dtw(s1: np.ndarray, s2: np.ndarray) -> float:
        if s1.ndim == 1:
            s1 = s1[:, None]
        if s2.ndim == 1:
            s2 = s2[:, None]
        n, m = len(s1), len(s2)
        # Local cost matrix via broadcasting
        cost = np.linalg.norm(s1[:, None] - s2[None], axis=-1)  # (n, m)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(
                    dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
                )
        return float(dtw[n, m])

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        s1 = np.asarray(y_true, dtype=float)
        s2 = np.asarray(y_pred, dtype=float)
        dist = self._dtw(s1, s2)
        if self.normalize_length:
            dist /= max(len(s1) + len(s2), 1)
        return dist


class TemporalAutocorrelationMetric(EvaluationMetric):
    """
    RMSE between the autocorrelation functions (ACF) of two 1-D time series.

    Lower is better; 0 = identical ACF profiles.
    """
    best_value = 0.0
    lower_is_better = True

    def __init__(self, n_lags: int = 20, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.n_lags = n_lags

    @staticmethod
    def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
        x = x - float(np.mean(x))
        n = len(x)
        var = float(np.dot(x, x))
        if var < 1e-12 or n <= 1:
            return np.zeros(n_lags + 1)
        full = np.correlate(x, x, mode='full')
        acf = full[n - 1: n - 1 + min(n_lags + 1, n)] / var
        if len(acf) < n_lags + 1:
            acf = np.pad(acf, (0, n_lags + 1 - len(acf)))
        return acf

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        s1 = np.asarray(y_true, dtype=float).ravel()
        s2 = np.asarray(y_pred, dtype=float).ravel()
        acf1 = self._acf(s1, self.n_lags)
        acf2 = self._acf(s2, self.n_lags)
        return float(np.sqrt(np.mean((acf1 - acf2) ** 2)))


class SpectralMetric(EvaluationMetric):
    """
    RMSE between normalized power spectral densities.

    Both signals are zero-padded to the same length before the FFT.

    Args:
        normalize_psd: If True, normalize each PSD to sum to 1 before comparison.
        normalize: If True, map the result to [0, 1].
    """
    best_value = 0.0
    lower_is_better = True

    def __init__(self, normalize_psd: bool = True, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.normalize_psd = normalize_psd

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        s1 = np.asarray(y_true, dtype=float).ravel()
        s2 = np.asarray(y_pred, dtype=float).ravel()
        n = max(len(s1), len(s2))
        # Remove the mean so the DC bin does not dominate, then compare the
        # oscillation spectrum only (drop the DC bin).
        s1 = s1 - s1.mean()
        s2 = s2 - s2.mean()
        psd1 = np.abs(np.fft.rfft(s1, n=n)) ** 2
        psd2 = np.abs(np.fft.rfft(s2, n=n)) ** 2
        psd1 = psd1[1:]
        psd2 = psd2[1:]
        if self.normalize_psd:
            psd1 = psd1 / (float(np.sum(psd1)) + 1e-12)
            psd2 = psd2 / (float(np.sum(psd2)) + 1e-12)
        return float(np.sqrt(np.mean((psd1 - psd2) ** 2)))


class ConditionalIndependenceMetric(EvaluationMetric):
    """
    Normalized HSIC (Hilbert-Schmidt Independence Criterion).

    Measures whether the residuals (M output - E output) are
    statistically independent of the high-level model outputs (used as a proxy
    for the inputs). Low HSIC indicates that errors are uniformly
    distributed across the input space rather than concentrated in
    specific input regions, reflecting global rather than locally
    compensating agreement.

    Score is the normalized HSIC, so 0 = perfect
    (residuals fully independent of inputs).

    Naming: this class backs the ``EvaluationConfig(metric='hsic')`` option (see
    :func:`get_metric`) and the ``"HSIC"`` label used in the results JSON and
    figures; there is no separate ``HSICMetric`` class.
    """
    best_value = 0.0
    lower_is_better = True
    def __init__(self, bandwidth: Optional[float] = None, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.bandwidth = bandwidth

    def _kernel(self, X: np.ndarray, sigma: float) -> np.ndarray:
        sq = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
        return np.exp(-sq / (2.0 * sigma ** 2))

    def _bw(self, X: np.ndarray) -> float:
        sq = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
        tri = sq[np.triu_indices_from(sq, k=1)]
        pos = tri[tri > 0]
        return float(np.sqrt(float(np.median(pos)) / 2.0)) if len(pos) else 1.0

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.as_array()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.as_array()

        X = np.asarray(y_true, dtype=float)  # high-level model outputs (used as inputs here)
        Y = np.asarray(y_pred, dtype=float)  # low-level model outputs

        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]

        n = min(len(X), len(Y))
        if n < 20:
            return float('nan')

        X, Y = X[:n], Y[:n]

        # Residuals: r = M output - E output
        R = Y - X  # (n, d)

        # Test independence between residuals R and high-level model inputs X
        # (X here plays the role of the "input" u)
        sx = self.bandwidth if self.bandwidth is not None else self._bw(X)
        sr = self.bandwidth if self.bandwidth is not None else self._bw(R)
        K = self._kernel(X, sx)
        L = self._kernel(R, sr)
        H = np.eye(n) - np.ones((n, n)) / n
        HKH = H @ K @ H
        HLH = H @ L @ H
        hsic = float(np.trace(HKH @ HLH)) / (n - 1) ** 2
        nk = float(np.sqrt(max(float(np.trace(HKH @ HKH)), 0.0))) / (n - 1)
        nl = float(np.sqrt(max(float(np.trace(HLH @ HLH)), 0.0))) / (n - 1)
        if nk < 1e-10 or nl < 1e-10:
            return 0.0
        return float(np.clip(hsic / (nk * nl), 0.0, 1.0))

    def _normalize_score(self, score: float) -> float:
        return score


class VarianceDecompositionMetric(EvaluationMetric):
    """
    Unexplained variance fraction: Var(y_true − y_pred) / Var(y_true).
    Related to (1 − R²) but uses the variance of residuals rather than SS_res
    (invariant to a constant offset bias).

    Note: despite the name, this is a fraction-of-variance-unexplained (NMSE-like)
    statistic, not a Sobol/ANOVA partition of output variance into per-input
    first-order and interaction contributions. That sensitivity partition is
    implemented separately in ``SobolSensitivityMetric``.
    """
    best_value = 0.0
    lower_is_better = True

    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _measure(self, y_true: Any, y_pred: Any) -> float:
        if isinstance(y_true, ProbabilityDistribution):
            y_true = y_true.as_array()
        if isinstance(y_pred, ProbabilityDistribution):
            y_pred = y_pred.as_array()

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if len(y_true) < 2:
            _metrics_logger.debug("VarianceDecompositionMetric: need ≥2 samples, got %d. Returning NaN.", len(y_true))
            return float("nan")

        if y_true.shape != y_pred.shape:
            return float("nan")

        var_y = float(np.var(y_true))
        if var_y < 1e-12:
            return 0.0 if np.allclose(y_true, y_pred) else 1.0
        return float(np.clip(np.var(y_true - y_pred) / var_y, 0.0, None))

    def _normalize_score(self, score: float) -> float:
        # The raw unexplained-variance ratio is unbounded above; squash it into
        # [0, 1) with tanh so the metric honors the module's [0, 1] convention.
        return float(np.tanh(score))


def get_metric(config: EvaluationConfig, normalize: bool = True) -> EvaluationMetric:
    """Construct the dissimilarity metric D selected by ``config.metric``.

    Args:
        config: Evaluation config whose ``metric`` string selects the class.
        normalize: Passed to the metric so its score maps to [0, 1].

    Returns:
        The matching ``EvaluationMetric`` instance.

    Raises:
        ValueError: If ``config.metric`` is not a recognized metric string.
    """
    mode = config.metric
    if mode == 'hard':
        return SubspaceCheckMetric(normalize=normalize)
    elif mode == 'l2':
        return L2Metric(normalize=normalize)
    elif mode == 'mse':
        return MSEMetric(normalize=normalize)
    elif mode == 'rmse':
        return RMSEMetric(normalize=normalize)
    elif mode == 'nmse':
        return NMSEMetric(normalize=normalize)
    elif mode == 'r2':
        return R2Metric(normalize=normalize)
    elif mode == 'kl':
        return KLDivergenceMetric(normalize=normalize)
    elif mode == 'jsd':
        return JSDivergenceMetric(n_bins=config.jsd_bins, normalize=normalize)
    elif mode == 'mmd':
        return MMDMetric(normalize=normalize)
    elif mode == 'traj_mse':
        return TrajectoryMSEMetric(normalize=normalize)
    elif mode == 'dtw':
        return DTWMetric(normalize=normalize)
    elif mode == 'autocorr':
        return TemporalAutocorrelationMetric(normalize=normalize)
    elif mode == 'spectral':
        return SpectralMetric(normalize=normalize)
    elif mode == 'hsic':
        return ConditionalIndependenceMetric(normalize=normalize)
    elif mode == 'var_decomp':
        return VarianceDecompositionMetric(normalize=normalize)
    else:
        raise ValueError(f"Unknown metric: {mode}")

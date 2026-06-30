"""Subspace classes representing constraints on variables and valid domains."""

import math
import warnings
from typing import Tuple, List, Optional

import numpy as np


class Subspace:
    """
    Base class for subspaces.
    Subspaces represent regions in the micro-variable space used for grounding and abstraction.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Returns boolean array indicating if x is within the subspace."""
        raise NotImplementedError

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generates random samples uniformly (where possible) from the subspace."""
        raise NotImplementedError

    def volume(self) -> float:
        """Returns the hypervolume of the subspace."""
        raise NotImplementedError

    def loss(self, x: np.ndarray) -> np.ndarray:
        """Returns a distance metric from x to the subspace (0 if inside)."""
        raise NotImplementedError

    def centroid(self) -> np.ndarray:
        """Returns the geometric center of the subspace."""
        return self.sample((1, self.dim), rng=None).squeeze(0)

    def _get_rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        """Helper to ensure a valid RNG exists."""
        return rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _ensure_batched(x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensures the input array is 2D (batched) and returns a flag if it was modified."""
        x = np.asarray(x)
        if x.ndim == 0:
            return x[np.newaxis, np.newaxis], True  # Scalar -> (1, 1)
        if x.ndim == 1:
            return x[np.newaxis, :], True  # (D,) -> (1, D)
        return x, False


class RectSubspace(Subspace):
    """A hyper-rectangular subspace defined by min/max bounds per dimension."""

    def __init__(self, *intervals: tuple):
        """Build a hyper-rectangle from one ``(low, high)`` interval per dimension.

        Raises:
            ValueError: If no intervals are given.
        """
        if not intervals:
            raise ValueError("RectSubspace requires at least one interval.")
        super().__init__(dim=len(intervals))
        self.intervals = intervals
        self.lows = np.array([i[0] for i in intervals], dtype=float)
        self.highs = np.array([i[1] for i in intervals], dtype=float)

    def contains(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        if x.shape[1] != self.dim:
            # If mismatch, it cannot be contained
            return np.zeros(x.shape[0], dtype=bool) if not was_unbatched else False

        in_bounds = np.ones((x.shape[0], self.dim), dtype=bool)
        in_bounds &= (x >= self.lows)
        in_bounds &= (x <= self.highs)

        result = np.all(in_bounds, axis=1)
        return result[0] if was_unbatched else result

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample uniformly within the rectangle (exponential/normal on infinite bounds).

        Raises:
            ValueError: If ``shape``'s dimension does not match the subspace dim.
        """
        rng = self._get_rng(rng)
        batch_size, d = shape
        if d != self.dim:
            raise ValueError(f"Requested dim {d} != subspace dim {self.dim}")

        samples = np.empty(shape, dtype=float)

        # Handle finite bounds
        finite_mask = np.isfinite(self.lows) & np.isfinite(self.highs)
        if np.any(finite_mask):
            lows = self.lows[finite_mask]
            highs = self.highs[finite_mask]
            rand_vals = rng.random((batch_size, np.sum(finite_mask)))
            samples[:, finite_mask] = lows + rand_vals * (highs - lows)

        # Handle infinite bounds using exponential or normal distributions
        right_inf_mask = np.isfinite(self.lows) & np.isinf(self.highs)
        if np.any(right_inf_mask):
            samples[:, right_inf_mask] = (
                    self.lows[right_inf_mask] + rng.exponential(1.0, (batch_size, np.sum(right_inf_mask)))
            )

        left_inf_mask = np.isinf(self.lows) & np.isfinite(self.highs)
        if np.any(left_inf_mask):
            samples[:, left_inf_mask] = (
                    self.highs[left_inf_mask] - rng.exponential(1.0, (batch_size, np.sum(left_inf_mask)))
            )

        full_inf_mask = np.isinf(self.lows) & np.isinf(self.highs)
        if np.any(full_inf_mask):
            samples[:, full_inf_mask] = rng.standard_normal((batch_size, np.sum(full_inf_mask)))

        return samples

    def volume(self) -> float:
        if np.any(np.isinf(self.lows)) or np.any(np.isinf(self.highs)):
            return np.inf
        return float(np.prod(self.highs - self.lows))

    def loss(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        if x.shape[1] != self.dim:
            return np.full((x.shape[0],), np.inf)

        below = np.maximum(self.lows - x, 0)
        above = np.maximum(x - self.highs, 0)
        total_loss = np.sum(below + above, axis=1)

        return total_loss[0] if was_unbatched else total_loss

    def centroid(self) -> np.ndarray:
        cents = (self.lows + self.highs) / 2.0
        # Fix infinite centers
        both_inf = np.isinf(self.lows) & np.isinf(self.highs)
        cents[both_inf] = 0.0
        mask_low = np.isinf(self.lows) & ~np.isinf(self.highs)
        mask_high = ~np.isinf(self.lows) & np.isinf(self.highs)
        cents[mask_low] = self.highs[mask_low] - 1.0
        cents[mask_high] = self.lows[mask_high] + 1.0
        return cents


class SphereSubspace(Subspace):
    """A hyper-spherical subspace."""

    def __init__(self, center: tuple, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        super().__init__(dim=len(center))

    def contains(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        if x.shape[1] != self.dim:
            return np.zeros(x.shape[0], dtype=bool) if not was_unbatched else False
        diff = x - self.center
        result = np.linalg.norm(diff, axis=1) <= self.radius
        return result[0] if was_unbatched else result

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._get_rng(rng)
        batch, d = shape
        direction = rng.standard_normal((batch, d))
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        unit = direction / (norms + 1e-8)
        u = rng.random((batch, 1))
        r = np.power(u, 1.0 / d) * self.radius
        return self.center + unit * r

    def volume(self) -> float:
        d = self.dim
        if d == 0: return 1.0
        return (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * (self.radius ** d)

    def loss(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        dists = np.linalg.norm(x - self.center, axis=1)
        result = np.maximum(dists - self.radius, 0)
        return result[0] if was_unbatched else result

    def centroid(self) -> np.ndarray:
        return self.center.copy()


class UnionSubspace(Subspace):
    """Represents the union of multiple subspaces."""

    def __init__(self, *subspaces: Subspace, weighted: bool = False):
        """Build the union of one or more subspaces.

        Args:
            subspaces: The component subspaces.
            weighted: If True, sample each component in proportion to its volume.

        Raises:
            ValueError: If no subspaces are given.
        """
        self.subspaces = list(subspaces)
        if not self.subspaces:
            raise ValueError("UnionSubspace requires at least one subspace.")
        super().__init__(dim=self.subspaces[0].dim)
        self.weighted = weighted

    def contains(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        res = np.zeros(x.shape[0], dtype=bool)
        for s in self.subspaces:
            res = res | s.contains(x)
        return res[0] if was_unbatched else res

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._get_rng(rng)
        batch, d = shape
        vols = [s.volume() for s in self.subspaces]

        # Determine sampling probabilities based on volume
        if any(math.isinf(v) for v in vols) or not self.weighted:
            probs = np.ones(len(self.subspaces)) / len(self.subspaces)
        else:
            total = sum(vols)
            probs = np.array([v / total if total > 0 else 1.0 / len(self.subspaces) for v in vols])

        indices = rng.choice(len(self.subspaces), size=batch, p=probs)
        samples = np.empty(shape, dtype=float)

        for i, s in enumerate(self.subspaces):
            mask = (indices == i)
            num_samples = np.sum(mask)
            if num_samples > 0:
                samples[mask] = s.sample((num_samples, d), rng=rng)
        return samples

    def volume(self) -> float:
        total = sum(s.volume() for s in self.subspaces)
        # Sums component volumes without accounting for overlaps, so the result
        # may overestimate; treat PrecisionMetric on a UnionSubspace with care.
        return total

    def loss(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        losses = np.stack([s.loss(x) for s in self.subspaces], axis=1)
        result = np.min(losses, axis=1)
        return result[0] if was_unbatched else result

    def centroid(self) -> np.ndarray:
        total_vol = self.volume()
        if not self.weighted or math.isinf(total_vol) or total_vol == 0:
            return self.subspaces[0].centroid()
        weighted_centroids = [s.centroid() * (s.volume() / total_vol) for s in self.subspaces]
        return np.sum(np.stack(weighted_centroids), axis=0)


class ComplementSubspace(Subspace):
    """Represents the complement of a set of subspaces."""

    def __init__(self, subspaces_to_exclude: List[Subspace], dim: int):
        self.subspaces_to_exclude = subspaces_to_exclude
        super().__init__(dim=dim)

    def contains(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        in_any_excluded = np.zeros(x.shape[0], dtype=bool)
        for s in self.subspaces_to_exclude:
            in_any_excluded = in_any_excluded | s.contains(x)
        result = ~in_any_excluded
        return result[0] if was_unbatched else result

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._get_rng(rng)
        # Rejection sampling (inefficient but generic)
        batch_size, d = shape
        accepted_samples = np.empty((0, d), dtype=float)
        max_tries = 1000
        for _ in range(max_tries):
            candidates = rng.standard_normal((batch_size * 5, d)) * 10
            is_in_complement = self.contains(candidates)
            accepted_samples = np.concatenate([accepted_samples, candidates[is_in_complement]], axis=0)
            if accepted_samples.shape[0] >= batch_size:
                return accepted_samples[:batch_size]

        # Rejection sampling could not fill the batch within max_tries: the
        # complement is empty or negligibly small under the sampling envelope.
        # Warn before returning the NaN-padded array, otherwise the NaNs
        # propagate silently into interventions and metric values.
        warnings.warn(
            f"ComplementSubspace.sample: collected only "
            f"{accepted_samples.shape[0]}/{batch_size} valid sample(s) after "
            f"{max_tries} rejection-sampling rounds; remaining rows are NaN. "
            f"The complement region is likely empty or negligibly small under "
            f"the sampling envelope.",
            RuntimeWarning,
            stacklevel=2,
        )
        final_samples = np.full(shape, np.nan)
        if accepted_samples.shape[0] > 0:
            final_samples[:accepted_samples.shape[0]] = accepted_samples
        return final_samples

    def volume(self) -> float:
        return np.inf

    def loss(self, x: np.ndarray) -> np.ndarray:
        return (~self.contains(x)).astype(float)

    def centroid(self) -> np.ndarray:
        """A deterministic representative point of the complement.

        The complement of a bounded set of excluded regions is unbounded, so it
        has no finite geometric center; we return the origin when it lies in the
        complement, otherwise the first point found by a deterministic outward
        probe along each axis. Returns a NaN-filled point if every probe is
        excluded (the complement is empty or negligibly small).
        """
        origin = np.zeros(self.dim)
        if bool(np.asarray(self.contains(origin)).ravel()[0]):
            return origin
        for radius in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            for axis in range(self.dim):
                for sign in (1.0, -1.0):
                    probe = np.zeros(self.dim)
                    probe[axis] = sign * radius
                    if bool(np.asarray(self.contains(probe)).ravel()[0]):
                        return probe
        return np.full(self.dim, np.nan)


class FullSubspace(Subspace):
    """Represents the entire R^dim space."""

    def __init__(self, dim: int):
        super().__init__(dim)

    def contains(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        result = np.ones(x.shape[0], dtype=bool)
        return result[0] if was_unbatched else result

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        # Defaults to standard normal distribution
        rng = self._get_rng(rng)
        return rng.standard_normal(shape)

    def volume(self) -> float:
        return np.inf

    def loss(self, x: np.ndarray) -> np.ndarray:
        x, was_unbatched = self._ensure_batched(x)
        result = np.zeros(x.shape[0], dtype=float)
        return result[0] if was_unbatched else result

    def centroid(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=float)


class UniformSubspace(RectSubspace):
    """Convenience class for a 1D uniform interval."""
    def __init__(self, low: float, high: float):
        super().__init__((low, high))


class GaussianSubspace(RectSubspace):
    """
    Approximates a Gaussian distribution using a RectSubspace.
    Bounds are defined by n_std deviations.
    Sampling uses actual Gaussian distribution.
    """
    def __init__(self, mean: float, std: float, n_std: float = 3.0):
        self.mean = mean
        self.std = std
        self.n_std = n_std
        low = self.mean - self.n_std * self.std
        high = self.mean + self.n_std * self.std
        super().__init__((low, high))

    def sample(self, shape: tuple, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._get_rng(rng)
        return rng.normal(self.mean, self.std, size=shape)

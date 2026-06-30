"""
Atomic primitives for the causal abstraction framework, unifying
representations of system states, abstract variables, and probability distributions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypeVar, Union, TypedDict

import numpy as np

# A label for a discrete state (e.g., 'red', 0, 'on')
StateLabel = TypeVar("StateLabel")


class _UnmappedSentinel(str):
    """Singleton sentinel used to represent a value that could not be mapped."""
    _instance = None

    def __new__(cls, copy=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls, "\x00__UNMAPPED__\x00")  # unprintable prefix
        return cls._instance

    def __eq__(self, other):
        return other is self  # identity check, not value check

    def __hash__(self):
        return id(self)

UNMAPPED = _UnmappedSentinel()


# Distribution primitives

class ProbabilityDistribution(ABC):
    """Abstract base class for representing a distribution of values/states."""

    @abstractmethod
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n samples from the distribution."""
        pass

    @abstractmethod
    def mode(self) -> Any:
        """Return the mode (most likely value) or central tendency."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Return a representative array (mode/mean) for point-wise comparison."""
        pass


@dataclass
class DiscreteDistribution(ProbabilityDistribution):
    """
    Explicit probability table (e.g., high-level model predictions).

    Attributes:
        probs: Dictionary mapping state labels to probabilities.
    """
    probs: Dict[StateLabel, float]

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n`` labels according to the probability table.

        Probabilities that sum to a value other than 1 (within tolerance) are
        renormalized with a warning.

        Raises:
            ValueError: If the probabilities sum to 0.
        """
        rng = rng if rng is not None else np.random.default_rng()

        labels = list(self.probs.keys())
        p = np.array(list(self.probs.values()), dtype=float)

        # Validation: Ensure sum is approximately 1.0
        total_p = np.sum(p)
        if not np.isclose(total_p, 1.0, atol=1e-5):
            if total_p == 0:
                raise ValueError("Probabilities sum to 0.")
            import warnings
            warnings.warn(
                f"DiscreteDistribution probabilities sum to {total_p:.6f}, normalizing.",
                UserWarning, stacklevel=2
            )
            p = p / total_p

        out = rng.choice(labels, size=n, p=p)
        # Preserve numeric dtype when labels are numeric to avoid object arrays downstream.
        if all(isinstance(l, (int, float, np.integer, np.floating)) for l in labels):
            out = out.astype(float)
        return out

    def mode(self) -> Any:
        return max(self.probs, key=self.probs.get)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "discrete", "probs": self.probs}

    def as_array(self) -> np.ndarray:
        return np.array(list(self.probs.values()))


@dataclass
class EmpiricalDistribution(ProbabilityDistribution):
    """
    A collection of samples from a simulator (e.g., low-level model outcomes).

    Attributes:
        samples: Array of shape (N_samples, *State_Dim).
    """
    samples: np.ndarray

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        # Bootstrap resample with replacement
        indices = rng.choice(len(self.samples), size=n)
        return self.samples[indices]

    def mode(self) -> Any:
        # Use mean as a proxy for mode/centroid for continuous data
        return np.mean(self.samples, axis=0)

    def mean(self) -> np.ndarray:
        """Return the sample mean over axis 0."""
        return np.mean(self.samples, axis=0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "empirical", "samples": self.samples.tolist()}

    def as_array(self) -> np.ndarray:
        return self.mean()


# Core primitives

@dataclass
class AbstractVariable:
    """
    Represents a variable in the high-level causal model.

    Attributes:
        name: The identifier of the variable.
        distribution: Optional discrete probability distribution.
        domain: Optional Subspace object defining the variable's domain.
    """
    name: str
    distribution: Optional[Dict[StateLabel, float]] = field(default_factory=dict)
    domain: Any = None  # Expected type: Subspace

    def __post_init__(self):
        # Normalize distribution if provided
        if self.distribution:
            total = sum(self.distribution.values())
            if total > 0 and abs(total - 1.0) > 1e-6:
                self.distribution = {k: v / total for k, v in self.distribution.items()}


@dataclass
class SystemState:
    """
    Unified container for system state.

    Values can be:
    1. np.ndarray (point estimate / single run)
    2. ProbabilityDistribution (stochastic outcome)
    3. UNMAPPED sentinel
    """
    values: Dict[str, Union[np.ndarray, ProbabilityDistribution, Any]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any):
        self.values[key] = value

    def merge(self, other: 'SystemState') -> 'SystemState':
        """Returns a new SystemState merging this state with another."""
        new_values = self.values.copy()
        new_values.update(other.values)
        return SystemState(values=new_values)

    def to_dict(self) -> Dict[str, Any]:
        """Serialization helper."""
        serialized = {}
        for k, v in self.values.items():
            if isinstance(v, ProbabilityDistribution):
                serialized[k] = v.to_dict()
            elif isinstance(v, np.ndarray):
                serialized[k] = v.tolist()
            elif v is UNMAPPED:
                serialized[k] = "UNMAPPED"
            else:
                serialized[k] = v
        return serialized


def ensure_batch_2d(val):
    """
    Normalize an array-like value to shape (batch, *features).

    Rules:
      scalar / 0-d         -> (1, 1)
      1-d (n,)             -> (n, 1)       # n is batch, single feature
      2-d (batch, feat)    -> unchanged
      3-d+ (batch, *, *)   -> unchanged

    Canonical shape contract for micro-variable values; call at boundaries
    (model output, metric input) to avoid silent shape mismatches.
    """
    if not isinstance(val, np.ndarray):
        val = np.asarray(val, dtype=float)
    if val.ndim == 0:
        return val.reshape(1, 1)
    if val.ndim == 1:
        return val.reshape(-1, 1)
    return val


def infer_batch_size(spec: dict, default: int = 1) -> int:
    """
    Infer the batch size from an intervention spec dictionary, returning
    ``default`` if no batch dimension can be detected.

    This is the single source of truth for batch-size inference in the engine's
    path steps. Low-level model implementations receive raw interventions and may
    use their own logic.
    """
    if not spec:
        return default

    for key, val in spec.items():
        # Skip sentinel key
        if isinstance(key, str) and key.startswith('_'):
            continue

        # Dict-style spec: {'labels': [...], 'micro_values': array}
        if isinstance(val, dict):
            labels = val.get('labels')
            if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
                return len(labels)
            mv = val.get('micro_values')
            if isinstance(mv, np.ndarray) and mv.ndim >= 2:
                return mv.shape[0]

        # Raw ndarray (from grounded interventions)
        elif isinstance(val, np.ndarray):
            if val.ndim >= 2:
                return val.shape[0]

        # Plain list: batch of scalar labels (from _step_high_level_model_predict output)
        # or list of (index, array) tuples (partial selector writes)
        elif isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], tuple):
                _, arr = val[0]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    return arr.shape[0]
            else:
                return len(val)

    return default


class InterventionEntry(TypedDict, total=False):
    """
    A single variable's intervention specification, as produced by samplers.

    Attributes:
        labels (list): Abstract labels, one per batch item (length = batch_size).
            May contain None for variables where only micro_values are set
            (e.g. bottom-up samplers that skip abstraction).
        micro_values (np.ndarray | None): Grounded micro-level values, shape
            (batch_size, *feature_dims). None defers grounding to the path step
            (top-down flow).
    """
    labels: list
    micro_values: Optional[np.ndarray]


# The full spec is a dict from abstract variable names to InterventionEntry.
# Sentinel keys (e.g. '_phi', '_pair_a') use str keys but non-standard values.
InterventionSpec = Dict[str, InterventionEntry]

# After grounding, interventions become micro-level:
#   { micro_var_name: ndarray | list[(index, ndarray)] }
GroundedInterventions = Dict[str, Any]
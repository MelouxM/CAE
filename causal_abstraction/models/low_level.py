"""Low-level model interface: how models expose their state and handle interventions."""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import numpy as np

from ..primitives import SystemState


class LowLevelModel(ABC):
    """
    Abstract base class for all low-level models.

    A low-level model represents a complex system (e.g., a neural network, a physics simulation)
    on which we can perform interventions on internal components.
    """

    def forward(self, input_state: SystemState) -> SystemState:
        """
        Performs a standard forward pass without interventions.

        Args:
            input_state: The initial state of the system (e.g., input values).

        Returns:
            The final state of the system (including outputs and internals).
        """
        return self.forward_with_interventions(input_state, {})

    @abstractmethod
    def forward_with_interventions(
            self,
            input_state: SystemState,
            interventions: Dict[str, np.ndarray]
    ) -> SystemState:
        """
        Performs a forward pass with interventions applied to micro-variables.

        Stochastic models can also take an optional `rng` keyword
        (`np.random.Generator`); the engine then passes its seeded generator
        (see `run_low_level_model`).

        Args:
            input_state: The initial state of the system.
            interventions: A dictionary mapping micro-variable names to replacement values (numpy arrays).

        Returns:
            The final state of the system.
        """
        pass

    def save_state(self) -> Any:
        """Capture a snapshot of the model's current internal state.

        Returns:
            An opaque snapshot, or None for stateless models (the default).
        """
        return None

    def load_state(self, state_snapshot: Any):
        """Restore the model from a snapshot produced by `save_state`.

        Args:
            state_snapshot: A previously saved snapshot.
        """
        pass


def _accepts_rng(model: LowLevelModel) -> bool:
    """Return True if `model.forward_with_interventions` declares an `rng` parameter."""
    try:
        params = inspect.signature(model.forward_with_interventions).parameters
    except (TypeError, ValueError):
        return False
    return 'rng' in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def run_low_level_model(model: LowLevelModel,
                        input_state: SystemState,
                        interventions: Dict[str, Any],
                        rng: Optional[np.random.Generator] = None) -> SystemState:
    """Call `model.forward_with_interventions`, passing `rng` if the model accepts it.

    Args:
        model: The low-level model M.
        input_state: The initial state of the system.
        interventions: Micro-level interventions.
        rng: Seeded generator for the model's own stochastic draws (ignored by
            models that do not declare an `rng` parameter, and when None).

    Returns:
        The final state returned by the model.
    """
    if rng is not None and _accepts_rng(model):
        return model.forward_with_interventions(input_state, interventions, rng=rng)
    return model.forward_with_interventions(input_state, interventions)


class NoisyLowLevelModel(LowLevelModel):
    """
    Wraps any LowLevelModel to add measurement noise to the array-valued variables
    in its output state.

    Args:
        base_model: The underlying model M to wrap.
        noise_std: Standard deviation of the noise (semantics depend on noise_type).
        noise_vars: Names of the micro-variables to corrupt (all array-valued variables if None).
        noise_type: One of `'gaussian'` (additive N(0, noise_std²)), `'relative'`
            (additive proportional N(0, (noise_std · |x|)²)), or `'poisson'`
            (replace x with a Poisson(max(x, 0)) sample).
        rng: Fallback random number generator, used when no generator is passed
            to `forward_with_interventions`.

    Raises:
        ValueError: If `noise_type` is not one of the three supported values.
    """

    def __init__(
        self,
        base_model: LowLevelModel,
        noise_std: float,
        noise_vars: Optional[List[str]] = None,
        noise_type: str = 'gaussian',
        rng: Optional[np.random.Generator] = None,
    ):
        if noise_type not in ('gaussian', 'relative', 'poisson'):
            raise ValueError(f"noise_type must be 'gaussian', 'relative', or 'poisson', got '{noise_type}'")
        self.base_model = base_model
        self.noise_std = noise_std
        self.noise_vars = noise_vars
        self.noise_type = noise_type
        self._rng = rng if rng is not None else np.random.default_rng()

    def forward_with_interventions(
            self,
            input_state: SystemState,
            interventions: Dict[str, np.ndarray],
            rng: Optional[np.random.Generator] = None,
    ) -> SystemState:
        result = run_low_level_model(self.base_model, input_state, interventions, rng=rng)
        gen = rng if rng is not None else self._rng

        new_values = dict(result.values)
        for var, val in new_values.items():
            if self.noise_vars is not None and var not in self.noise_vars:
                continue
            if not isinstance(val, np.ndarray):
                continue
            val = val.astype(float)
            if self.noise_type == 'gaussian':
                new_values[var] = val + gen.normal(0.0, self.noise_std, val.shape)
            elif self.noise_type == 'relative':
                std = self.noise_std * np.abs(val)
                new_values[var] = val + gen.normal(0.0, std + 1e-12)
            elif self.noise_type == 'poisson':
                new_values[var] = gen.poisson(np.maximum(val, 0.0)).astype(float)

        return SystemState(values=new_values)

    def save_state(self) -> Any:
        return self.base_model.save_state()

    def load_state(self, state_snapshot: Any):
        self.base_model.load_state(state_snapshot)
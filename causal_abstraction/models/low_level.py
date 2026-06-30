"""Low-level model interface: how models expose their state and handle interventions."""

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
        """Restore the model from a snapshot produced by :meth:`save_state`.

        Args:
            state_snapshot: A previously saved snapshot.
        """
        pass


class NoisyLowLevelModel(LowLevelModel):
    """
    Wraps any LowLevelModel to add measurement noise to the array-valued variables
    in its output state.

    Args:
        base_model: The underlying model M to wrap.
        noise_std: Standard deviation of the noise (semantics depend on noise_type).
        noise_vars: Names of the micro-variables to corrupt (all array-valued variables if None).
        noise_type: One of ``'gaussian'`` (additive N(0, noise_std²)), ``'relative'``
            (additive proportional N(0, (noise_std · |x|)²)), or ``'poisson'``
            (replace x with a Poisson(max(x, 0)) sample).
        rng: Random number generator.

    Raises:
        ValueError: If ``noise_type`` is not one of the three supported values.
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
    ) -> SystemState:
        result = self.base_model.forward_with_interventions(input_state, interventions)

        new_values = dict(result.values)
        for var, val in new_values.items():
            if self.noise_vars is not None and var not in self.noise_vars:
                continue
            if not isinstance(val, np.ndarray):
                continue
            val = val.astype(float)
            if self.noise_type == 'gaussian':
                new_values[var] = val + self._rng.normal(0.0, self.noise_std, val.shape)
            elif self.noise_type == 'relative':
                std = self.noise_std * np.abs(val)
                new_values[var] = val + self._rng.normal(0.0, std + 1e-12)
            elif self.noise_type == 'poisson':
                new_values[var] = self._rng.poisson(np.maximum(val, 0.0)).astype(float)

        return SystemState(values=new_values)

    def save_state(self) -> Any:
        return self.base_model.save_state()

    def load_state(self, state_snapshot: Any):
        self.base_model.load_state(state_snapshot)
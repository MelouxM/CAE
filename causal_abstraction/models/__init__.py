"""
Structural causal models used by the framework.

Bundles the two SCMs compared on the commuting diagram: the low-level model
``M`` (:class:`LowLevelModel`, :class:`~causal_abstraction.models.low_level.NoisyLowLevelModel`)
and the high-level model ``E`` (:class:`CausalGraph`). Also provides the PyTorch
wrapper :class:`NeuralModel`, imported lazily (see ``__getattr__``) so that
``import causal_abstraction.models`` does not require ``torch``.
"""
from .low_level import LowLevelModel
from .high_level import CausalGraph

__all__ = [
    "LowLevelModel",
    "CausalGraph",
    "NeuralModel"
]


def __getattr__(name):
    # PEP 562 lazy import: defer the torch dependency until NeuralModel is used.
    if name == "NeuralModel":
        from .neural import NeuralModel
        return NeuralModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
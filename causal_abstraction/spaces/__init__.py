"""
Subspace geometry for variable domains.

Defines the regions that interventions are sampled from and checked against
(:class:`RectSubspace`, :class:`SphereSubspace`, the set-algebra combinators
:class:`UnionSubspace` / :class:`ComplementSubspace` / :class:`FullSubspace`, and
the sampling distributions :class:`UniformSubspace` / :class:`GaussianSubspace`),
all built on the :class:`Subspace` base class.
"""
from .base import (
    Subspace,
    RectSubspace,
    SphereSubspace,
    UnionSubspace,
    ComplementSubspace,
    FullSubspace,
    UniformSubspace,
    GaussianSubspace,
)

__all__ = [
    "Subspace",
    "RectSubspace",
    "SphereSubspace",
    "UnionSubspace",
    "ComplementSubspace",
    "FullSubspace",
    "UniformSubspace",
    "GaussianSubspace",
]
"""
Subspace geometry for variable domains.

Defines the regions that interventions are sampled from and checked against
(`RectSubspace`, `SphereSubspace`, the set-algebra combinators
`UnionSubspace` / `ComplementSubspace` / `FullSubspace`, and
the sampling distributions `UniformSubspace` / `GaussianSubspace`),
all built on the `Subspace` base class.
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
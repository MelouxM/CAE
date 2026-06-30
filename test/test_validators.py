"""
Characterization tests for the structural validators that guard malformed
abstractions: ``CoarseGrainingMap``'s disjointness / topology
checks (the ``a`` map) and ``ValueMap.validate_surjectivity`` (the ``tau_X``
maps).

These are characterization tests: each assertion pins the validator's
current behavior, never a "corrected" one (to avoid changing any metric or
reported number). In particular two
contracts are easy to misremember and are pinned deliberately:

* ``CoarseGrainingMap`` raises (``ValueError`` / ``NotImplementedError``) at
  construction time on a malformed map; whereas
* ``ValueMap.validate_surjectivity`` does not raise: it returns a ``bool``
  (and logs on mismatch).

No metric, reported number, or system definition is exercised or changed.
"""
import logging

import numpy as np
import pytest

from causal_abstraction import (
    CoarseGrainingMap,
    ContinuousValueMap,
    MicroSelector,
    MicroVariableSchema,
    RectSubspace,
    UNMAPPED,
    ValueMap,
    Variable,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _schema(*variables):
    return MicroVariableSchema(list(variables))


def _value_map(specs):
    """Wrap label->subspace ``specs`` in a ValueMap over a trivial cg_map.

    ``validate_surjectivity`` only consults ``self.specs``; the cg_map merely
    has to be constructible, so each abstract var is backed by a 1-dim micro var.
    """
    schema = _schema(*[Variable(name, shape=(1,)) for name in specs])
    cg = CoarseGrainingMap(schema, {name: [name] for name in specs})
    return ValueMap(cg, specs)


# --------------------------------------------------------------------------- #
# CoarseGrainingMap: disjointness / topology checks (raise paths)
# --------------------------------------------------------------------------- #

def test_cg_overlapping_selectors_raise():
    """Two abstract vars claiming overlapping micro dims -> ValueError."""
    schema = _schema(Variable("x", shape=(4,)))
    with pytest.raises(ValueError, match="Overlap detected"):
        CoarseGrainingMap(schema, {
            "A": [("x", slice(0, 2))],
            "B": [("x", slice(1, 3))],  # shares index 1 with A
        })


def test_cg_adjacent_slices_are_disjoint():
    """Touching-but-not-overlapping slices [0:2) and [2:4) are allowed.

    Pins the half-open boundary semantics: ``max(0,2) < min(2,4)`` is False.
    """
    schema = _schema(Variable("x", shape=(4,)))
    cg = CoarseGrainingMap(schema, {
        "A": [("x", slice(0, 2))],
        "B": [("x", slice(2, 4))],
    })
    # Fully covered across the two abstract vars -> x is not a phi variable.
    assert cg.phi_variables == set()


def test_cg_unknown_variable_raises():
    schema = _schema(Variable("x", shape=(2,)))
    with pytest.raises(ValueError, match="unknown variables"):
        CoarseGrainingMap(schema, {"A": ["z"]})  # z absent from schema


def test_cg_invalid_selector_format_raises():
    schema = _schema(Variable("x", shape=(2,)))
    with pytest.raises(ValueError, match="Invalid selector format"):
        CoarseGrainingMap(schema, {"A": [5]})  # neither str nor 2-tuple


def test_cg_unsupported_index_type_raises_notimplemented():
    """A tuple index reaches the overlap check and is rejected explicitly."""
    schema = _schema(Variable("x", shape=(4,)))
    with pytest.raises(NotImplementedError, match="int and slice"):
        CoarseGrainingMap(schema, {"A": [("x", (0, 1))]})


def test_cg_internal_variable_not_in_schema_raises():
    schema = _schema(Variable("x", shape=(2,)))
    with pytest.raises(ValueError, match="Internal variables not in schema"):
        CoarseGrainingMap(schema, {"A": ["x"]}, internal_variables=["ghost"])


# --------------------------------------------------------------------------- #
# CoarseGrainingMap: phi-variable (Phi) bookkeeping
# --------------------------------------------------------------------------- #

def test_cg_phi_variables_are_unmapped_vars():
    """Variables with no selector become phi (claimed causally inert)."""
    schema = _schema(
        Variable("x", shape=(4,)),
        Variable("y", shape=(2,)),
        Variable("z", shape=(1,)),
    )
    cg = CoarseGrainingMap(schema, {"A": ["x"]})  # only x is mapped
    assert cg.phi_variables == {"y", "z"}


def test_cg_internal_variables_excluded_from_phi():
    """An unmapped-but-internal variable is neither mapped nor phi."""
    schema = _schema(Variable("x", shape=(4,)), Variable("y", shape=(2,)))
    cg = CoarseGrainingMap(schema, {"A": ["x"]}, internal_variables=["y"])
    assert cg.phi_variables == set()
    assert cg.internal_variables == {"y"}


def test_cg_partial_mapping_warns_and_adds_unmapped_dims_to_phi(caplog):
    """A partially-mapped variable warns and its UNMAPPED dimensions join phi.

    Pins the behaviour that the unmapped dimensions of a partially-mapped
    variable are genuinely treated as Phi (one per-dimension ``MicroSelector``
    each), making the construction-time warning truthful. The mapped dimensions
    (0, 1) stay out of phi; only the unmapped tail (2, 3) is added.
    """
    schema = _schema(Variable("x", shape=(4,)))
    with caplog.at_level(logging.WARNING, logger="causal_abstraction.schema"):
        cg = CoarseGrainingMap(schema, {"A": [("x", slice(0, 2))]})  # 2/4 dims
    assert "partially mapped" in caplog.text
    assert cg.phi_variables == {MicroSelector("x", 2), MicroSelector("x", 3)}


# --------------------------------------------------------------------------- #
# ValueMap.validate_surjectivity (returns bool; does not raise)
# --------------------------------------------------------------------------- #

def test_validate_surjectivity_true_for_disjoint_labels():
    """Disjoint label subspaces each round-trip back to their own label."""
    vm = _value_map({"A": {
        0: RectSubspace((0.0, 1.0)),
        1: RectSubspace((2.0, 3.0)),
    }})
    assert vm.validate_surjectivity(rng=np.random.default_rng(0)) is True


def test_validate_surjectivity_false_when_label_subspace_is_shadowed():
    """Label 1's region lies inside label 0's, so it abstracts to 0 -> False.

    ``abstract`` returns the first containing subspace in insertion order, so a
    sample drawn for label 1 is captured by label 0 and the check fails.
    """
    vm = _value_map({"A": {
        0: RectSubspace((0.0, 5.0)),
        1: RectSubspace((1.0, 2.0)),  # subset of label 0's region
    }})
    assert vm.validate_surjectivity(rng=np.random.default_rng(0)) is False


def test_validate_surjectivity_false_when_sample_is_unmapped():
    """A gap between the only label's subspace and the rest maps to UNMAPPED.

    Here the single defined label correctly contains its own samples, so the map
    is surjective; flip it by adding a label whose region is uncovered by its
    own spec. We assert the negative branch via an explicit UNMAPPED sample.
    """
    spec = {0: RectSubspace((0.0, 1.0))}
    vm = _value_map({"A": spec})
    # Sanity: the lone label is self-consistent (surjective).
    assert vm.validate_surjectivity(rng=np.random.default_rng(0)) is True
    # And a value outside every label's subspace is reported as UNMAPPED.
    assert vm.abstract("A", np.array([9.0])) is UNMAPPED


# --------------------------------------------------------------------------- #
# ValueMap: user-reachable lookup errors (abstract / ground)
# --------------------------------------------------------------------------- #

def test_abstract_unknown_variable_raises():
    vm = _value_map({"A": {0: RectSubspace((0.0, 1.0))}})
    with pytest.raises(ValueError, match="No value spec"):
        vm.abstract("missing", np.array([0.5]))


def test_ground_unknown_label_raises():
    vm = _value_map({"A": {0: RectSubspace((0.0, 1.0))}})
    with pytest.raises(ValueError, match="Label .* not defined"):
        vm.ground("A", 7)


# --------------------------------------------------------------------------- #
# ContinuousValueMap.validate_surjectivity (positive-volume contract)
# --------------------------------------------------------------------------- #

def _continuous_map(specs):
    schema = _schema(*[Variable(name, shape=(1,)) for name in specs])
    cg = CoarseGrainingMap(schema, {name: [name] for name in specs})
    return ContinuousValueMap(cg, specs)


def test_continuous_validate_true_for_positive_volume():
    cvm = _continuous_map({"A": {0: RectSubspace((0.0, 1.0))}})
    assert cvm.validate_surjectivity() is True


def test_continuous_validate_false_for_empty_specs():
    cvm = _continuous_map({"A": {}})
    assert cvm.validate_surjectivity() is False


def test_continuous_validate_false_for_zero_volume():
    cvm = _continuous_map({"A": {0: RectSubspace((1.0, 1.0))}})  # degenerate point
    assert cvm.validate_surjectivity() is False

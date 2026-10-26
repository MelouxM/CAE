"""
Unit tests for the intervention samplers.

Covers seeded reproducibility (same seed -> identical interventions), the
target-selection contract (respecting `max_interventions` / `force_all`),
and the faithfulness sampler's phi-sentinel behavior, including the configurable
`phi_prob` inclusion probability wired through
`EvaluationConfig.phi_selection_prob`.
"""
import copy
from types import SimpleNamespace

import numpy as np

from causal_abstraction import (
    BottomUpSampler,
    CombinedFaithfulnessSampler,
    PHI_DUMMY_NAME,
    TopDownSampler,
)


def _variables(logic_system):
    return list(logic_system["valid"].variables.values())


def _canonical(spec):
    """Stable, comparable representation of an intervention spec's labels."""
    out = {}
    for name, entry in spec.items():
        labels = entry.get("labels") if isinstance(entry, dict) else entry
        out[name] = np.asarray(labels).ravel().tolist()
    return out


def test_topdown_reproducible_under_seed(logic_system):
    td = TopDownSampler(logic_system["vm"])
    variables = _variables(logic_system)
    s1 = td.sample_intervention(variables, batch_size=4, max_interventions=2,
                                rng=np.random.default_rng(123))
    s2 = td.sample_intervention(variables, batch_size=4, max_interventions=2,
                                rng=np.random.default_rng(123))
    assert _canonical(s1) == _canonical(s2)


def test_bottomup_reproducible_under_seed(logic_system):
    bu = BottomUpSampler(logic_system["vm"])
    variables = _variables(logic_system)
    s1 = bu.sample_intervention(variables, batch_size=4, max_interventions=2,
                                rng=np.random.default_rng(7))
    s2 = bu.sample_intervention(variables, batch_size=4, max_interventions=2,
                                rng=np.random.default_rng(7))
    assert _canonical(s1) == _canonical(s2)
    for name in s1:
        np.testing.assert_array_equal(s1[name]["micro_values"], s2[name]["micro_values"])


def test_selection_varies_with_seed(logic_system):
    td = TopDownSampler(logic_system["vm"])
    variables = _variables(logic_system)
    seen = {tuple(sorted(td.sample_intervention(
        variables, batch_size=2, max_interventions=2, rng=np.random.default_rng(seed)).keys()))
        for seed in range(15)}
    assert len(seen) > 1


def test_select_targets_respects_max_interventions(logic_system):
    td = TopDownSampler(logic_system["vm"])
    variables = _variables(logic_system)
    rng = np.random.default_rng(0)
    for _ in range(30):
        targets = td._select_targets(variables, max_interventions=2, force_all=False, rng=rng)
        assert 1 <= len(targets) <= 2


def test_select_targets_force_all(logic_system):
    td = TopDownSampler(logic_system["vm"])
    variables = _variables(logic_system)
    targets = td._select_targets(variables, max_interventions=None, force_all=True,
                                 rng=np.random.default_rng(0))
    assert len(targets) == len(variables)


def test_faithfulness_force_all_bypasses_phi(logic_system):
    cfs = CombinedFaithfulnessSampler(TopDownSampler(logic_system["vm"]))
    spec = cfs.sample_intervention(_variables(logic_system), batch_size=2,
                                   force_all=True, rng=np.random.default_rng(0))
    assert PHI_DUMMY_NAME not in spec


def _vm_with_phi(logic_system):
    """The adder's value map, seen through a coarse-graining map with a non-empty Phi."""
    vm = copy.copy(logic_system["vm"])
    vm.cg_map = SimpleNamespace(phi_variables={"unmapped_wire"})
    return vm


def test_phi_selection_probability(logic_system):
    """The configurable phi_prob (EvaluationConfig.phi_selection_prob) gates phi."""
    variables = _variables(logic_system)
    base = TopDownSampler(_vm_with_phi(logic_system))
    always = CombinedFaithfulnessSampler(base, phi_prob=1.0)
    never = CombinedFaithfulnessSampler(base, phi_prob=0.0)
    for seed in range(10):
        s_always = always.sample_intervention(variables, batch_size=2, max_interventions=2,
                                              rng=np.random.default_rng(seed))
        s_never = never.sample_intervention(variables, batch_size=2, max_interventions=2,
                                            rng=np.random.default_rng(seed))
        assert PHI_DUMMY_NAME in s_always
        assert PHI_DUMMY_NAME not in s_never
        # real intervention targets coexist with the phi sentinel
        assert any(k != PHI_DUMMY_NAME for k in s_always)


def test_empty_phi_never_selects_phi(logic_system):
    """With Phi empty the faithfulness test is vacuous: no phi slot, same draws as the base."""
    assert not logic_system["cg"].phi_variables
    variables = _variables(logic_system)
    base = TopDownSampler(logic_system["vm"])
    for phi_prob in (None, 1.0):
        cfs = CombinedFaithfulnessSampler(base, phi_prob=phi_prob)
        for seed in range(10):
            spec = cfs.sample_intervention(variables, batch_size=2, max_interventions=2,
                                           rng=np.random.default_rng(seed))
            ref = base.sample_intervention(variables, batch_size=2, max_interventions=2,
                                           rng=np.random.default_rng(seed))
            assert PHI_DUMMY_NAME not in spec
            assert _canonical(spec) == _canonical(ref)

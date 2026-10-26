"""Tests for overlapping cells in the coarse-graining map.

Toy system: micro scalars a, b, c, d and an unmapped e; M sets d = max(a, b, c).
Macro X reads (a, b), Y reads (b, c) (they share b), Z reads d, and E is Z = X or Y.
"""
import numpy as np
import pytest

from causal_abstraction import (
    BottomUpSampler,
    CausalGraph,
    CoarseGrainingMap,
    EvaluationConfig,
    EvaluationEngine,
    LowLevelModel,
    MicroVariableSchema,
    RectSubspace,
    SystemState,
    TopDownSampler,
    ValueMap,
)
from causal_abstraction.paths import OverlapConflict
from causal_abstraction.tasks import StandardTasks

_ZERO = (-0.1, 0.1)
_ONE = (0.9, 1.1)


class _MaxModel(LowLevelModel):
    def forward_with_interventions(self, input_state, interventions):
        def get(name):
            val = interventions.get(name)
            return None if val is None else np.asarray(val, dtype=float).reshape(-1, 1)

        given = {k: get(k) for k in "abc"}
        batch = max((v.shape[0] for v in given.values() if v is not None), default=1)
        vals = {k: (v if v is not None else np.zeros((batch, 1))) for k, v in given.items()}
        d = np.maximum(np.maximum(vals["a"], vals["b"]), vals["c"])
        return SystemState(values={**vals, "d": d, "e": np.zeros((batch, 1))})


class _JointValueMap(ValueMap):
    def ground_joint(self, labels, rng=None):
        return {"a": [labels.get("X", 0)], "b": [0.0], "c": [labels.get("Y", 0)]}


def _system(value_map_cls=ValueMap):
    schema = MicroVariableSchema.from_names(["a", "b", "c", "d", "e"])
    cg = CoarseGrainingMap(schema, {"X": ["a", "b"], "Y": ["b", "c"], "Z": ["d"]})
    vm = value_map_cls(cg, {
        "X": {0: RectSubspace(_ZERO, _ZERO), 1: RectSubspace(_ONE, _ZERO)},
        "Y": {0: RectSubspace(_ZERO, _ZERO), 1: RectSubspace(_ZERO, _ONE)},
        "Z": {0: RectSubspace(_ZERO), 1: RectSubspace(_ONE)},
    })
    e = CausalGraph()
    e.add_variable("X", equation=lambda: 0, parents=[], distribution={0: 0.5, 1: 0.5})
    e.add_variable("Y", equation=lambda: 0, parents=[], distribution={0: 0.5, 1: 0.5})
    e.add_variable("Z", equation=lambda X, Y: int(X or Y), parents=["X", "Y"],
                   distribution={0: 0.5, 1: 0.5})
    engine = EvaluationEngine(e, _MaxModel(), vm, cg, EvaluationConfig(metric="hard", seed=0))
    return cg, vm, engine


def test_overlap_allowed_and_recorded():
    cg, _, _ = _system()
    assert cg.overlaps == {frozenset(("X", "Y")): {"b"}}
    assert cg.conflicts(["Y", "X", "Z"]) == [("X", "Y")]
    assert cg.conflicts(["X", "Z"]) == []
    assert cg.phi_variables == {"e"}


def test_overlap_rejected_when_disallowed():
    schema = MicroVariableSchema.from_names(["a", "b", "c"])
    with pytest.raises(ValueError, match="Overlap detected"):
        CoarseGrainingMap(schema, {"X": ["a", "b"], "Y": ["b", "c"]}, allow_overlap=False)


def test_ground_raises_without_joint_grounding():
    _, _, engine = _system()
    spec = {"X": {"labels": [1], "micro_values": None}, "Y": {"labels": [0], "micro_values": None}}
    with pytest.raises(OverlapConflict):
        engine.builder._step_ground_or_passthrough(spec, rng=np.random.default_rng(0))


def test_skipped_batches_are_reported():
    _, vm, engine = _system()
    res = engine.run_experiment(TopDownSampler(vm), 12, batch_size=1, max_interventions=2,
                                intervention_domain=["X", "Y"])
    # X and Y are both roots, so every batch intervenes on both
    assert res.metadata["overlap_skipped"] == {"n_batches": 12, "n_skipped": 12,
                                               "pairs": {"X/Y": 12}}
    assert np.isnan(res.score)


def test_joint_grounding_is_used():
    _, vm, engine = _system(_JointValueMap)
    res = engine.run_experiment(TopDownSampler(vm), 12, batch_size=2, max_interventions=2,
                                intervention_domain=["X", "Y"])
    assert "overlap_skipped" not in res.metadata
    assert res.score == 0.0


def test_bottom_up_overlap_is_skipped_in_tasks():
    # Bottom-up cells come with their own micro-values, so the joint grounding cannot apply
    _, vm, engine = _system(_JointValueMap)
    task = StandardTasks.score(engine.builder, sampler=BottomUpSampler(vm), name="CAE_up")
    res = engine.run_tasks([task], n_samples=8, batch_size=1, max_interventions=2,
                           intervention_domain=["X", "Y"])["CAE_up"]
    assert res.metadata["overlap_skipped"]["n_skipped"] == 8

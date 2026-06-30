"""
Characterization tests for the commuting-diagram machinery (``paths.py``).

``DiagramBuilder`` / ``CausalPath`` compose the abstraction (``tau``),
grounding (the inverse value map ``tau^{-1}``), the low-level model ``M``,
the high-level model ``E`` and Phi-noise steps that the Causal Abstraction
Error is built from. They were previously exercised only indirectly through one
engine test; this module drives the individual path steps and the assembled
paths directly on the canonical fast triple (the 2-bit adder ``logic_system``
fixture).

These are characterization tests: every assertion pins the
current observed behaviour (snapshot), never a "corrected" value, so they add
a regression net without changing any metric or reported number. Values that are
not deterministic across processes (subspace-sampled grounding; Phi-noise draw
order, which depends on set-iteration order of ``phi_variables``) are pinned only
by their guaranteed-stable properties (in-range membership, shape/dtype, and
same-seed reproducibility), while the genuinely reproducible cases (single Phi
variable, pure high-level prediction) are pinned exactly.
"""
import numpy as np
import pytest

from causal_abstraction import (
    EvaluationConfig,
    MicroVariableSchema,
    PHI_DUMMY_NAME,
    SystemState,
    UNMAPPED,
)
from causal_abstraction.paths import CausalPath, DiagramBuilder
from causal_abstraction.schema import CoarseGrainingMap, MicroSelector, Variable

# A fixed full-input intervention on the high-level roots and the abstraction it
# must produce on a valid abstraction (A=2, B=1, Cin=1 -> sum 4: S=0, Cout=1,
# internal carry C1=1). Hand-derived, then confirmed against the live code.
_ROOT_SPEC = {
    "Operand_A": {"labels": [2]},
    "Operand_B": {"labels": [1]},
    "Carry_In": {"labels": [1]},
}
_EXPECTED_LABELS = {
    "Operand_A": 2,
    "Operand_B": 1,
    "Carry_In": 1,
    "Internal_Carries": 1,
    "Result_Sum": 0,
    "Result_Carry": 1,
}


def _builder(logic_system, model=None, config=None):
    """Construct a ``DiagramBuilder`` exactly as ``EvaluationEngine`` does."""
    return DiagramBuilder(
        model or logic_system["valid"],
        logic_system["low_level"],
        logic_system["vm"],
        logic_system["cg"],
        config or EvaluationConfig(metric="hard"),
    )


def _phi_builder(logic_system, phi_specs):
    """A builder whose CoarseGrainingMap has the given ``(name, dtype)`` Phi vars.

    Only ``_step_inject_phi_noise`` is exercised on it; that step reads solely
    ``self.cg`` and ``self.config``, so reusing the fixture's models/value-map
    against an unrelated schema is sound and keeps the noise test self-contained.
    """
    schema = MicroVariableSchema(
        [Variable("m")] + [Variable(n, dtype=dt) for n, dt in phi_specs]
    )
    cg = CoarseGrainingMap(schema, {"X": ["m"]})
    return DiagramBuilder(
        logic_system["valid"], logic_system["low_level"], logic_system["vm"],
        cg, EvaluationConfig(metric="hard"),
    )


def _f(x):
    """Extract the single scalar from a 1-element array-like as a float."""
    return float(np.asarray(x).reshape(-1)[0])


def _ints(out):
    """Collapse a path/step output (``{var: [label, ...]}``, batch 1) to ints."""
    return {k: int(np.asarray(v[0]).ravel()[0]) for k, v in out.items()}


def _canon(out):
    """Like ``_ints`` but maps UNMAPPED/None to a sentinel for comparison."""
    def c(x):
        if x is UNMAPPED or x is None:
            return "EMPTY"
        return int(np.asarray(x).ravel()[0])
    return {k: [c(x) for x in v] for k, v in out.items()}


# --- CausalPath ------------------------------------------------------------

def test_causalpath_defaults():
    p = CausalPath("p", [])
    assert p.name == "p"
    assert p.steps == []
    assert p.exclude_intervened is True
    assert p.input_type is dict
    assert p.output_type is dict


def test_causalpath_execute_composes_steps_and_threads_rng():
    seen = []

    def s1(data, rng=None):
        seen.append(("s1", rng))
        return data + [1]

    def s2(data, rng=None):
        seen.append(("s2", rng))
        return data + [2]

    rng = np.random.default_rng(0)
    out = CausalPath("p", [s1, s2]).execute([0], rng=rng)
    assert out == [0, 1, 2]                      # steps applied left-to-right
    assert [name for name, _ in seen] == ["s1", "s2"]
    assert all(r is rng for _, r in seen)        # same rng handed to every step


# --- grounding step (the inverse value map tau^{-1}) -----------------------

def test_ground_or_passthrough_grounds_labels_in_range(logic_system):
    b = _builder(logic_system)
    g = b._step_ground_or_passthrough(
        {"Operand_A": {"labels": [2]}, "Carry_In": {"labels": [0]}}
    )
    assert set(g) == {"A0", "A1", "C0"}
    for k in g:
        assert np.asarray(g[k]).shape == (1, 1)
    # label 2 -> bit0 low, bit1 high; label 0 -> low. Subspaces: low (-0.1,0.1), high (0.9,1.1).
    assert -0.1 <= _f(g["A0"]) <= 0.1
    assert 0.9 <= _f(g["A1"]) <= 1.1
    assert -0.1 <= _f(g["C0"]) <= 0.1


def test_ground_or_passthrough_batched_labels(logic_system):
    b = _builder(logic_system)
    g = b._step_ground_or_passthrough({"Operand_A": {"labels": [0, 3]}})
    assert set(g) == {"A0", "A1"}
    for k in ("A0", "A1"):
        arr = np.asarray(g[k])
        assert arr.shape == (2, 1)
        assert -0.1 <= arr[0, 0] <= 0.1   # row 0: label 0 -> both bits low
        assert 0.9 <= arr[1, 0] <= 1.1    # row 1: label 3 -> both bits high


def test_ground_or_passthrough_micro_values_split(logic_system):
    b = _builder(logic_system)
    g = b._step_ground_or_passthrough(
        {"Operand_A": {"micro_values": np.array([[0.95, 0.05]])}}
    )
    # Two selectors, last dim == 2 -> split element-wise (deterministic passthrough).
    assert np.asarray(g["A0"]).ravel().tolist() == [0.95]
    assert np.asarray(g["A1"]).ravel().tolist() == [0.05]


def test_ground_or_passthrough_unknown_var_strict_vs_lenient(logic_system):
    strict = _builder(logic_system, config=EvaluationConfig(metric="hard", strict_mode=True))
    with pytest.raises(ValueError):
        strict._step_ground_or_passthrough({"NoSuchVar": {"labels": [0]}})
    # Non-strict: unmappable abstract var is skipped, yielding no interventions.
    lenient = _builder(logic_system)
    assert lenient._step_ground_or_passthrough({"NoSuchVar": {"labels": [0]}}) == {}


# --- low-level mechanism step ----------------------------------------------

def test_low_level_execute_runs_mechanism_and_normalizes(logic_system):
    b = _builder(logic_system)
    st = b._step_low_level_model_execute({
        "A0": np.array([[1.0]]), "A1": np.array([[1.0]]),
        "B0": np.array([[0.0]]), "B1": np.array([[0.0]]),
        "C0": np.array([[0.0]]),
    })
    assert isinstance(st, SystemState)
    # A=3, B=0, Cin=0 -> sum 3 (S0=1, S1=1), no carry out (C2=0), internal C1=0.
    for wire in ("S0", "S1", "C2", "C1"):
        assert np.asarray(st[wire]).shape == (1, 1)
    assert _f(st["S0"]) == 1.0
    assert _f(st["S1"]) == 1.0
    assert _f(st["C2"]) == 0.0
    assert _f(st["C1"]) == 0.0


# --- abstraction step ------------------------------------------------------

def test_abstract_maps_micro_state_to_labels(logic_system):
    b = _builder(logic_system)
    micro = {"A0": 1, "A1": 1, "B0": 0, "B1": 0, "C0": 0,
             "C1": 0, "S0": 1, "S1": 1, "C2": 0}
    state = SystemState(values={w: np.array([[float(x)]]) for w, x in micro.items()})
    assert _ints(b._step_abstract(state)) == {
        "Operand_A": 3, "Operand_B": 0, "Carry_In": 0,
        "Internal_Carries": 0, "Result_Sum": 3, "Result_Carry": 0,
    }


def test_abstract_missing_micro_is_unmapped_or_raises(logic_system):
    micro = {"A0": 1, "A1": 1, "B0": 0, "B1": 0, "C0": 0,
             "S0": 1, "S1": 1, "C2": 0}  # C1 (Internal_Carries) absent
    state = SystemState(values={w: np.array([[float(x)]]) for w, x in micro.items()})

    lenient = _builder(logic_system)
    assert lenient._step_abstract(state)["Internal_Carries"] is UNMAPPED

    strict = _builder(logic_system, config=EvaluationConfig(metric="hard", strict_mode=True))
    with pytest.raises(ValueError):
        strict._step_abstract(state)


# --- high-level mechanism step ---------------------------------------------

def test_high_level_predict_is_deterministic(logic_system):
    b = _builder(logic_system)
    assert _ints(b._step_high_level_model_predict(_ROOT_SPEC)) == _EXPECTED_LABELS


def test_high_level_predict_ignores_non_high_level_keys(logic_system):
    b = _builder(logic_system)
    spec = dict(_ROOT_SPEC, A0={"labels": [9]})  # micro key, not a high-level var
    assert _ints(b._step_high_level_model_predict(spec)) == _EXPECTED_LABELS


def test_high_level_predict_all_none_labels_skipped(logic_system):
    b = _builder(logic_system)
    omitted = b._step_high_level_model_predict(
        {"Operand_A": {"labels": [2]}, "Operand_B": {"labels": [1]}}
    )
    all_none = b._step_high_level_model_predict(
        {"Operand_A": {"labels": [2]}, "Operand_B": {"labels": [1]},
         "Carry_In": {"labels": [None]}}
    )
    # An all-None label set is treated as "not provided": identical predictions.
    assert _canon(all_none) == _canon(omitted)


# --- Phi-noise step --------------------------------------------------------

def test_phi_noise_is_noop_without_phi_variables(logic_system):
    b = _builder(logic_system)
    assert logic_system["cg"].phi_variables == set()
    ints = {"A0": np.array([[1.0]])}
    out = b._step_inject_phi_noise(ints, rng=np.random.default_rng(0))
    assert set(out) == {"A0"}
    np.testing.assert_array_equal(out["A0"], ints["A0"])


def test_phi_noise_is_reproducible_and_typed(logic_system):
    mb = _phi_builder(logic_system, [("phi_f", float), ("phi_i", int), ("phi_b", bool)])
    a = mb._step_inject_phi_noise({}, rng=np.random.default_rng(0), batch_size=4)
    b = mb._step_inject_phi_noise({}, rng=np.random.default_rng(0), batch_size=4)
    assert set(a) == {"phi_f", "phi_i", "phi_b"}
    for k in a:
        assert np.asarray(a[k]).shape == (4, 1)
        np.testing.assert_array_equal(a[k], b[k])  # same seed -> identical draws
    assert set(np.asarray(a["phi_i"]).ravel().tolist()).issubset({-1, 0, 1})
    assert set(np.asarray(a["phi_b"]).ravel().tolist()).issubset({True, False})
    assert np.all(np.isfinite(np.asarray(a["phi_f"])))


@pytest.mark.parametrize("dtype,expected", [
    (float, [0.00012301533574825743, 0.02987455375084699, -0.027413785536221758]),
    (int, [1, 0, 1]),
    (bool, [False, False, False]),
])
def test_phi_noise_exact_single_variable(logic_system, dtype, expected):
    # One Phi variable -> draw order is unambiguous, so exact values are stable.
    out = _phi_builder(logic_system, [("p", dtype)])._step_inject_phi_noise(
        {}, rng=np.random.default_rng(7), batch_size=3
    )["p"]
    assert np.asarray(out).shape == (3, 1)
    if dtype is float:
        np.testing.assert_allclose(np.asarray(out).ravel(), expected)
    else:
        assert np.asarray(out).ravel().tolist() == expected


def test_phi_noise_into_partially_mapped_variable_writes_unmapped_dims(logic_system):
    """Partial-mapping Phi: noise is written into only the UNMAPPED dimensions as
    ``(index, value)`` partial writes, leaving the grounded mapped write intact.

    Pins the consumer side of the partial-mapping fix: the unmapped dims (2, 3)
    of ``x`` become per-dimension Phi selectors, and ``_step_inject_phi_noise``
    layers a noise write for each on top of the existing mapped write for [0:2)."""
    schema = MicroVariableSchema([Variable("x", shape=(4,))])
    cg = CoarseGrainingMap(schema, {"A": [("x", slice(0, 2))]})  # 2/4 dims mapped
    assert cg.phi_variables == {MicroSelector("x", 2), MicroSelector("x", 3)}

    b = DiagramBuilder(
        logic_system["valid"], logic_system["low_level"], logic_system["vm"],
        cg, EvaluationConfig(metric="hard"),
    )
    mapped = (slice(0, 2), np.array([[1.0, 2.0]]))
    out = b._step_inject_phi_noise({"x": [mapped]}, rng=np.random.default_rng(0), batch_size=1)

    writes = out["x"]
    assert isinstance(writes, list)
    # The grounded mapped write is preserved unchanged.
    assert any(w[0] == slice(0, 2) for w in writes)
    # Each unmapped dim got one (int_index, (batch, 1)) noise write.
    int_writes = [w for w in writes if isinstance(w[0], int)]
    assert sorted(w[0] for w in int_writes) == [2, 3]
    for _, val in int_writes:
        assert np.asarray(val).shape == (1, 1)


# --- assembled path factories & full traversal -----------------------------

def test_path_factories_have_expected_shape(logic_system):
    b = _builder(logic_system)
    hl = b.build_path_standard_high_level_model()
    ll = b.build_path_standard_low_level_model()
    comb = b.build_path_combined_low_level_model()
    assert (hl.name, len(hl.steps)) == ("CAE_high_level_model", 1)
    assert (ll.name, len(ll.steps)) == ("CAE_low_level_model", 3)
    assert (comb.name, len(comb.steps)) == ("Combined_CAE_low_level_model", 3)
    assert hl.exclude_intervened and ll.exclude_intervened and comb.exclude_intervened


def test_full_paths_commute_on_valid_abstraction(logic_system):
    b = _builder(logic_system)
    hl = b.build_path_standard_high_level_model().execute(_ROOT_SPEC, rng=np.random.default_rng(0))
    ll = b.build_path_standard_low_level_model().execute(_ROOT_SPEC, rng=np.random.default_rng(0))
    # The defining property of a valid abstraction: the two paths agree,
    # p (E on abstracted inputs)  ==  q (tau o M o tau^{-1}).
    assert _ints(hl) == _EXPECTED_LABELS
    assert _ints(ll) == _EXPECTED_LABELS


def test_combined_path_matches_plain_when_phi_empty(logic_system):
    b = _builder(logic_system)
    plain = b.build_path_standard_low_level_model().execute(_ROOT_SPEC, rng=np.random.default_rng(0))
    comb = b.build_path_combined_low_level_model().execute(_ROOT_SPEC, rng=np.random.default_rng(0))
    assert _ints(comb) == _ints(plain) == _EXPECTED_LABELS
    # The PHI sentinel key is stripped before grounding and noise injection is a
    # no-op when there are no Phi variables, so adding it changes nothing.
    spec_phi = dict(_ROOT_SPEC)
    spec_phi[PHI_DUMMY_NAME] = {"labels": [None]}
    comb_phi = b.build_path_combined_low_level_model().execute(spec_phi, rng=np.random.default_rng(0))
    assert _ints(comb_phi) == _EXPECTED_LABELS

"""
Characterization tests for the analytical baseline family
(``AnalyticalMetric`` subclasses in ``causal_abstraction.analytical_metrics``).

This module is the safety net for the most logic-dense, previously untested
part of the library: the 30+ candidate baselines the paper benchmarks against
CAE. The tests must not change any metric or any reported
number; they instead pin (snapshot) the value each metric currently produces
on the small, deterministic 2-bit-adder abstraction. Every expected number in
``_EXPECTED`` was captured by running the metric on the fixture at a fixed
seed; none is a hand-chosen "correct" value. A future change to a metric
implementation therefore surfaces here as a failing characterization test
rather than as a silent drift in ``test/results/*.json``.

The battery mirrors the analytical metrics wired into the headline benchmark
suite ``test/01_logic_circuit.py`` and additionally covers the three metrics
the discrete suites omit:

* ``MallowsCpMetric``: runs directly on the adder.
* ``StructuralDeviationMetric`` / ``CausalSensitivityIndexMetric``: the
  parameter-perturbation metrics need a ``make_high_level_model`` factory; here
  it switches between the valid and failing high-level models on a synthetic
  parameter ``w``, which exercises their full inner-engine code path.

Each metric is pinned at two operating points: a valid abstraction (where
most baselines collapse to 0) and a failing one, so regressions that move a
value at either point are caught.
"""
import math

import pytest

from causal_abstraction import (
    BCCMetric,
    BottomUpSampler,
    CausalSensitivityIndexMetric,
    CIBLagrangianMetric,
    ComplexityShiftMetric,
    DCCMetric,
    EvaluationConfig,
    EvaluationEngine,
    IBLagrangianMetric,
    IIAMetric,
    InfidelityMetric,
    MacroscopicInvarianceMetric,
    MallowsCpMetric,
    ProbingMetric,
    RelationalFidelityMetric,
    SobolSensitivityMetric,
    StructuralDeviationMetric,
    SymbionMetric,
)

_N = 100
_SEED = 0
_OUTPUTS = ["Result_Sum", "Result_Carry"]

# Current (snapshot) value of each metric on the adder, captured at seed=0,
# n_samples=100. {metric_name: {high_level_model_key: value}}. See module docstring; these
# pin existing behaviour and must never be edited to a "desired" value.
_EXPECTED = {
    "IIAMetric":                    {"valid": 0.0,                   "failing": 0.30326918430132155},
    "BCCMetric":                    {"valid": 0.0,                   "failing": 0.0},
    "ProbingMetric":                {"valid": 1.000032736824118e-30, "failing": 1.000032736824118e-30},
    # Re-pinned 2026-06-30 to the corrected implementation's captured value after the
    # authorized I(T;Y) per-sample alignment fix (REPORT item 2 / MET-IB-001); not a "desired" edit.
    "IBLagrangianMetric":           {"valid": 0.43547196223411105,   "failing": 0.43547196223411105},
    "CIBLagrangianMetric":          {"valid": 0.0,                   "failing": 0.0},
    "ComplexityShiftMetric":        {"valid": 0.0,                   "failing": 0.0},
    "SobolSensitivityMetric":       {"valid": 0.0,                   "failing": 0.0},
    "InfidelityMetric":             {"valid": 0.0,                   "failing": 0.0},
    "SymbionMetric":                {"valid": 0.0,                   "failing": 0.5625},
    "RelationalFidelityMetric":     {"valid": 0.0,                   "failing": 0.0},
    "MacroscopicInvarianceMetric":  {"valid": 0.0,                   "failing": 0.0},
    "DCCMetric":                    {"valid": 0.0,                   "failing": 0.0},
    "MallowsCpMetric":              {"valid": 0.0,                   "failing": 0.0},
    "StructuralDeviationMetric":    {"valid": 0.0,                   "failing": 0.7615941559544727},
    "CausalSensitivityIndexMetric": {"valid": 1.0,                   "failing": 0.0},
}


def _metric_factories(logic_system):
    """Fresh-metric factories keyed by class name (metrics may carry state)."""
    valid = logic_system["valid"]
    failing = logic_system["failing"]

    def make_high_level_model(params):
        # Synthetic 1-parameter family: w>=0.5 -> valid model, else failing.
        return valid if params["w"] >= 0.5 else failing

    return {
        "IIAMetric": lambda: IIAMetric(inner_metric="mse", output_vars=_OUTPUTS, n_pairs=_N),
        "BCCMetric": lambda: BCCMetric(n_pairs=_N, inner_metric="mse"),
        "ProbingMetric": lambda: ProbingMetric(n_train=_N, inner_metric="mse"),
        "IBLagrangianMetric": lambda: IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        "CIBLagrangianMetric": lambda: CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        "ComplexityShiftMetric": lambda: ComplexityShiftMetric(output_vars=_OUTPUTS),
        "SobolSensitivityMetric": lambda: SobolSensitivityMetric(n_samples=_N, output_vars=_OUTPUTS),
        "InfidelityMetric": lambda: InfidelityMetric(output_vars=_OUTPUTS),
        "SymbionMetric": lambda: SymbionMetric(output_vars=_OUTPUTS),
        "RelationalFidelityMetric": lambda: RelationalFidelityMetric(output_vars=_OUTPUTS),
        "MacroscopicInvarianceMetric": lambda: MacroscopicInvarianceMetric(n_pairs=_N, inner_metric="mse"),
        "DCCMetric": lambda: DCCMetric(n_pairs=50, inner_metric="hard"),
        "MallowsCpMetric": lambda: MallowsCpMetric(n_params=1, output_vars=_OUTPUTS),
        "StructuralDeviationMetric": lambda: StructuralDeviationMetric(
            param_names=["w"], nominal_params={"w": 1.0},
            make_high_level_model=make_high_level_model, inner_metric="mse"),
        "CausalSensitivityIndexMetric": lambda: CausalSensitivityIndexMetric(
            param_names=["w"], nominal_params={"w": 1.0},
            make_high_level_model=make_high_level_model, inner_metric="mse"),
    }


def _run_metric(logic_system, high_level_model_key, metric):
    """Run a single analytical metric through the public engine entry point."""
    cfg = EvaluationConfig(metric="mse", seed=_SEED)
    engine = EvaluationEngine(
        logic_system[high_level_model_key],
        logic_system["low_level"],
        logic_system["vm"],
        logic_system["cg"],
        cfg,
    )
    sampler = BottomUpSampler(logic_system["vm"])
    results = engine.run_analytical_metrics([metric], sampler=sampler, n_samples=_N)
    return results[type(metric).__name__]


def _assert_pinned(actual, expected):
    assert isinstance(actual, float), f"expected a float score, got {type(actual).__name__}: {actual!r}"
    assert math.isfinite(actual), f"score is not finite: {actual!r}"
    if abs(expected) < 1e-9:
        # near-zero (perfect-abstraction collapse, or epsilon-floored value)
        assert actual == pytest.approx(0.0, abs=1e-9)
    else:
        assert actual == pytest.approx(expected, rel=1e-4)


_CASES = [(name, hlm) for name in _EXPECTED for hlm in ("valid", "failing")]


@pytest.mark.parametrize("name,high_level_model_key", _CASES, ids=[f"{n}-{h}" for n, h in _CASES])
def test_analytical_metric_value_pinned(logic_system, name, high_level_model_key):
    """Each analytical metric reproduces its captured value on the adder."""
    metric = _metric_factories(logic_system)[name]()
    actual = _run_metric(logic_system, high_level_model_key, metric)
    _assert_pinned(actual, _EXPECTED[name][high_level_model_key])


def test_battery_returns_only_finite_floats(logic_system):
    """run_analytical_metrics over the whole battery yields finite floats, never
    an ``{'error': ...}`` placeholder (guards the aggregation happy-path)."""
    factories = _metric_factories(logic_system)
    metrics = [factories[name]() for name in _EXPECTED]
    cfg = EvaluationConfig(metric="mse", seed=_SEED)
    engine = EvaluationEngine(
        logic_system["failing"], logic_system["low_level"],
        logic_system["vm"], logic_system["cg"], cfg,
    )
    sampler = BottomUpSampler(logic_system["vm"])
    results = engine.run_analytical_metrics(metrics, sampler=sampler, n_samples=_N)

    assert set(results) == set(_EXPECTED)
    for name, value in results.items():
        assert isinstance(value, float) and math.isfinite(value), f"{name} -> {value!r}"


def test_analytical_metric_reproducible_under_seed(logic_system):
    """Same seed -> identical score (the reproducibility property of the unit suite)."""
    factories = _metric_factories(logic_system)
    for name in ("IIAMetric", "SymbionMetric", "CausalSensitivityIndexMetric"):
        first = _run_metric(logic_system, "failing", factories[name]())
        second = _run_metric(logic_system, "failing", factories[name]())
        assert first == second, f"{name} not reproducible: {first!r} != {second!r}"

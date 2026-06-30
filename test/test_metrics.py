"""
Unit tests for the dissimilarity-metric family (``EvaluationMetric``).

Covers estimator correctness (known closed-form values), metric stability
(normalized scores stay in [0, 1] and are monotone in the error), and the
``get_metric`` config dispatch, including the ``'hsic'`` ->
``ConditionalIndependenceMetric`` indirection.
"""
import numpy as np
import pytest

from causal_abstraction import (
    ConditionalIndependenceMetric,
    EvaluationConfig,
    L2Metric,
    MSEMetric,
    NMSEMetric,
    R2Metric,
    RMSEMetric,
    SubspaceCheckMetric,
    get_metric,
)

_IDENTITY_METRICS = [MSEMetric, RMSEMetric, L2Metric, NMSEMetric, SubspaceCheckMetric]


@pytest.mark.parametrize("cls", _IDENTITY_METRICS)
def test_identical_inputs_score_zero(cls):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cls().measure(x, x) == pytest.approx(0.0)


def test_mse_raw_value():
    assert MSEMetric(normalize=False).measure(np.zeros(3), np.ones(3)) == pytest.approx(1.0)


def test_rmse_raw_value():
    assert RMSEMetric(normalize=False).measure(np.zeros(4), np.full(4, 2.0)) == pytest.approx(2.0)


def test_subspace_check_is_fraction_of_mismatches():
    m = SubspaceCheckMetric()
    assert m.measure(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(0.0)
    assert m.measure(np.array([1, 2, 3]), np.array([4, 5, 6])) == pytest.approx(1.0)
    assert m.measure(np.array([1, 2, 3, 4]), np.array([1, 9, 3, 9])) == pytest.approx(0.5)


def test_normalized_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=8)
        b = rng.normal(size=8)
        for cls in (MSEMetric, RMSEMetric, L2Metric):
            score = cls().measure(a, b)
            assert 0.0 <= score <= 1.0


def test_normalized_score_monotone_in_error():
    m = MSEMetric()
    base = np.zeros(5)
    assert m.measure(base, np.full(5, 0.1)) < m.measure(base, np.full(5, 2.0))


def test_metric_orientation_attributes():
    assert MSEMetric.lower_is_better is True
    assert MSEMetric.best_value == 0.0
    assert R2Metric.lower_is_better is False
    assert R2Metric.best_value == 1.0


@pytest.mark.parametrize("mode,cls", [
    ("hard", SubspaceCheckMetric),
    ("mse", MSEMetric),
    ("rmse", RMSEMetric),
    ("nmse", NMSEMetric),
    ("r2", R2Metric),
    ("hsic", ConditionalIndependenceMetric),  # 'hsic' label has no HSICMetric class
])
def test_get_metric_dispatch(mode, cls):
    assert isinstance(get_metric(EvaluationConfig(metric=mode)), cls)


def test_get_metric_unknown_raises():
    with pytest.raises(ValueError):
        get_metric(EvaluationConfig(metric="does_not_exist"))

"""
Unit tests for opt-in stochastic high-level-model (HLM) support.

These cover the additive ``noise_dist`` / exogenous-noise feature on
``CausalGraph`` (the high-level model E as a full SCM with ``P_U``):

* a regression guard that the deterministic default path is unchanged and a
  point-mass ``P_U`` reproduces deterministic ``predict`` outputs exactly, plus a
  pinned end-to-end CAE on the real 2-bit-adder fixture;
* positive tests that a stochastic node yields its analytically expected
  macro distribution, that draws are reproducible under a seed, that the existing
  divergence metrics (JSD/MMD) discriminate a wrong ``P_U`` (so no new metric is
  needed), and that a stochastic E flows through the real engine.

The deterministic-HLM path must stay byte-for-byte identical; nothing here touches
metric definitions or any stored ``test/results/*.json``.
"""
import numpy as np
import pytest

from causal_abstraction import (
    BottomUpSampler,
    DiscreteDistribution,
    EvaluationConfig,
    EvaluationEngine,
    JSDivergenceMetric,
    MMDMetric,
    TopDownSampler,
)
from causal_abstraction.models.high_level import CausalGraph, _equation_accepts_u
from causal_abstraction.tasks import StandardTasks

_N_SAMPLES = 80
_BATCH = 10
_MAX_INT = 2


# Helpers

def _deterministic_chain():
    """Y = 2*X, X a root, deterministic (no noise_dist)."""
    g = CausalGraph()
    g.add_variable("X", equation=lambda: 1, parents=[])
    g.add_variable("Y", equation=lambda X: 2 * X, parents=["X"])
    return g


def _pointmass_stochastic_clone(graph):
    """Clone ``graph`` with a degenerate point-mass ``P_U`` on every node.

    Each equation is wrapped to accept (and ignore) the noise keyword ``u``; the
    point mass at 0 makes every draw constant, so outputs must match the
    deterministic original value-for-value.
    """
    clone = CausalGraph(list(graph.variables.values()))
    point_mass = DiscreteDistribution({0: 1.0})
    for name, info in graph._nodes.items():
        wrapped = (lambda _eq: (lambda u=0, **kw: _eq(**kw)))(info["equation"])
        clone.add_variable(name, equation=wrapped, parents=list(info["parents"]),
                           noise_dist=point_mass)
    return clone


def _draw_macro(graph, var, n, seed):
    """Empirical path-p samples of ``var`` from a stochastic graph under a seed."""
    rng = np.random.default_rng(seed)
    return np.array([graph.predict({}, rng=rng)[var] for _ in range(n)], dtype=float)


def _cae_scores(model, logic_system, seed):
    cfg = EvaluationConfig(metric="hard", seed=seed)
    engine = EvaluationEngine(
        model, logic_system["low_level"], logic_system["vm"], logic_system["cg"], cfg
    )
    td = TopDownSampler(logic_system["vm"])
    bu = BottomUpSampler(logic_system["vm"])
    tasks = [
        StandardTasks.score(engine.builder, sampler=td, name="CAE_down", include_faithfulness=True),
        StandardTasks.score(engine.builder, sampler=bu, name="CAE_up", include_faithfulness=True),
    ]
    return engine.run_tasks(
        tasks, n_samples=_N_SAMPLES, batch_size=_BATCH, max_interventions=_MAX_INT
    ).summary()


# Signature-introspection helper

def test_equation_accepts_u():
    assert _equation_accepts_u(lambda x, u: x) is True          # explicit u
    assert _equation_accepts_u(lambda u=0: u) is True           # u with default
    assert _equation_accepts_u(lambda x, **kw: x) is True       # **kwargs
    assert _equation_accepts_u(lambda x: x) is False            # no u
    assert _equation_accepts_u(lambda: 0) is False              # nullary


# Regression: deterministic default unchanged

def test_deterministic_default_is_not_stochastic_and_unchanged():
    g = _deterministic_chain()
    assert g.is_stochastic is False
    # predict is unchanged and ignores a passed rng (no draws happen).
    rng = np.random.default_rng(123)
    assert g.predict({"X": 3}) == {"X": 3, "Y": 6}
    assert g.predict({"X": 3}, rng=rng) == {"X": 3, "Y": 6}
    # The rng must be untouched by a deterministic predict (byte-for-byte path).
    probe = np.random.default_rng(123)
    assert rng.integers(0, 1_000_000) == probe.integers(0, 1_000_000)


def test_pointmass_reproduces_deterministic_predict():
    det = _deterministic_chain()
    sto = _pointmass_stochastic_clone(det)
    assert sto.is_stochastic is True
    rng = np.random.default_rng(0)
    for x in (-2, 0, 1, 5, 17):
        assert sto.predict({"X": x}, rng=rng) == det.predict({"X": x}), f"mismatch at X={x}"


# Construction-time validation

def test_noise_dist_requires_u_in_equation():
    g = CausalGraph()
    with pytest.raises(ValueError, match="does not accept a 'u'"):
        g.add_variable("Y", equation=lambda X: X, parents=["X"],
                       noise_dist=DiscreteDistribution({0: 1.0}))


# Positive: stochastic node matches the expected macro distribution

@pytest.mark.parametrize("p1", [0.5, 0.9, 0.1])
def test_stochastic_node_matches_expected_distribution(p1):
    g = CausalGraph()
    g.add_variable("Y", equation=lambda u: u, parents=[],
                   noise_dist=DiscreteDistribution({0: 1.0 - p1, 1: p1}))
    draws = _draw_macro(g, "Y", n=4000, seed=2024)
    frac1 = float(np.mean(draws == 1.0))
    # Monte-Carlo tolerance for n=4000 (~3 std ≈ 0.024 at p=0.5).
    assert abs(frac1 - p1) < 0.03


def test_stochastic_predict_reproducible_under_seed():
    g = CausalGraph()
    g.add_variable("Y", equation=lambda u: u, parents=[],
                   noise_dist=DiscreteDistribution({0: 0.5, 1: 0.5}))
    first = _draw_macro(g, "Y", n=200, seed=7)
    second = _draw_macro(g, "Y", n=200, seed=7)
    assert np.array_equal(first, second)              # same seed -> identical
    other = _draw_macro(g, "Y", n=200, seed=8)
    assert not np.array_equal(first, other)           # different seed -> differs


def test_explicit_u_overrides_the_draw():
    g = CausalGraph()
    g.add_variable("Y", equation=lambda u: u, parents=[],
                   noise_dist=DiscreteDistribution({0: 0.5, 1: 0.5}))
    # Supplying u bypasses the noise_dist draw entirely.
    assert g.predict({}, u={"Y": 1})["Y"] == 1
    assert g.predict({}, u={"Y": 0})["Y"] == 0


# Positive: existing divergence metrics consume the macro distribution and
# discriminate a wrong P_U (confirms no new metric is needed).

def test_divergence_metrics_discriminate_wrong_pu():
    e_valid = CausalGraph()
    e_valid.add_variable("Y", equation=lambda u: u, parents=[],
                         noise_dist=DiscreteDistribution({0: 0.5, 1: 0.5}))
    samples_p = _draw_macro(e_valid, "Y", n=500, seed=1)

    # Matched reference (true P_U) vs a wrong P_U skewed to 0.95.
    ref_match = _draw_macro(e_valid, "Y", n=500, seed=2)
    e_wrong = CausalGraph()
    e_wrong.add_variable("Y", equation=lambda u: u, parents=[],
                         noise_dist=DiscreteDistribution({0: 0.05, 1: 0.95}))
    ref_wrong = _draw_macro(e_wrong, "Y", n=500, seed=3)

    for metric in (JSDivergenceMetric(), MMDMetric()):
        d_match = metric.measure(samples_p, ref_match)
        d_wrong = metric.measure(samples_p, ref_wrong)
        assert d_wrong > d_match, f"{type(metric).__name__}: wrong {d_wrong} !> match {d_match}"


# Regression + integration on the real (M, E, tau) fixture

def test_deterministic_cae_pinned(logic_system):
    """Pin the deterministic-HLM CAE on the 2-bit adder, protecting reported numbers."""
    scores = _cae_scores(logic_system["valid"], logic_system, seed=0)
    assert logic_system["valid"].is_stochastic is False
    assert scores["CAE_down"] == 0.0
    assert scores["CAE_up"] == 0.0


def test_stochastic_hlm_runs_through_engine(logic_system):
    """A stochastic E (point-mass clone of the valid HLM) flows through the real
    engine + metrics and still scores the valid abstraction near zero."""
    sto = _pointmass_stochastic_clone(logic_system["valid"])
    assert sto.is_stochastic is True
    scores = _cae_scores(sto, logic_system, seed=0)
    assert scores["CAE_down"] < 0.05
    assert scores["CAE_up"] < 0.05


def _noisy_clone(graph, p=0.5):
    """Clone with a real Bernoulli ``P_U`` whose draw perturbs each non-root node,
    so the exogenous noise actually changes the macro output through the engine."""
    clone = CausalGraph(list(graph.variables.values()))
    noise = DiscreteDistribution({0: 1.0 - p, 1: p})
    for name, info in graph._nodes.items():
        parents = list(info["parents"])
        if parents:
            wrapped = (lambda _eq: (lambda u=0, **kw: _eq(**kw) + int(u)))(info["equation"])
            clone.add_variable(name, equation=wrapped, parents=parents, noise_dist=noise)
        else:
            clone.add_variable(name, equation=info["equation"], parents=parents)
    return clone


def test_stochastic_hlm_engine_reproducible_under_seed(logic_system):
    """The engine threads its seeded per-batch rng into a stochastic E, so noise
    that genuinely perturbs the output is still bit-reproducible under a fixed seed
    (would fail if predict() drew from an unseeded RNG)."""
    noisy = _noisy_clone(logic_system["valid"], p=0.5)
    assert noisy.is_stochastic is True
    first = _cae_scores(noisy, logic_system, seed=0)
    second = _cae_scores(noisy, logic_system, seed=0)
    assert first == second                                   # reproducible via threaded seed
    # The injected exogenous noise actually flows through to the macro output:
    # the otherwise-valid abstraction is now degraded above its deterministic 0.
    assert first["CAE_down"] > 0.0

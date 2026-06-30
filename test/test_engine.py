"""
End-to-end unit tests for ``EvaluationEngine`` on a tiny real abstraction
(the 2-bit adder logic circuit).

Covers the three core properties of the unit suite at the
engine level: a correct (valid) abstraction scores ~0 under CAE (estimator
correctness), repeated seeded runs are bit-identical (reproducibility of
baseline results), and an invalid abstraction scores strictly higher
(discrimination, the core estimator behaviour).
"""
from causal_abstraction import (
    BottomUpSampler,
    EvaluationConfig,
    EvaluationEngine,
    TopDownSampler,
)
from causal_abstraction.tasks import StandardTasks

_N_SAMPLES = 80
_BATCH = 10
_MAX_INT = 2


def _cae_scores(logic_system, model, seed):
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


def test_valid_abstraction_scores_near_zero(logic_system):
    scores = _cae_scores(logic_system, logic_system["valid"], seed=0)
    assert scores["CAE_down"] < 0.05
    assert scores["CAE_up"] < 0.05


def test_results_reproducible_under_seed(logic_system):
    first = _cae_scores(logic_system, logic_system["valid"], seed=0)
    second = _cae_scores(logic_system, logic_system["valid"], seed=0)
    assert first == second


def test_invalid_abstraction_scores_higher(logic_system):
    valid = _cae_scores(logic_system, logic_system["valid"], seed=0)
    failing = _cae_scores(logic_system, logic_system["failing"], seed=0)
    assert failing["CAE_down"] > valid["CAE_down"]
    assert failing["CAE_up"] > valid["CAE_up"]

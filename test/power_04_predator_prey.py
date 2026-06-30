"""
Statistical power / convergence study for the Predator-Prey (LV vs ABM).
"""
from pathlib import Path

import numpy as np

from utils import load_system

_pp = load_system("04_predator_prey.py", "predator_prey")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, ContinuousValueMap,
    EvaluationConfig, EvaluationEngine,
    FullSubspace, RectSubspace,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.schema import Variable
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_04_predator_prey.json"
N_GRID       = [2, 4, 6, 8, 10, 15, 20, 30]
TARGET_RUNS  = 50
CONDITIONS   = ["valid", "fail_alpha", "spatial", "stochastic", "aging", "noise", "complex"]

# Shared setup

_BASE_CONFIG = {
    "prey_reproduce_prob": 0.1, "predation_prob": 0.001,
    "predator_reproduce_prob": 0.5, "predator_starvation_prob": 0.05,
    "num_steps": 20, "grid_size": 20,
    "prey_age_death_prob": 0.02, "predator_age_death_prob": 0.04,
}
_IDEAL_CONFIG = dict(_BASE_CONFIG, spatial=False,
                     stochastic_reproduction=False, agent_aging=False)
_LOW_LEVEL_MODEL_IDEAL = _pp.AgentBasedModel(_IDEAL_CONFIG)

_SEARCH = {
    "alpha": np.linspace(0.08, 0.12, 3), "beta": np.linspace(0.0008, 0.0012, 3),
    "delta": np.linspace(0.0004, 0.0006, 3), "gamma": np.linspace(0.04, 0.06, 3),
}
print("Calibrating Lotka-Volterra parameters (one-time)...")
_HIGH_LEVEL_MODEL_PARAMS = _pp.calibrate_high_level_model(_LOW_LEVEL_MODEL_IDEAL, _SEARCH)
print(f"  params: {_HIGH_LEVEL_MODEL_PARAMS}\n")

_PREY_DOM = RectSubspace((50, 250))
_PRED_DOM = RectSubspace((20, 100))
_OUT_DOM  = FullSubspace(2)
_SCHEMA   = MicroVariableSchema([Variable(x) for x in
                                  ["prey_t", "predator_t", "final_populations"]])
_CG       = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM       = ContinuousValueMap(_CG, {
    "prey_t": {0: _PREY_DOM}, "predator_t": {0: _PRED_DOM},
    "final_populations": {0: _OUT_DOM},
})


def _make_high_level_model(params):
    g = CausalGraph()
    g.add_variable("prey_t", lambda: None, domain=_PREY_DOM)
    g.add_variable("predator_t", lambda: None, domain=_PRED_DOM)
    g.add_variable("final_populations",
                   lambda prey_t, predator_t: _pp.solve_lotka_volterra(
                       prey_t, predator_t, params),
                   parents=["prey_t", "predator_t"], domain=_OUT_DOM)
    return g


def _make_low_level_model(spatial, stochastic, aging):
    return _pp.AgentBasedModel(dict(_BASE_CONFIG, spatial=spatial,
                                    stochastic_reproduction=stochastic,
                                    agent_aging=aging))


def _build_tasks(builder):
    td = TopDownSampler(_VM)
    bu = BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    if cond == "fail_alpha":
        params = dict(_HIGH_LEVEL_MODEL_PARAMS, alpha=_HIGH_LEVEL_MODEL_PARAMS["alpha"] * 1.5)
        low_level_model    = _LOW_LEVEL_MODEL_IDEAL
    else:
        params = _HIGH_LEVEL_MODEL_PARAMS
        low_level_model    = {
            "valid":      _LOW_LEVEL_MODEL_IDEAL,
            "spatial":    _make_low_level_model(True,  False, False),
            "stochastic": _make_low_level_model(False, True,  False),
            "aging":      _make_low_level_model(False, False, True),
            "noise":      NoisyLowLevelModel(_LOW_LEVEL_MODEL_IDEAL, noise_std=1.0,
                                             noise_type="gaussian"),
            "complex":    _make_low_level_model(True, True, True),
        }[cond]
    high_level_model    = _make_high_level_model(params)
    seed   = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=BottomUpSampler(_VM),
        task_kwargs=dict(batch_size=1, max_interventions=2,
                         intervention_domain=["prey_t", "predator_t"]),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Predator-Prey (LV vs ABM) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

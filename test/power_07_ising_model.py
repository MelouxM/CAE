"""
Statistical power / convergence study for the Ising Model.
"""
from pathlib import Path

from utils import load_system

_ising = load_system("07_ising_model.py", "ising_model")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, ContinuousValueMap,
    EvaluationConfig, EvaluationEngine,
    RectSubspace, FullSubspace,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_07_ising_model.json"
N_GRID       = [2, 4, 6, 8, 10, 15, 20, 30]
TARGET_RUNS  = 30
CONDITIONS   = ["valid", "fail", "noise"]
GRID_SIDE    = 8
HIGH_LEVEL_MODEL_SWEEPS   = (500, 200)

# Shared objects

_D_TEMP  = RectSubspace((0.5, 4.0))
_D_FIELD = RectSubspace((-0.5, 0.5))
_D_MAG   = FullSubspace(1)
_SCHEMA  = MicroVariableSchema.from_names(
    ["Temperature", "ExternalField", "PredictedMagnetization"])
_CG      = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM      = ContinuousValueMap(_CG, {
    "Temperature": {0: _D_TEMP}, "ExternalField": {0: _D_FIELD},
    "PredictedMagnetization": {0: _D_MAG},
})
_LOW_LEVEL_MODEL = _ising.MolecularDynamicsIsingModel(grid_side=GRID_SIDE, vibrational_temp=0.0)
_LOW_LEVEL_MODEL.params["equil_sweeps"]   = HIGH_LEVEL_MODEL_SWEEPS[0]
_LOW_LEVEL_MODEL.params["measure_sweeps"] = HIGH_LEVEL_MODEL_SWEEPS[1]


def _make_high_level_model(J):
    g = CausalGraph()
    g.add_variable("Temperature",   lambda: None, domain=_D_TEMP)
    g.add_variable("ExternalField", lambda: None, domain=_D_FIELD)
    def predict_mag(Temperature, ExternalField):
        seed = (int(round(Temperature, 8) * 1e8)
                + int(round(ExternalField + 10, 8) * 1e8))
        return _ising._run_rigid_simulation(
            GRID_SIDE, Temperature, ExternalField, J,
            HIGH_LEVEL_MODEL_SWEEPS[0], HIGH_LEVEL_MODEL_SWEEPS[1], seed)
    g.add_variable("PredictedMagnetization", predict_mag,
                   parents=["Temperature", "ExternalField"], domain=_D_MAG)
    return g


def _build_tasks(builder):
    s  = TopDownSampler(_VM)
    bu = BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=s,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=s,  name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    high_level_model  = _make_high_level_model(J=2.0 if cond == "fail" else 1.0)
    low_level_model  = (NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.25, noise_type="gaussian")
            if cond == "noise" else _LOW_LEVEL_MODEL)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(batch_size=1, max_interventions=2,
                         intervention_domain=["Temperature", "ExternalField"]),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Ising Model (rigid vs MD) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

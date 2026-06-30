"""
Statistical power / convergence study for the 1D Heat Equation.
"""
from pathlib import Path

from utils import load_system

_h = load_system("05_heat_equation.py", "heat_equation")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, FullSubspace,
    EvaluationConfig, EvaluationEngine,
    BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_05_heat_equation.json"
N_GRID       = [2, 4, 6, 8, 10, 15, 20, 30]
TARGET_RUNS  = 50
CONDITIONS   = ["valid", "fail", "noise"]

L, ALPHA, T_MAX, STEPS, BINS, N_PART = 1.0, 0.1, 0.2, 200, 50, 1000
DT = T_MAX / STEPS

# Shared objects

_LOW_LEVEL_MODEL    = _h.BrownianParticleSystem(n_particles=N_PART, n_steps=STEPS,
                                    diff_coeff=ALPHA, box_len=L, dt=DT)
_SCHEMA = MicroVariableSchema.from_names(["T_initial", "T_final"])
_CG     = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM     = _h.BinningValueMap(_CG,
              {"T_initial": {0: FullSubspace(BINS)},
               "T_final":   {0: FullSubspace(BINS)}},
              n_bins=BINS, box_len=L, n_particles=N_PART)
_SAMPLER = _h.SmoothProfileSampler(_VM, BINS)


def _make_high_level_model(diff_coeff):
    solver = _h.HeatEquationSolver(n_bins=BINS, diff_coeff=diff_coeff,
                                    box_len=L, dt=DT, n_steps=STEPS)
    g = CausalGraph()
    g.add_variable("T_initial", lambda: None, domain=FullSubspace(BINS))
    g.add_variable("T_final", lambda T_initial: solver.solve(T_initial),
                   parents=["T_initial"], domain=FullSubspace(BINS))
    return g


def _build_tasks(builder):
    s  = _SAMPLER
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
    high_level_model  = _make_high_level_model(ALPHA if cond != "fail" else 0.0)
    low_level_model  = (NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=0.15, noise_type="gaussian")
            if cond == "noise" else _LOW_LEVEL_MODEL)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=_SAMPLER,
        task_kwargs=dict(batch_size=1, max_interventions=1,
                         intervention_domain=["T_initial"]),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Heat Equation 1D (Brownian diffusion) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )
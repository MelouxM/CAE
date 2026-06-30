"""
Statistical power / convergence study for the 2D Heat Equation (phonon lattice).
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from utils import load_system

_h2d = load_system("06_heat_equation_2d.py", "heat_equation_2d")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap,
    EvaluationConfig, EvaluationEngine,
    RectSubspace, FullSubspace,
    TopDownSampler, BottomUpSampler, NoisyLowLevelModel,
)
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_06_heat_equation_2d.json"
N_GRID       = [2, 4, 6, 8, 12, 20]
TARGET_RUNS  = 20
CONDITIONS   = ["valid", "fail", "noise"]

GRID_SIZE, SIM_TIME, P_SCATTER, N_AVG = 16, 5.0, 0.2, 10
MAP_SIGMA = max(0.1, GRID_SIZE * 0.05)

# Shared objects

_SRC_X  = RectSubspace((0.3, 0.7))
_SRC_Y  = RectSubspace((0.3, 0.7))
_SRC_E  = RectSubspace((500.0, 1000.0))
_OUT_T  = FullSubspace(GRID_SIZE * GRID_SIZE)
_SCHEMA = MicroVariableSchema.from_names(
    ["source_x", "source_y", "source_E", "final_temp_map"])
_CG     = CoarseGrainingMap(_SCHEMA, {k: [k] for k in _SCHEMA.variable_names})
_VM     = _h2d.SpatialValueMap(_CG,
              {"source_x": {0: _SRC_X}, "source_y": {0: _SRC_Y},
               "source_E": {0: _SRC_E}, "final_temp_map": {0: _OUT_T}},
              (GRID_SIZE, GRID_SIZE))
_LOW_LEVEL_MODEL = _h2d.PhononLatticeModel(grid_h=GRID_SIZE, grid_w=GRID_SIZE,
                                 p_scatter=P_SCATTER, sim_time=SIM_TIME,
                                 n_averages=N_AVG)

print("Calibrating 2D heat equation alpha (one-time)...")
_ALPHA = _h2d.calibrate_alpha_opt(
    GRID_SIZE, P_SCATTER, SIM_TIME, MAP_SIGMA, n_calib_averages=N_AVG)
print(f"  alpha = {_ALPHA:.6f}\n")


def _make_high_level_model(alpha):
    g = CausalGraph()
    g.add_variable("source_x", lambda: None, domain=_SRC_X)
    g.add_variable("source_y", lambda: None, domain=_SRC_Y)
    g.add_variable("source_E", lambda: None, domain=_SRC_E)
    def predict(source_x, source_y, source_E):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        grid[int(source_y * GRID_SIZE), int(source_x * GRID_SIZE)] = 1.0
        grid = gaussian_filter(grid, sigma=MAP_SIGMA)
        final = _h2d.solve_heat_pde_neumann(grid, alpha, dt=0.2,
                                             steps=int(SIM_TIME / 0.2))
        t = np.sum(final)
        return final / t if t > 1e-9 else final
    g.add_variable("final_temp_map", predict,
                   parents=["source_x", "source_y", "source_E"], domain=_OUT_T)
    return g


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
    seed  = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    alpha = _ALPHA * 10.0 if cond == "fail" else _ALPHA
    high_level_model   = _make_high_level_model(alpha)
    low_level_model   = (NoisyLowLevelModel(_LOW_LEVEL_MODEL, noise_std=150.0, noise_type="gaussian")
             if cond == "noise" else _LOW_LEVEL_MODEL)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=TopDownSampler(_VM),
        task_kwargs=dict(batch_size=1, max_interventions=3,
                         intervention_domain=["source_x", "source_y", "source_E"]),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Heat Equation 2D (phonon lattice) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )
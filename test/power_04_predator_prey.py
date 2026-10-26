"""
Statistical power / convergence study for the predator-prey system.

The conditions are the evaluation suite's own (`test/04_predator_prey.py`): same models,
maps, samplers and task settings. The power runner keeps only the CAE tasks and
varies `n_samples`.
"""
from pathlib import Path

from utils import load_system

from power_runner import run_power_suite, parse_power_args

TEST_DIR = Path(__file__).parent
_suite = load_system(TEST_DIR / "04_predator_prey.py", "suite_04_predator_prey")

# Config

RESULTS_FILE = TEST_DIR / "results" / "power_04_predator_prey.json"
N_GRID       = [2, 4, 6, 8, 10, 15, 20, 30]
TARGET_RUNS  = 50
CONDITIONS   = _suite.CONDITIONS

build_condition = _suite.build_condition


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

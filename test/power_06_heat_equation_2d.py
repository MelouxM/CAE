"""
Statistical power / convergence study for the 2D heat equation (phonon lattice).

The conditions are the evaluation suite's own (`test/06_heat_equation_2d.py`): same models,
maps, samplers and task settings. The power runner keeps only the CAE tasks and
varies `n_samples`.
"""
from pathlib import Path

from utils import load_system

from power_runner import run_power_suite, parse_power_args

TEST_DIR = Path(__file__).parent
_suite = load_system(TEST_DIR / "06_heat_equation_2d.py", "suite_06_heat_equation_2d")

# Config

RESULTS_FILE = TEST_DIR / "results" / "power_06_heat_equation_2d.json"
N_GRID       = [2, 4, 6, 8, 12, 20]
TARGET_RUNS  = 20
CONDITIONS   = _suite.CONDITIONS

build_condition = _suite.build_condition


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

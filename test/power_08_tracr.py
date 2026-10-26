"""
Statistical power / convergence study for the Tracr sort-rank transformer.

The conditions are the evaluation suite's own (`test/08_tracr.py`): same models,
maps, samplers and task settings. The power runner keeps only the CAE tasks and
varies `n_samples`.
"""
from pathlib import Path

from utils import load_system

from power_runner import run_power_suite, parse_power_args

TEST_DIR = Path(__file__).parent
_suite = load_system(TEST_DIR / "08_tracr.py", "suite_08_tracr")

# Config

RESULTS_FILE = TEST_DIR / "results" / "power_08_tracr.json"
N_GRID       = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS  = 100
CONDITIONS   = _suite.CONDITIONS

build_condition = _suite.build_condition


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Tracr (RASP sort-rank transformer) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

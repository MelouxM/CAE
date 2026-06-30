"""
Runs all power/convergence experiments sequentially.

Fast systems run first; slow systems (heat2D, transistor, gas) run last.
Each system uses its own TARGET_RUNS default unless overridden with --runs.
The --runs flag only applies to systems whose default is 100; systems with
explicitly lower defaults (gas=20, heat2D=20, ising=30, predator-prey=50,
heat1D=50, CPU=50) keep those values.

Usage:
    python run_power.py              # all systems, default run counts
    python run_power.py --runs 50   # cap fast-system runs at 50
    python run_power.py --skip 03 06  # skip specific systems by number
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from utils import load_system

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))

from power_runner import run_power_suite


def _load(filename: str):
    return load_system(TEST_DIR / filename, filename[:-3])


def _args_with_runs(runs: int) -> SimpleNamespace:
    return SimpleNamespace(runs=runs)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run all power/convergence experiments")
    p.add_argument("--runs", type=int, default=None,
                   help="Override run count for 100-default systems.")
    p.add_argument("--skip", nargs="+", default=[],
                   help="System numbers to skip, e.g. --skip 03 06.")
    return p.parse_args(argv)


def main():
    args  = _parse_args()
    skip  = set(args.skip)
    # Run count for systems whose default is 100; others keep their own defaults.
    fast_runs = args.runs if args.runs is not None else 100

    # Order: fast → medium → slow (heat2D → transistor → gas)
    plan = [
        # (system_number_str, filename, target_runs)
        # Fast / discrete
        ("01", "power_01_logic_circuit.py",     fast_runs),
        ("08", "power_08_tracr.py",              fast_runs),
        ("09", "power_09_grn.py",                fast_runs),
        # Controlled experiments (handled separately below)
        ("11", None,                             fast_runs),
        # Medium
        ("05", "power_05_heat_equation.py",      50),
        ("04", "power_04_predator_prey.py",      50),
        ("07", "power_07_ising_model.py",        30),
        ("10", "power_10_cpu_6502.py",           50),
        # Slow - always last
        ("06", "power_06_heat_equation_2d.py",   20),
        ("02", "power_02_transistor_circuit.py", fast_runs),
        #("03", "power_03_gas_simulation.py",     20),
    ]

    for sys_num, filename, target_runs in plan:
        if sys_num in skip:
            print(f"\n[Skipping system {sys_num}]")
            continue

        print(f"\n{'#' * 60}")

        if sys_num == "11":
            # Controlled experiments: load module and iterate sub-experiments.
            print(f"# System 11 - Controlled experiments ({target_runs} rounds each)")
            print(f"{'#' * 60}")
            m11 = _load("power_11_controlled.py")
            for eid in sorted(m11.POWER_EXPERIMENTS.keys()):
                exp = m11.POWER_EXPERIMENTS[eid]
                run_power_suite(
                    title=exp["title"],
                    results_file=exp["results_file"],
                    conditions=exp["conditions"],
                    build_condition=exp["build_condition"],
                    n_grid=exp["n_grid"],
                    target_runs=target_runs,
                    args=_args_with_runs(target_runs),
                )
        else:
            print(f"# System {sys_num} - {filename}  ({target_runs} rounds)")
            print(f"{'#' * 60}")
            mod = _load(filename)
            run_power_suite(
                title=getattr(mod, "__doc__", filename).strip().split("\n")[0],
                results_file=mod.RESULTS_FILE,
                conditions=mod.CONDITIONS,
                build_condition=mod.build_condition,
                n_grid=mod.N_GRID,
                target_runs=target_runs,
                args=_args_with_runs(target_runs),
            )

    print("\nAll power experiments complete.")


if __name__ == "__main__":
    main()
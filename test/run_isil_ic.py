"""Run the ISIL/IC baselines (Zennaro et al., 2023) on the benchmark suites.

Reuses each suite's `build_condition`, keeps only the ISIL/IC tasks and writes the
results to `test/results/isil_ic/`.

Usage:
    python test/run_isil_ic.py --system 01 --runs 10
    python test/run_isil_ic.py --system 09 --runs 10
    python test/run_isil_ic.py --system 07 --runs 10            # any suite 01-11
    python test/run_isil_ic.py --system 11 --runs 10            # all six experiments
    python test/run_isil_ic.py --system 11 --experiment 2 5 --runs 10
    python test/run_isil_ic.py --system all --runs 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils import load_system
from causal_abstraction.tasks import ZennaroTasks
from runner import run_suite

TEST_DIR = Path(__file__).parent
OUT_DIR = TEST_DIR / "results" / "isil_ic"

SUITE_FILES = {
    "01": "01_logic_circuit.py",
    "02": "02_transistor_circuit.py",
    "03": "03_gas_simulation.py",
    "04": "04_predator_prey.py",
    "05": "05_heat_equation.py",
    "06": "06_heat_equation_2d.py",
    "07": "07_ising_model.py",
    "08": "08_tracr.py",
    "09": "09_grn.py",
    "10": "10_cpu_6502.py",
    "11": "11_controlled_experiments.py",
}


def relative_edges(rel: float, floor: float = 1e-3, top: float = 1e6) -> np.ndarray:
    """Bin edges of relative width `rel` on |x| in [floor, top], for both signs.

    Values below `floor` in magnitude share one bin.
    """
    pos = floor * (1.0 + rel) ** np.arange(int(np.ceil(np.log(top / floor) / np.log1p(rel))) + 1)
    return np.concatenate([-pos[::-1], pos])


# Suites that don't build the ISIL/IC tasks themselves. ISIL/IC use the suite's CAE
# samplers. Continuous outputs are binned at 5% (main columns) and 25% (`*_coarse`).
# `n` is the number of cells, `reps` the draws per cell. The 6502 has its own
# --isil-ic mode.
_REL_FINE, _REL_COARSE = relative_edges(0.05), relative_edges(0.25)
_ISING_FINE = {"PredictedMagnetization": np.arange(-1.05, 1.0501, 0.05)}
_ISING_COARSE = {"PredictedMagnetization": np.arange(-1.25, 1.2501, 0.25)}
EXTRA_SYSTEMS = {
    "03": dict(n=10, reps=5),
    "04": dict(n=50, reps=5),
    "05": dict(n=30, reps=5),
    "06": dict(n=10, reps=3),
    "07": dict(n=30, reps=5, fine=_ISING_FINE, coarse=_ISING_COARSE),
    "08": dict(n=100, reps={"noise": 20}, discrete=True),
}


def _with_isil_tasks(build_condition, cfg):
    """Add the ISIL/IC tasks to a suite's condition factory (systems in EXTRA_SYSTEMS)."""

    def wrapped(cond, run_index):
        cr = build_condition(cond, run_index)
        by_name = {t.name: t for t in cr.tasks}
        reps = cfg["reps"].get(cond, 1) if isinstance(cfg["reps"], dict) else cfg["reps"]
        kw = {}
        if not cfg.get("discrete"):
            names = list(cr.engine.high_level_model.variables)
            fine = {n: _REL_FINE for n in names} | cfg.get("fine", {})
            coarse = {n: _REL_COARSE for n in names} | cfg.get("coarse", {})
            kw = dict(bin_edges=fine, extra_resolutions={"coarse": coarse})
        cr.tasks = list(cr.tasks) + ZennaroTasks.suite(
            cr.engine.builder, n_instances=cfg["n"], n_replicates=reps,
            top_down=by_name["CAE_down_nf"].sampler, bottom_up=by_name["CAE_up_nf"].sampler, **kw)
        return cr

    return wrapped


# With --reference, the CAE columns are recomputed next to ISIL/IC
CAE_NAMES = ("CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up")


def _isil_only(build_condition, keep):
    """Wrap a suite's condition factory to keep only the `keep` tasks."""

    def wrapped(cond, run_index):
        cr = build_condition(cond, run_index)
        cr.tasks = [t for t in cr.tasks if t.name in keep]
        cr.analytical = []
        return cr

    return wrapped


def _run(title, results_file, conditions, comparisons, build_condition, runs,
         keep=None, print_every=5):
    run_suite(
        title=title,
        results_file=results_file,
        conditions=conditions,
        comparisons=comparisons,
        build_condition=_isil_only(build_condition, keep or set(ZennaroTasks.NAMES)),
        print_table_every=print_every,
        args=SimpleNamespace(metrics=None, runs=runs),
    )


def run_system(system: str, runs: int, experiments=None, keep=None) -> None:
    if system == "10":
        # The 6502 suite has its own ISIL/IC mode
        subprocess.run([sys.executable, str(TEST_DIR / SUITE_FILES[system]), "--isil-ic",
                        "--runs", str(runs)], check=True)
        return
    suite = load_system(str(TEST_DIR / SUITE_FILES[system]), f"suite_{system}")

    if system == "11":
        ids = experiments or sorted(suite.EXPERIMENTS)
        for eid in ids:
            e = suite.EXPERIMENTS[eid]
            print(f"\n{'=' * 70}\n  ISIL/IC - {e['title']}\n{'=' * 70}\n")
            _run(
                title=f"ISIL/IC - {e['title']}",
                results_file=OUT_DIR / f"11_exp{eid}_isil_ic_results.json",
                conditions=e["conditions"],
                comparisons=e["comparisons"],
                build_condition=e["build_condition"],
                runs=runs,
                keep=keep,
            )
        return

    name = SUITE_FILES[system].replace(".py", "")
    build = suite.build_condition
    if system in EXTRA_SYSTEMS:
        build = _with_isil_tasks(build, EXTRA_SYSTEMS[system])
        keep = (keep or set(ZennaroTasks.NAMES)) | {f"{n}_coarse" for n in ZennaroTasks.NAMES}
    print(f"\n{'=' * 70}\n  ISIL/IC - {name}\n{'=' * 70}\n")
    _run(
        title=f"ISIL/IC - {name}",
        results_file=OUT_DIR / f"{name}_isil_ic_results.json",
        conditions=suite.CONDITIONS,
        comparisons=suite.COMPARISONS,
        build_condition=build,
        runs=runs,
        keep=keep,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the ISIL/IC baseline columns")
    p.add_argument("--system", default="all", choices=[*SUITE_FILES, "all"],
                   help="Benchmark system to run (default: all).")
    p.add_argument("--experiment", nargs="+", type=int, default=None,
                   help="For --system 11: which controlled experiments (1-6).")
    p.add_argument("--runs", type=int, default=10, help="Number of rounds.")
    p.add_argument("--reference", action="store_true",
                   help="Also compute the CAE columns, for reference.")
    args = p.parse_args(argv)

    keep = set(ZennaroTasks.NAMES)
    if args.reference:
        keep |= set(CAE_NAMES)

    systems = list(SUITE_FILES) if args.system == "all" else [args.system]
    for s in systems:
        run_system(s, args.runs, args.experiment, keep=keep)


if __name__ == "__main__":
    main()

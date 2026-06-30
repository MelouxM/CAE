"""
Shared runner for statistical power / convergence experiments.

Data structure saved per system:
    data[condition][metric_name][str(n)] = [score_run0, score_run1, ...]

Supports:
  - Convergence plots : mean +/- CI of score vs n
  - Power curves      : fraction of runs where a test rejects H0 at each n

Round structure: outer loop is round_idx, middle is n, inner is condition.
After every complete round every (condition, n) cell has exactly one new
data point, so stopping early yields balanced data across all sample sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

POWER_METRICS = frozenset({"CAE_down", "CAE_down_nf", "CAE_up_nf", "CAE_up"})


# Persistence

def load_power_results(results_file: Path, conditions: List[str]) -> Dict:
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            print(f"Loaded existing results from {results_file.name}")
            return data
        except Exception as e:
            print(f"Warning: could not load {results_file.name} ({e}). Starting fresh.")
    return {cond: {} for cond in conditions}


def save_power_results(data: Dict, results_file: Path) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(results_file)


# Helpers

def _count(data: Dict, cond: str, metric: str, n: int) -> int:
    return len(data.get(cond, {}).get(metric, {}).get(str(n), []))


def _append(data: Dict, cond: str, metric: str, n: int, score: float) -> None:
    stored = None if (score is None or (isinstance(score, float) and np.isnan(score))) else float(score)
    (data.setdefault(cond, {})
         .setdefault(metric, {})
         .setdefault(str(n), [])
         .append(stored))


def _cell_done(data: Dict, cond: str, n: int, round_idx: int) -> bool:
    # Done if any metric already has round_idx+1 entries for this cell.
    # Uses max so that a metric failing in one round doesn't block the cell.
    best = max((_count(data, cond, m, n) for m in POWER_METRICS), default=0)
    return best > round_idx


# Main runner

def run_power_suite(
    title: str,
    results_file: Path,
    conditions: List[str],
    build_condition: Callable,
    n_grid: List[int],
    target_runs: int = 100,
    args=None,
) -> None:
    """
    Run the power/convergence experiment for one system.

    build_condition(cond, run_index) must return a Condition whose
    task_kwargs does NOT include n_samples (the runner injects it).
    All other kwargs (batch_size, max_interventions, intervention_domain)
    are preserved.
    """
    data   = load_power_results(results_file, conditions)
    target = (args.runs if args is not None and getattr(args, "runs", None) is not None
              else target_runs)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  n_grid = {n_grid}  |  target_runs = {target}")
    print(f"{'=' * 60}\n")

    # Find the round to resume from: the minimum number of completed entries
    # any (condition, n) cell has across all metrics.  Starting from 0 would
    # cause _cell_done to return True for every cell (since best > 0) and
    # immediately trigger the "all done" early exit.
    start_round = min(
        max((_count(data, cond, m, n) for m in POWER_METRICS), default=0)
        for cond in conditions
        for n in n_grid
    )
    if start_round > 0:
        print(f"  Resuming from round {start_round + 1}.\n")

    try:
        for round_idx in range(start_round, target):
            round_did_work = False

            for n in n_grid:
                for cond in conditions:
                    if _cell_done(data, cond, n, round_idx):
                        continue

                    round_did_work = True
                    cr = build_condition(cond, round_idx)
                    tasks = [t for t in cr.tasks if t.name in POWER_METRICS]
                    if not tasks:
                        continue

                    kwargs = dict(cr.task_kwargs)
                    kwargs["n_samples"] = n
                    result = cr.engine.run_tasks(tasks, **kwargs)

                    n_saved = 0
                    for name, res in result.items():
                        score = float(res.score)
                        _append(data, cond, name, n, score)
                        n_saved += 1

                    save_power_results(data, results_file)
                    print(f"[Round {round_idx + 1:>4}/{target}]  "
                          f"n={n:>4}  {cond:24s}  "
                          f"({n_saved} metrics)  -> {results_file.name}")

            if not round_did_work:
                print(f"\nAll {target} rounds complete for {title}.")
                break

            if (round_idx + 1) % 10 == 0:
                _print_summary(data, conditions, n_grid, round_idx + 1)

    except KeyboardInterrupt:
        print(f"\nStopped. Results saved to {results_file}")


def _print_summary(data, conditions, n_grid, rounds_done):
    print(f"\n  Summary after {rounds_done} rounds:")
    for cond in conditions:
        counts = {n: max((_count(data, cond, m, n) for m in POWER_METRICS), default=0)
                  for n in n_grid}
        row = "  ".join(f"n={n}:{counts[n]}" for n in n_grid)
        print(f"    {cond:24s}  {row}")
    print()


# CLI

def parse_power_args(argv=None):
    p = argparse.ArgumentParser(description="Power/convergence experiment")
    p.add_argument("--runs", type=int, default=None,
                   help="Number of rounds. Default: run until interrupted.")
    return p.parse_args(argv)
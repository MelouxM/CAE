"""
Shared test runner for causal-abstraction evaluation suites.

Resume / catch-up behavior:
The runner tracks per-metric run counts.  When some metrics have fewer data
points than others (e.g. because the user deleted a metric's column from the
JSON, or because a new metric was added to _build_analytical), the runner
automatically fills in only those lagging metrics until they are level with
the rest, then resumes normal all-metric operation.

The logic is simple: in round n, a metric is only run if it currently has
fewer than n+1 data points.  Metrics that already have n+1 or more are
silently skipped.  This means:

  - Always iterates from round 0 (fast no-ops for metrics that already have data).
  - Deleted/new metrics fill in during the early rounds.
  - Once all metrics are level the script behaves exactly as before.

Workflow for re-computing a metric after a bug fix:
1. Manually delete the metric's key from the relevant conditions in the JSON.
2. Re-run the experiment script (no flags needed).
   The runner fills in the missing data, stopping once the metric matches
   the existing run count.
3. On a second execution the runner just continues adding new runs as usual.

Usage examples
    python test/01_logic_circuit.py                          # run forever
    python test/01_logic_circuit.py --runs 10                # stop after 10 complete rounds
    python test/01_logic_circuit.py --metrics IIAMetric      # re-run only IIA
    python test/01_logic_circuit.py --metrics IIAMetric --runs 5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils import collect_scores, load_results, save_results, n_runs, print_table


# CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Causal abstraction evaluation suite")
    p.add_argument(
        '--metrics', nargs='+', default=None,
        help='Only (re)compute these metrics. Names can be task names '
             '(e.g. CAE_down_nf, CAE_up_nf, MSE) or analytical class names (e.g. IIAMetric, BCCMetric).',
    )
    p.add_argument(
        '--runs', type=int, default=None,
        help='Number of complete rounds. Default: match existing count when --metrics '
             'is set, otherwise run forever.',
    )
    return p.parse_args(argv)


# Condition descriptor
@dataclass
class Condition:
    """Everything the runner needs to evaluate one experimental condition."""
    engine: Any
    tasks: list
    analytical: list
    sampler: Any
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    analytical_n_samples: Optional[int] = None


# Filtering helpers
def _filter_tasks(tasks, names):
    if names is None:
        return tasks
    s = set(names)
    return [t for t in tasks if t.name in s]


def _filter_analytical(metrics, names):
    if names is None:
        return metrics
    s = set(names)
    return [m for m in metrics if type(m).__name__ in s]


def _metric_count(data: Dict, cond: str, name: str) -> int:
    """Number of existing data points for (condition, metric_name)."""
    return len(data.get(cond, {}).get(name, []))


# Main entry point
def run_suite(
    title: str,
    results_file: Path,
    conditions: List[str],
    comparisons: List[Tuple[str, str, str]],
    build_condition: Callable[[str, int], Condition],
    print_table_every: int = 10,
    args=None,
):
    """
    Run an evaluation suite with automatic per-metric catch-up.

    Args:
        title: Display title for the table.
        results_file: JSON file for incremental persistence.
        conditions: Condition names (keys in the results dict).
        comparisons: Pairs to compare in the table (cond_a, cond_b, label).
        build_condition: Factory ``(cond_name, run_index) -> Condition`` that builds
            everything for one condition + run.
        print_table_every: Print the summary table every N complete rounds.
        args: Pre-parsed CLI args. Parsed from sys.argv if None.
    """
    if args is None:
        args = parse_args()

    data = load_results(results_file, conditions)
    metric_filter = args.metrics

    existing_max = n_runs(data, conditions)

    if metric_filter:
        mf_set = set(metric_filter)
        target_n = args.runs if args.runs is not None else existing_max
        if target_n == 0:
            print("No existing runs to replace. Run without --metrics first.")
            return
        for cond in conditions:
            if cond in data:
                for mk in list(data[cond].keys()):
                    if mk in mf_set:
                        data[cond][mk] = []
        save_results(data, results_file)
        print(f"Re-running {target_n} run(s) for metrics: {metric_filter}")
        total_target = target_n
    else:
        total_target = args.runs  # None = infinite

    print(
        f"Existing runs: {existing_max}.  "
        f"{'Running until interrupted.' if total_target is None else f'Target: {total_target} rounds.'}\n"
        f"Lagging metrics (if any) will be filled in first.\n"
    )

    # Always iterate from n=0.  For metrics that already have >= n+1 data
    # points, _metric_count < n+1 is False and they are skipped silently.
    # Early rounds are therefore no-ops for already-complete metrics, and
    # lagging metrics (count=0) catch up naturally round by round.
    n = 0
    try:
        while total_target is None or n < total_target:
            any_work_done = False

            for cond in conditions:
                cr = build_condition(cond, n)

                tasks = [
                    t for t in _filter_tasks(cr.tasks, metric_filter)
                    if _metric_count(data, cond, t.name) < n + 1
                ]
                analytical = [
                    m for m in _filter_analytical(cr.analytical, metric_filter)
                    if _metric_count(data, cond, type(m).__name__) < n + 1
                ]

                if not tasks and not analytical:
                    continue

                any_work_done = True
                scores: Dict[str, float] = {}

                if tasks:
                    task_results = cr.engine.run_tasks(tasks, **cr.task_kwargs)
                    scores.update(collect_scores(task_results, {}))

                if analytical:
                    an = cr.analytical_n_samples or cr.task_kwargs.get('n_samples', 100)
                    a_results = cr.engine.run_analytical_metrics(
                        analytical, sampler=cr.sampler, n_samples=an,
                    )
                    scores.update(collect_scores({}, a_results))

                if cond not in data:
                    data[cond] = {}
                for metric, value in scores.items():
                    stored = None if (value is None or (isinstance(value, float) and np.isnan(value))) else float(value)
                    data[cond].setdefault(metric, []).append(stored)

                save_results(data, results_file)
                print(f"[Round {n + 1:>4}]  {cond:20s}  ->  {results_file.name}")

            n += 1

            if not any_work_done:
                # All metrics already had data for this round; no need to print.
                continue

            if n % print_table_every == 0:
                print_table(data, n, title, conditions, comparisons)

    except KeyboardInterrupt:
        print(f"\nStopped after {n} complete round(s). Results in {results_file}")

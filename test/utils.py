"""
Shared utilities for causal-abstraction evaluation suites.

Provides:
  load_system     - import a benchmark system (or sibling script) by file path
  cohens_d        - unpaired Cohen's d effect size
  significance    - Mann-Whitney U two-sided test with significance markers
  collect_scores  - flatten task + analytical results into a dict of floats
  load_results    - load a JSON results file (or return an empty skeleton)
  save_results    - atomically write a results dict to JSON
  n_runs          - infer the number of completed runs from the longest score list
  min_runs        - infer the minimum completed runs across all metrics/conditions
  append_run      - append one run's scores into the accumulator dict
  print_table     - pretty-print a conditions × metrics table with effect sizes

Importing this module also puts the repository root and ``systems/`` on
``sys.path`` (see :func:`load_system`), so suites no longer each repeat that
bootstrap.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# System loading
REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_paths() -> None:
    """Put the repo root and ``systems/`` on ``sys.path`` (idempotent).

    The evaluation suites run as scripts (``python test/NN_*.py``) and import both
    the in-place ``causal_abstraction`` package (at the repo root) and sibling
    benchmark modules (under ``systems/``). Importing this module performs that
    bootstrap once so the suites don't each repeat the ``sys.path.insert`` dance.
    """
    for p in (str(REPO_ROOT), str(REPO_ROOT / "systems")):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_repo_paths()


def load_system(filename: str, module_name: Optional[str] = None) -> Any:
    """Import a benchmark system (or sibling script) by file path.

    The benchmark modules use numeric prefixes (``01_logic_circuit.py`` …) that
    are not importable via normal package mechanics, so every suite loads its
    matching module by path. This centralizes the ``spec_from_file_location`` /
    ``module_from_spec`` / ``exec_module`` boilerplate (and the ``sys.path``
    bootstrap it relies on) that was duplicated across the suites.

    It is a pure import-mechanism helper: it changes how a module is loaded,
    never what the loaded module computes.

    Args:
        filename: Path to the ``.py`` file to load. A relative path (e.g.
            ``"01_logic_circuit.py"`` or ``"09_grn/grn.py"``) is resolved against
            ``systems/``; an absolute path (e.g. ``TEST_DIR / "03_gas_simulation.py"``)
            is used as given, so sibling ``test/`` modules can be loaded too.
        module_name: Name to assign the loaded module. Defaults to the file stem.

    Returns:
        The loaded, executed module object.
    """
    _ensure_repo_paths()
    path = Path(filename)
    if not path.is_absolute():
        path = REPO_ROOT / "systems" / path
    name = module_name or path.stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Statistical helpers
def cohens_d(a: List[float], b: List[float]) -> float:
    """Unpaired Cohen's d = (mean_a − mean_b) / pooled_std."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled < 1e-12:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else float('inf')
    return float((np.mean(a) - np.mean(b)) / pooled)


def significance(a: List[float], b: List[float]) -> Tuple[float, str]:
    """Mann-Whitney U (two-sided). Returns (p_value, marker)."""
    if len(a) < 2 or len(b) < 2:
        return float('nan'), "-"
    try:
        _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        p = float(p)
    except Exception as e:
        logger.debug("significance: Mann-Whitney U failed, returning n/a: %s", e)
        return float('nan'), "-"
    if p < 0.001:   marker = "***"
    elif p < 0.01:  marker = "**"
    elif p < 0.05:  marker = "*"
    else:           marker = "ns"
    return p, marker


# Score collection
def collect_scores(task_results, analytical) -> Dict[str, float]:
    """Flatten task results and analytical metrics into a plain dict of floats."""
    scores: Dict[str, float] = {}
    try:
        items = list(task_results.items())
    except TypeError:
        items = list(task_results.items)
    for name, res in items:
        try:
            scores[name] = float(res.score)
        except Exception as e:
            logger.warning("collect_scores: dropping task %r (score not float-convertible): %s", name, e)
    for name, val in analytical.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            try:
                scores[name] = float(val)
            except Exception as e:
                logger.warning("collect_scores: dropping metric %r (value not float-convertible): %s", name, e)
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        scores[f"{name}/{k}"] = float(v)
                    except Exception as e:
                        logger.warning("collect_scores: dropping metric %r (value not float-convertible): %s", f"{name}/{k}", e)
    return scores


# Persistence
def load_results(results_file: Path, conditions: List[str]) -> Dict:
    """Load existing JSON results, or return an empty per-condition skeleton."""
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            n = n_runs(data, conditions)
            print(f"Loaded {n} existing run(s) from {results_file.name}")
            return data
        except Exception as e:
            print(f"Warning: could not load {results_file.name} ({e}). Starting fresh.")
    return {cond: {} for cond in conditions}


def save_results(data: Dict, results_file: Path) -> None:
    """Atomically write results dict to JSON (write to .tmp then rename)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(results_file)


def n_runs(data: Dict, conditions: List[str]) -> int:
    """Infer number of completed runs from the longest score list."""
    best = 0
    for cond in conditions:
        for v in data.get(cond, {}).values():
            if isinstance(v, list):
                best = max(best, len(v))
    return best


def min_runs(data: Dict, conditions: List[str]) -> int:
    """Minimum run count for any metric across all conditions.

    Used by the runner to determine where to resume so that metrics with
    fewer data points (e.g. after manual deletion from the JSON) automatically
    catch up before normal progression resumes.
    """
    m: float = float('inf')
    found = False
    for cond in conditions:
        for vals in data.get(cond, {}).values():
            if isinstance(vals, list):
                m = min(m, len(vals))
                found = True
    return int(m) if found else 0


def append_run(data: Dict, run_scores: Dict) -> None:
    """Append one run's scores into the accumulator dict, skipping NaNs."""
    for cond, scores in run_scores.items():
        if cond not in data:
            data[cond] = {}
        for metric, val in scores.items():
            stored = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
            data[cond].setdefault(metric, []).append(stored)


# Pretty-printing
def print_table(
    data: Dict,
    n: int,
    title: str,
    conditions: List[str],
    comparisons: List[Tuple[str, str, str]],
    cw: int = 16,
    ew: int = 18,
) -> None:
    """Print a conditions × metrics table with Cohen's d effect sizes."""
    all_metrics: List[str] = sorted({
        m for cond in conditions for m in data.get(cond, {})
    })
    if not all_metrics:
        return

    mw  = max(max(len(m) for m in all_metrics), 8) + 1
    sep = "─" * (mw + len(conditions) * (cw + 2) + len(comparisons) * (ew + 2))

    print(f"\n{sep}")
    print(f"  {title}  |  {n} runs")
    print(sep)

    hdr = f"{'Metric':<{mw}}"
    for cond in conditions:
        hdr += f"  {cond:^{cw}}"
    for _, _, label in comparisons:
        hdr += f"  {label:^{ew}}"
    print(hdr)
    print(sep)

    for metric in all_metrics:
        row   = f"{metric:<{mw}}"
        means: Dict[str, Optional[list]] = {}

        for cond in conditions:
            vals = data.get(cond, {}).get(metric, [])
            vals = [x for x in vals if x is not None]
            if vals:
                m    = np.mean(vals)
                s    = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                cell = f"{m:.3f}±{s:.3f}"
                means[cond] = vals
            else:
                cell = "n/a"
                means[cond] = None
            row += f"  {cell:^{cw}}"

        for cond_a, cond_b, _ in comparisons:
            a_vals = means.get(cond_a)
            b_vals = means.get(cond_b)
            if (a_vals is not None and b_vals is not None
                    and len(a_vals) >= 2 and len(b_vals) >= 2):
                d      = cohens_d(a_vals, b_vals)
                _, sig = significance(a_vals, b_vals)
                d_str  = f"{d:+.2f}" if np.isfinite(d) else ("∞" if d > 0 else "-∞")
                cell   = f"d={d_str} [{sig}]"
            else:
                cell = "-"
            row += f"  {cell:^{ew}}"

        print(row)

    print(sep + "\n")
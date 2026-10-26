"""Summarize the ISIL/IC runs as markdown tables (or the LaTeX table of the paper).

Reads the results of `test/run_isil_ic.py` and `test/exact_isil_ic.py`. Per pair:
mean and sd of each metric, Cohen's d, Mann-Whitney U, the 0.10/0.15 threshold
rule and AUROC.

Usage:
    python test/summarize_isil_ic.py                 # markdown to stdout
    python test/summarize_isil_ic.py --out tables.md
    python test/summarize_isil_ic.py --latex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import cohens_d, significance

RESULTS_DIR = Path(__file__).parent / "results" / "isil_ic"

# (file stem, display title, valid condition)
PAIRS: List[Tuple[str, str, str]] = [
    ("01_logic_circuit_isil_ic_results", "01 logic circuit (2-bit adder)", "valid"),
    ("02_transistor_circuit_isil_ic_results", "02 transistor circuit (CMOS half-adder)", "valid"),
    ("03_gas_simulation_isil_ic_results", "03 gas (LJ fluid vs ideal gas / VdW)", "valid"),
    ("04_predator_prey_isil_ic_results", "04 predator-prey (ABM vs Lotka-Volterra)", "valid"),
    ("05_heat_equation_isil_ic_results", "05 heat equation 1D", "valid"),
    ("06_heat_equation_2d_isil_ic_results", "06 heat equation 2D", "valid"),
    ("07_ising_model_isil_ic_results", "07 Ising model", "valid"),
    ("08_tracr_isil_ic_results", "08 Tracr sort-rank", "valid"),
    ("09_grn_isil_ic_results", "09 GRN segment polarity (Wg/Hh/En pathway)", "valid"),
    ("10_cpu_6502_isil_ic_results", "10 MOS 6502", "valid"),
    ("11_exp1_isil_ic_results", "Exp 1 hidden confounder", "valid"),
    ("11_exp2_isil_ic_results", "Exp 2 XOR backup pathway", "valid"),
    ("11_exp3_isil_ic_results", "Exp 3 wrong intermediate", "valid"),
    ("11_exp4_isil_ic_results", "Exp 4 spurious mediator", "valid_fork"),
    ("11_exp5_isil_ic_results", "Exp 5 unreachable states", "valid"),
    ("11_exp6_isil_ic_results", "Exp 6 wrong causal direction", "valid_chain"),
]

METRIC_ORDER = ["ISIL_sup", "IC_sup", "ISIL_mean", "IC_mean",
                "ISIL_sup_coarse", "IC_sup_coarse", "ISIL_mean_coarse", "IC_mean_coarse",
                "CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"]
CAE_NAMES = ["CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"]

# Benchmark systems (pooled together); the controlled experiments get their own table
SYSTEM_STEMS = [stem for stem, *_ in PAIRS if not stem.startswith("11_")]
CONTROLLED = [("11_exp1_isil_ic_results", "1 hidden confounder", "valid", "confounder"),
              ("11_exp2_isil_ic_results", "2 XOR backup pathway", "valid", "xor_leak"),
              ("11_exp3_isil_ic_results", "3 wrong intermediate", "valid", "wrong_intermediate"),
              ("11_exp4_isil_ic_results", "4 spurious mediator", "valid_fork", "spurious_chain"),
              ("11_exp5_isil_ic_results", "5 unreachable states", "valid", "wrong_threshold"),
              ("11_exp6_isil_ic_results", "6 wrong causal direction", "valid_chain", "wrong_fork")]

VALID_MAX = 0.10    # a valid abstraction should score at most this
INVALID_MIN = 0.15  # and an invalid one at least this


def _clean(vals) -> List[float]:
    return [v for v in (vals or []) if v is not None and not (isinstance(v, float) and np.isnan(v))]


def auroc(pos: List[float], neg: List[float]) -> float:
    """P(score(invalid) > score(valid)) + 0.5 P(equal), the rank-based AUROC."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    greater = (p[:, None] > n[None, :]).sum()
    equal = (p[:, None] == n[None, :]).sum()
    return float((greater + 0.5 * equal) / (p.size * n.size))


def _metrics_of(data: Dict) -> List[str]:
    seen = {m for cond in data.values() for m in cond}
    ordered = [m for m in METRIC_ORDER if m in seen]
    return ordered + sorted(seen - set(ordered))


def load(stem: str) -> Optional[Dict]:
    """Load one ISIL/IC results file.

    Missing CAE columns are taken from the suite's results file, and several valid
    conditions (gas, 6502) are merged into one `valid` column.
    """
    path = RESULTS_DIR / f"{stem}.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    suite = RESULTS_DIR.parent / stem.replace("_isil_ic_results", "_results.json")
    if suite.exists() and not any(m in c for c in data.values() for m in CAE_NAMES):
        ref = json.load(open(suite))
        for cond, cols in data.items():
            for m in CAE_NAMES:
                if m in ref.get(cond, {}):
                    cols[m] = ref[cond][m]
    valids = [c for c in data if c.startswith("valid")]
    if len(valids) > 1:
        merged = {}
        for c in valids:
            for m, v in data.pop(c).items():
                merged.setdefault(m, []).extend(v or [])
        data = {"valid": merged, **data}
    return data


def summarize_pair(stem: str, title: str, valid_cond: str) -> Optional[Dict]:
    data = load(stem)
    if data is None:
        return None
    if valid_cond not in data:
        valid_cond = next(iter(data))
    invalid_conds = [c for c in data if c != valid_cond]

    rows = []
    for metric in _metrics_of(data):
        valid_vals = _clean(data[valid_cond].get(metric))
        entry = {"metric": metric, "valid": valid_vals, "invalid": {}}
        for cond in invalid_conds:
            entry["invalid"][cond] = _clean(data[cond].get(metric))
        rows.append(entry)
    return {"title": title, "stem": stem, "valid_cond": valid_cond,
            "invalid_conds": invalid_conds, "rows": rows}


def _cell(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{m:.3f}±{s:.3f}"


def render(summaries: List[Dict]) -> str:
    out: List[str] = []
    pooled: Dict[str, Dict[str, List[float]]] = {}
    verdicts: Dict[str, List[bool]] = {}

    for s in summaries:
        out.append(f"\n### {s['title']}\n")
        header = ["Metric", f"{s['valid_cond']} (valid)"]
        header += [f"{c}" for c in s["invalid_conds"]]
        header += ["AUROC", "discriminates"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))

        for row in s["rows"]:
            metric, valid_vals = row["metric"], row["valid"]
            all_invalid = [v for c in s["invalid_conds"] for v in row["invalid"][c]]
            cells = [metric, _cell(valid_vals)]
            for c in s["invalid_conds"]:
                vals = row["invalid"][c]
                cell = _cell(vals)
                if valid_vals and len(vals) >= 2 and len(valid_vals) >= 2:
                    d = cohens_d(vals, valid_vals)
                    _, sig = significance(vals, valid_vals)
                    d_str = f"{d:+.2f}" if np.isfinite(d) else ("inf" if d > 0 else "-inf")
                    cell += f" (d={d_str} {sig})"
                cells.append(cell)

            a = auroc(all_invalid, valid_vals)
            cells.append("n/a" if np.isnan(a) else f"{a:.3f}")

            verdict_bucket = verdicts
            ok = bool(valid_vals) and float(np.mean(valid_vals)) <= VALID_MAX and all(
                row["invalid"][c] and float(np.mean(row["invalid"][c])) >= INVALID_MIN
                for c in s["invalid_conds"]
            )
            cells.append("yes" if ok else "no")
            if verdict_bucket is not None:
                verdict_bucket.setdefault(metric, []).append(ok)

            bucket = pooled.setdefault(metric, {"valid": [], "invalid": []})
            bucket["valid"] += valid_vals
            bucket["invalid"] += all_invalid

            out.append("| " + " | ".join(cells) + " |")

    out.append("\n### Pooled over all pairs\n")
    out.append("| Metric | mean valid | mean invalid | AUROC (pooled) | pairs discriminated |")
    out.append("|---|---|---|---|---|")
    for metric in [m for m in METRIC_ORDER if m in pooled] + \
                  sorted(set(pooled) - set(METRIC_ORDER)):
        b = pooled[metric]
        a = auroc(b["invalid"], b["valid"])
        v = verdicts.get(metric, [])
        out.append(
            f"| {metric} | {_cell(b['valid'])} | {_cell(b['invalid'])} | "
            f"{'n/a' if np.isnan(a) else f'{a:.3f}'} | {sum(v)}/{len(v)} |"
        )

    return "\n".join(out)


def render_systems_pooled() -> str:
    """Pooled statistics over the benchmark systems only."""
    pooled: Dict[str, Dict[str, List[float]]] = {}
    for stem in SYSTEM_STEMS:
        data = load(stem)
        if data is None:
            continue
        valid = next(iter(data))
        metrics = _metrics_of(data)
        for metric in METRIC_ORDER:
            # Label-valued systems have no coarse columns: use the main ones
            src = metric if metric in metrics else metric.removesuffix("_coarse")
            if src not in metrics:
                continue
            b = pooled.setdefault(metric, {"valid": [], "invalid": []})
            b["valid"] += _clean(data[valid].get(src))
            for cond in data:
                if cond != valid:
                    b["invalid"] += _clean(data[cond].get(src))

    out = ["\n### Pooled over the benchmark systems\n",
           "| Metric | AUROC | mean valid | mean invalid |", "|---|---|---|---|"]
    for metric in [m for m in METRIC_ORDER if m in pooled]:
        b = pooled[metric]
        a = auroc(b["invalid"], b["valid"])
        out.append(f"| {metric} | {'n/a' if np.isnan(a) else f'{a:.3f}'} | "
                   f"{_cell(b['valid'])} | {_cell(b['invalid'])} |")
    return "\n".join(out)


def render_controlled() -> str:
    """Condensed significance table for the six controlled experiments."""
    out = ["\n### Controlled experiments 1-6 (condensed)\n",
           "Cell: mean on the valid condition -> mean on the invalid one, with Cohen's d and the "
           "Mann-Whitney marker over 30 runs.\n"]
    metrics = ["ISIL_sup", "IC_sup", "ISIL_mean", "IC_mean",
               "CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"]
    out.append("| Experiment | " + " | ".join(metrics) + " |")
    out.append("|" + "---|" * (len(metrics) + 1))
    for stem, title, valid, invalid in CONTROLLED:
        path = RESULTS_DIR / f"{stem}.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        cells = [title]
        for m in metrics:
            v, w = _clean(data[valid].get(m)), _clean(data[invalid].get(m))
            if not v or not w:
                cells.append("n/a")
                continue
            d = cohens_d(w, v)
            _, sig = significance(w, v)
            d_str = f"{d:+.1f}" if np.isfinite(d) else ("inf" if d > 0 else "-inf")
            cells.append(f"{np.mean(v):.3f}->{np.mean(w):.3f} ({d_str} {sig})")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_exact() -> str:
    """Render the enumerated results, one row per (condition, assessment family)."""
    path = RESULTS_DIR / "exact_isil_ic.json"
    if not path.exists():
        return ""
    rows = json.load(open(path))
    out = ["\n### Exact (enumerated) ISIL/IC on the finite pairs\n",
           "| Condition | family | e_ISIL | e_IC | \\|J\\| | cells | P(disagree) | draws to sup |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        draws = r["expected_draws_to_sup"]
        out.append(
            f"| {r['condition']} | {r['family']} | {r['e_isil']:.3f} | {r['e_ic']:.3f} | "
            f"{r['assessment_pairs']} | {r['cells']} | "
            f"{'-' if r['family'] != 'domain' else format(r['p_hit'], '.4g')} | "
            f"{'-' if not np.isfinite(draws) else format(draws, '.3g')} |"
        )
    return "\n".join(out)


LATEX_COLS = ["ISIL_sup", "IC_sup", "ISIL_mean", "IC_mean",
              "CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"]
LATEX_HEAD = [r"ISIL$_{\sup}$", r"IC$_{\sup}$", r"ISIL$_{\mathbb{E}}$", r"IC$_{\mathbb{E}}$",
              r"\caedownnf{}", r"\caeupnf{}", r"\caedown{}", r"\caeup{}"]


def render_latex() -> str:
    """LaTeX table: per pair, the AUROC and the mean valid score of each metric.

    Continuous systems get a second row with the 25% binning when it changes anything.
    """
    def cell(data, valid, metric):
        v = _clean(data[valid].get(metric))
        w = [x for c in data if c != valid for x in _clean(data[c].get(metric))]
        if not v or not w:
            return "--", None
        return f"{auroc(w, v):.2f} ({np.mean(v):.2f})", (v, w)

    lines = [r"\begin{tabular}{l" + "c" * len(LATEX_COLS) + "}", r"\toprule",
             "Pair & " + " & ".join(LATEX_HEAD) + r" \\", r"\midrule"]
    for block, stems in (("systems", SYSTEM_STEMS), ("controlled", [c[0] for c in CONTROLLED])):
        pooled = {m: ([], []) for m in LATEX_COLS}
        for stem, title, _valid in PAIRS:
            if stem not in stems:
                continue
            data = load(stem)
            if data is None:
                continue
            valid = next(iter(data))
            coarse = any(m.endswith("_coarse") for c in data.values() for m in c)
            rows = [("", "")] + ([(" (25\\%)", "_coarse")] if coarse else [])
            first = None
            for label, suffix in rows:
                cells = []
                for m in LATEX_COLS:
                    if suffix and m.startswith("CAE"):
                        cells.append("")  # the CAE does not depend on the quantization
                        continue
                    src = m + suffix if m.startswith(("ISIL", "IC")) else m
                    txt, vw = cell(data, valid, src)
                    cells.append(txt)
                    if vw and suffix == "":
                        pooled[m][0].extend(vw[0]); pooled[m][1].extend(vw[1])
                if suffix and cells[:4] == first[:4]:
                    continue  # same verdict at both resolutions: one row is enough
                first = first or cells
                name = title.split(" (")[0] + label
                lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        cells = [f"{auroc(w, v):.2f} ({np.mean(v):.2f})" if v and w else "--"
                 for v, w in (pooled[m] for m in LATEX_COLS)]
        lines.append(r"\midrule")
        lines.append(f"Pooled ({block}) & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarize ISIL/IC runs")
    p.add_argument("--out", type=Path, default=None, help="Write markdown here.")
    p.add_argument("--latex", action="store_true", help="Print the paper's LaTeX table instead.")
    args = p.parse_args(argv)
    if args.latex:
        print(render_latex())
        return

    summaries = [s for s in (summarize_pair(*spec) for spec in PAIRS) if s]
    text = (render(summaries) + "\n" + render_systems_pooled() + "\n"
            + render_controlled() + "\n" + render_exact() + "\n")
    if args.out:
        args.out.write_text(text)
        print(f"Written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

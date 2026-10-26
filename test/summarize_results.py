"""Summary of the suite results: per-pair means and AUROCs, and one AUROC per metric
pooled over all systems.

AUROC is P(score_invalid > score_valid) + 0.5 P(tie) over the per-run scores. Scores are
pooled as they are, without rescaling across systems.

Usage:
    python test/summarize_results.py                  # all metrics
    python test/summarize_results.py --metrics CAE_down CAE_up IIAMetric
    python test/summarize_results.py --dir test/results --markdown out.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TEST_DIR = Path(__file__).parent

# (results file, valid condition, invalid condition): the pairs the paper pools
PAIRS: List[Tuple[str, str, str]] = [
    ("01_logic_circuit_results.json", "valid", "fail"),
    ("01_logic_circuit_results.json", "valid", "inv_internal"),
    ("01_logic_circuit_results.json", "valid", "noise"),
    ("02_transistor_circuit_results.json", "valid", "fail"),
    ("02_transistor_circuit_results.json", "valid", "noise"),
    ("03_gas_simulation_results.json", "valid_ideal", "invalid_alpha"),
    ("03_gas_simulation_results.json", "valid_ideal", "invalid_density"),
    ("03_gas_simulation_results.json", "valid_vdw", "invalid_density"),
    ("03_gas_simulation_results.json", "valid_ideal", "invalid_temp"),
    ("04_predator_prey_results.json", "valid", "fail_alpha"),
    ("04_predator_prey_results.json", "valid", "spatial"),
    ("04_predator_prey_results.json", "valid", "stochastic"),
    ("04_predator_prey_results.json", "valid", "aging"),
    ("04_predator_prey_results.json", "valid", "noise"),
    ("04_predator_prey_results.json", "valid", "complex"),
    ("05_heat_equation_results.json", "valid", "fail"),
    ("05_heat_equation_results.json", "valid", "noise"),
    ("06_heat_equation_2d_results.json", "valid", "fail"),
    ("06_heat_equation_2d_results.json", "valid", "noise"),
    ("07_ising_model_results.json", "valid", "fail"),
    ("07_ising_model_results.json", "valid", "rigid"),
    ("07_ising_model_results.json", "valid", "noise"),
    ("08_tracr_results.json", "valid", "fail"),
    ("08_tracr_results.json", "valid", "noise"),
    ("09_grn_results.json", "valid", "wrong_map"),
    ("09_grn_results.json", "valid", "wrong_high_level_model"),
    ("09_grn_results.json", "valid", "noise"),
    ("10_cpu_6502_results.json", "valid_gate_isa", "broken_gate_isa"),
    ("10_cpu_6502_results.json", "valid_transistor_gate", "broken_transistor_gate"),
    ("10_cpu_6502_results.json", "valid_transistor_isa", "broken_transistor_isa"),
    ("11_exp1_hidden_confounder_results.json", "valid", "confounder"),
    ("11_exp2_xor_backup_results.json", "valid", "xor_leak"),
    ("11_exp3_wrong_intermediate_results.json", "valid", "wrong_intermediate"),
    ("11_exp4_spurious_mediator_results.json", "valid_fork", "spurious_chain"),
    ("11_exp5_unreachable_states_results.json", "valid", "wrong_threshold"),
    ("11_exp6_wrong_direction_results.json", "valid_chain", "wrong_fork"),
]


def _clean(vals) -> List[float]:
    return [float(v) for v in (vals or [])
            if v is not None and not (isinstance(v, float) and np.isnan(v))]


def auroc(invalid: List[float], valid: List[float]) -> float:
    """P(invalid > valid) + 0.5 P(tie); NaN if either side is empty."""
    if not invalid or not valid:
        return float("nan")
    inv = np.asarray(invalid)[:, None]
    val = np.asarray(valid)[None, :]
    return float(np.mean((inv > val) + 0.5 * (inv == val)))


def load(results_dir: Path) -> Dict[str, Dict]:
    out = {}
    for fname in sorted({p[0] for p in PAIRS}):
        path = results_dir / fname
        if path.exists():
            with open(path) as f:
                out[fname] = json.load(f)
    return out


def summarize(results_dir: Path, metrics=None) -> str:
    data = load(results_dir)
    all_metrics = sorted({m for d in data.values() for c in d.values() for m in c})
    metrics = metrics or all_metrics
    lines = [f"Loaded {len(data)} result files from {results_dir}", ""]

    lines.append("## Runs per condition\n")
    for fname, d in data.items():
        counts = {c: max((len(_clean(v)) for v in d[c].values()), default=0) for c in d}
        lines.append(f"- {fname}: " + ", ".join(f"{c}={n}" for c, n in counts.items()))

    lines.append("\n## Pooled AUROC (all systems, all invalid conditions)\n")
    lines.append("| Metric | pairs | mean valid | mean invalid | AUROC |")
    lines.append("|---|---|---|---|---|")
    for m in metrics:
        valid_all, invalid_all, n_pairs = [], [], 0
        for fname, vc, ic in PAIRS:
            d = data.get(fname)
            if not d or vc not in d or ic not in d:
                continue
            v, i = _clean(d[vc].get(m)), _clean(d[ic].get(m))
            if not v or not i:
                continue
            valid_all += v
            invalid_all += i
            n_pairs += 1
        if n_pairs == 0:
            continue
        a = auroc(invalid_all, valid_all)
        lines.append(f"| {m} | {n_pairs} | {np.mean(valid_all):.4g} | {np.mean(invalid_all):.4g} | {a:.3f} |")

    lines.append("\n## Per-pair means and AUROC\n")
    for m in metrics:
        rows = []
        for fname, vc, ic in PAIRS:
            d = data.get(fname)
            if not d or vc not in d or ic not in d:
                continue
            v, i = _clean(d[vc].get(m)), _clean(d[ic].get(m))
            if v and i:
                rows.append(f"| {fname.replace('_results.json', '')} | {vc} | {ic} | "
                            f"{np.mean(v):.4g} | {np.mean(i):.4g} | {auroc(i, v):.3f} |")
        if rows:
            lines.append(f"\n### {m}\n")
            lines.append("| system | valid | invalid | mean valid | mean invalid | AUROC |")
            lines.append("|---|---|---|---|---|---|")
            lines += rows
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=TEST_DIR / "results")
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--markdown", type=Path, default=None, help="Also write the summary here.")
    args = p.parse_args(argv)
    text = summarize(args.dir, args.metrics)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)


if __name__ == "__main__":
    main()

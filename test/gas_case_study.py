"""Gas case study: CAE_down of the ideal gas law and the Van der Waals equation in the
three causal directions, without equilibration, and along density, temperature and
exponent sweeps.

Uses the gas model and VdW fit of test/03_gas_simulation.py. Each point is one CAE_down
evaluation of N_SAMPLES interventions. Runs are appended to the results file and resumed.

Usage (from repo root):
    python test/gas_case_study.py --part directions --runs 5
    python test/gas_case_study.py --part density --runs 5
    python test/gas_case_study.py --part plots          # figures + tables from stored runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
from utils import load_system  # noqa: E402

_suite = load_system(TEST_DIR / "03_gas_simulation.py", "gas_suite")
_gs = _suite._gs

from causal_abstraction import (  # noqa: E402
    CausalGraph, ContinuousValueMap, EvaluationConfig, EvaluationEngine, FullSubspace, RectSubspace,
)

RESULTS_FILE = TEST_DIR / "results" / "gas_case_study.json"
FIG_DIR = TEST_DIR / "figures"
N = _suite.N_PARTICLES
N_SAMPLES = _suite.N_SAMPLES
T_BOYLE = _gs.T_BOYLE
A_VDW, B_VDW = _suite._A_VDW, _suite._B_VDW
DILUTE_RHO = 0.05
WIDTH = 0.30  # +-30% intervention domains, as in the suite
# Relative squared error, since V ~ 10^3 and P ~ 0.1
METRIC = "l2"

DENSITIES = np.round(np.linspace(0.05, 0.85, 17), 4)
TEMPS = np.round(np.linspace(0.5, 5.0, 19), 4)
TEMP_SWEEP_RHO = 0.1
ALPHAS = np.round(np.linspace(0.5, 1.5, 21), 4)
NONEQ_STEPS = (0, 500)  # (equilibration, measurement) steps out of equilibrium; default is (5000, 5000)

# Causal directions: (name, inputs, output, index into get_*_laws)
DIRECTIONS = [("NVT", ["volume", "temperature"], "pressure", 0),
              ("NPT", ["pressure", "temperature"], "volume", 1),
              ("PVT", ["pressure", "volume"], "temperature", 2)]


def _domains(rho, t):
    v = N / rho
    p = _gs.get_vdw_laws(A_VDW, B_VDW, N)[0](t, v)
    box = lambda c: RectSubspace((c * (1 - WIDTH), c * (1 + WIDTH)))
    return {"volume": box(v), "temperature": box(t), "pressure": box(p)}


def _cae_down(law, inputs, output, rho, t, seed, low_level_model=None):
    doms = _domains(rho, t)
    vm = ContinuousValueMap(_suite._CG, {k: {0: d} for k, d in doms.items()})
    g = CausalGraph()
    for name in inputs:
        g.add_variable(name, lambda: None, domain=doms[name])
    g.add_variable(output, law, parents=inputs, domain=FullSubspace(1))
    llm = low_level_model or _gs.UniversalGasModel(n_particles=N)
    engine = EvaluationEngine(g, llm, vm, _suite._CG, EvaluationConfig(metric=METRIC, seed=seed))
    res = engine.run_experiment(_gs.ThermodynamicSampler(vm, n_particles=N), N_SAMPLES, batch_size=1,
                                max_interventions=2, intervention_domain=inputs)
    return float(res.score)


def _ideal_alpha(alpha):
    return lambda temperature, volume: N * float(temperature) ** alpha / float(volume)


def points(part):
    """Yield (key, thunk(seed)) for every evaluation point of a part."""
    igl, vdw = _gs.get_ideal_laws(N), _gs.get_vdw_laws(A_VDW, B_VDW, N)
    if part in ("directions", "noneq"):
        llm = (_gs.UniversalGasModel(equil_steps=NONEQ_STEPS[0], measure_steps=NONEQ_STEPS[1], n_particles=N)
               if part == "noneq" else None)
        for d, inputs, output, idx in DIRECTIONS:
            for model, laws in (("IGL", igl), ("VdW", vdw)):
                yield (f"{part}/{d}/{model}",
                       lambda s, f=laws[idx], i=inputs, o=output: _cae_down(f, i, o, DILUTE_RHO, T_BOYLE, s, llm))
    elif part == "density":
        for rho in DENSITIES:
            for model, laws in (("IGL", igl), ("VdW", vdw)):
                yield (f"density/{rho}/{model}",
                       lambda s, f=laws[0], r=rho: _cae_down(f, ["volume", "temperature"], "pressure", r, T_BOYLE, s))
    elif part == "temperature":
        for t in TEMPS:
            for model, laws in (("IGL", igl), ("VdW", vdw)):
                yield (f"temperature/{t}/{model}",
                       lambda s, f=laws[0], tt=t: _cae_down(f, ["volume", "temperature"], "pressure",
                                                            TEMP_SWEEP_RHO, tt, s))
    elif part == "alpha":
        for a in ALPHAS:
            yield (f"alpha/{a}/IGL",
                   lambda s, aa=a: _cae_down(_ideal_alpha(aa), ["volume", "temperature"], "pressure",
                                             DILUTE_RHO, T_BOYLE, s))
    else:
        raise ValueError(part)


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run(part, runs, path):
    pts = list(points(part))
    for r in range(runs):
        for k, (key, thunk) in enumerate(pts):
            data = _load(path)  # re-read: several parts may write to separate files
            if len(data.get(key, [])) > r:
                continue
            score = thunk(10_000 * r + k)
            data = _load(path)
            data.setdefault(key, []).append(score)
            path.write_text(json.dumps(data, indent=1))
            print(f"[run {r + 1}/{runs}] {key}: {score:.4f}", flush=True)


def _series(data, part, model, xs):
    m, s, ok = [], [], []
    for x in xs:
        v = data.get(f"{part}/{x}/{model}", [])
        ok.append(bool(v))
        m.append(np.mean(v) if v else np.nan)
        s.append(np.std(v) if v else np.nan)
    return np.asarray(m), np.asarray(s)


def plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(fname, part, xs, models, xlabel, vline=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        style = {"IGL": dict(color="tab:blue", marker="o", ls="--", label="Ideal gas law"),
                 "VdW": dict(color="tab:orange", marker="s", lw=2, label="Van der Waals")}
        n_runs = 0
        for model in models:
            mu, sd = _series(data, part, model, xs)
            n_runs = max(n_runs, max(len(data.get(f"{part}/{x}/{model}", [])) for x in xs))
            ax.plot(xs, mu, **style[model])
            ax.fill_between(xs, mu - sd, mu + sd, color=style[model]["color"], alpha=0.2)
        if vline is not None:
            ax.axvline(vline, color="green", ls="--", lw=1)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(r"CAE$_\downarrow$", fontsize=14)
        ax.set_title(f"{n_runs} runs", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="best", fontsize=12)
        ax.tick_params(labelsize=12)
        fig.savefig(FIG_DIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("saved", FIG_DIR / fname)

    panel("gas_density_curve.png", "density", DENSITIES, ["IGL", "VdW"], r"Density $\rho$ (reduced units)")
    panel("gas_alpha_curve.png", "alpha", ALPHAS, ["IGL"], r"Exponent $\alpha$", vline=1.0)
    panel("gas_temp_curve.png", "temperature", TEMPS, ["IGL", "VdW"], r"Temperature $T$ (reduced units)",
          vline=T_BOYLE)

    for part in ("directions", "noneq"):
        print(f"\n{part}: CAE_down mean ± sd (runs)")
        for model in ("IGL", "VdW"):
            cells = []
            for d, *_ in DIRECTIONS:
                v = data.get(f"{part}/{d}/{model}", [])
                cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f} ({len(v)})" if v else "-")
            print(f"  {model}: " + " | ".join(cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--part", required=True,
                   choices=["directions", "noneq", "density", "temperature", "alpha", "plots"])
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args(argv)
    print(f"VdW: a={A_VDW:.4f}, b={B_VDW:.4f}; N={N}, {N_SAMPLES} samples per evaluation")
    if args.part == "plots":
        data = {}
        for f in sorted(RESULTS_FILE.parent.glob("gas_case_study*.json")):
            data.update(_load(f))
        plots(data)
    else:
        run(args.part, args.runs, RESULTS_FILE.with_name(f"gas_case_study_{args.part}.json"))


if __name__ == "__main__":
    main()

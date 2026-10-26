"""Exact ISIL/IC errors, by enumeration, on the systems with finite macro-domains.

Covers the logic circuit (2-bit adder) and controlled experiments 2 and 5. For each
assessment pair (X', Y') and each cell of X', the two paths give distributions over
Y' and we take sqrt(JSD) between them; ISIL and IC are the max over everything.
Only Y' = complement(X') is enumerated, since a larger Y' can only increase JSD.

Two assessment families are reported:
- `domain`: X' within the suite's intervention domain, with every root fixed. This
  is what the sampled ISIL/IC columns estimate.
- `full`: every X', with the free roots marginalized.

`p_hit` is the probability that one draw of the suite's sampler hits a disagreeing
cell, so `1/p_hit` is the expected number of draws before the sampled max reaches
the exact value.

Usage:
    python test/exact_isil_ic.py                    # all three systems
    python test/exact_isil_ic.py --system 01
    python test/exact_isil_ic.py --json out.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from utils import load_system
from causal_abstraction import EvaluationConfig, EvaluationEngine

TEST_DIR = Path(__file__).parent
_RNG = np.random.default_rng(0)


# System description


@dataclass
class FiniteSystem:
    """One finite abstraction pair, ready for exhaustive enumeration.

    Attributes:
        name: Condition label used in the report.
        engine: A configured engine (we only use its paths).
        domains: `{macro_variable: [labels]}` - the finite domain of each variable.
        root_probs: `{root: {label: probability}}` - the exogenous distribution
            used when a root is marginalized rather than intervened on.
        max_interventions: The suite's cap on simultaneous interventions, used to
            reproduce the sampler's cell distribution.
        intervention_domain: The suite's intervention domain (None = all variables).
        micro_support: Optional `{variable: {label: [micro states]}}`, the finite
            micro states of each cell. Without it, cells are sampled.
    """
    name: str
    engine: Any
    domains: Dict[str, List[Any]]
    root_probs: Dict[str, Dict[Any, float]]
    max_interventions: int
    intervention_domain: Optional[List[str]] = None
    micro_support: Optional[Dict[str, Dict[Any, List[Any]]]] = None
    _cache: Dict[Tuple, Tuple] = field(default_factory=dict, repr=False)

    @property
    def variables(self) -> List[str]:
        return list(self.engine.high_level_model.variables)

    @property
    def roots(self) -> List[str]:
        return [v.name for v in self.engine.high_level_model.get_roots()]

    def preimage(self, var: str, label: Any, n_draws: int) -> List[np.ndarray]:
        """The micro states of one (variable, label) cell.

        Uses `micro_support` when given, otherwise `n_draws` uniform draws from the cell.
        """
        declared = (self.micro_support or {}).get(var, {}).get(label)
        if declared is not None:
            return [np.asarray(v, dtype=float).ravel() for v in declared]
        subspace = self.engine.value_map.specs[var][label]
        rng = np.random.default_rng(zlib.crc32(f"{var}/{label}".encode()))
        return [subspace.sample((1, subspace.dim), rng=rng).squeeze(0) for _ in range(n_draws)]

    def micro_grid(self, assignment: Dict[str, Any], n_draws: int, cap: int = 64
                   ) -> List[Dict[str, np.ndarray]]:
        """Joint micro representatives of an intervened cell (product of preimages)."""
        names = sorted(assignment)
        per_var = [self.preimage(v, assignment[v], n_draws) for v in names]
        sizes = [len(x) for x in per_var]
        total = 1
        for k in sizes:
            total *= k
        if total <= cap:
            return [dict(zip(names, combo)) for combo in itertools.product(*per_var)]
        # Too many combinations: pair the i-th representative of each variable
        n = min(max(sizes), cap)
        return [{v: per_var[i][j % sizes[i]] for i, v in enumerate(names)} for j in range(n)]

    def evaluate(self, assignment: Dict[str, Any], n_draws: int
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run both paths under `do(assignment)`, once per micro representative.

        The low-level path gets each micro state of the cell explicitly.

        Returns:
            `(high_level_output, [low_level_output per micro representative])`.
        """
        key = (n_draws,) + tuple(sorted((k, _hashable(v)) for k, v in assignment.items()))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        builder = self.engine.builder
        rng = np.random.default_rng(zlib.crc32(repr(key).encode()))
        label_spec = {k: {'labels': [v], 'micro_values': None} for k, v in assignment.items()}
        out_a = _flatten(builder.build_path_standard_high_level_model().execute(label_spec, rng=rng))

        path_b = builder.build_path_standard_low_level_model()
        outs_b = []
        for micro in self.micro_grid(assignment, n_draws):
            spec = {k: {'labels': [assignment[k]],
                        'micro_values': np.asarray(micro[k], dtype=float)[None, :]}
                    for k in assignment}
            outs_b.append(_flatten(path_b.execute(spec, rng=rng)))

        result = (out_a, outs_b)
        self._cache[key] = result
        return result


def _hashable(v):
    """Turn a macro label into a float or a tuple of floats, so both paths compare equal."""
    try:
        arr = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return str(v)
    if arr.size == 1:
        return float(arr[0])
    return tuple(float(x) for x in arr)


def _flatten(out: Dict[str, Any]) -> Dict[str, Any]:
    """Take the single batch element of each variable's path output."""
    flat = {}
    for k, v in out.items():
        if isinstance(v, list):
            flat[k] = _hashable(v[0]) if v else None
        else:
            flat[k] = _hashable(v)
    return flat


# Enumeration


def _sqrt_jsd(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    support = sorted(set(p) | set(q), key=repr)
    pv = np.array([p.get(k, 0.0) for k in support], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in support], dtype=float)
    pv /= pv.sum()
    qv /= qv.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        d = jensenshannon(pv, qv, base=2.0)
    if np.isnan(d):
        return 0.0 if np.allclose(pv, qv) else 1.0
    return float(np.clip(d, 0.0, 1.0))


def _subset_prob(system: FiniteSystem, subset: Tuple[str, ...]) -> float:
    """Probability that the suite's sampler selects exactly `subset` as targets."""
    pool = system.intervention_domain or system.variables
    if any(v not in pool for v in subset):
        return 0.0
    limit = min(len(pool), system.max_interventions)
    k = len(subset)
    if k < 1 or k > limit:
        return 0.0
    return (1.0 / limit) / math.comb(len(pool), k)


def analyze(system: FiniteSystem, family: str = "domain", n_draws: int = 32) -> Dict[str, Any]:
    """Enumerate one finite system and return e_ISIL and e_IC.

    For ISIL, the low-level side of a cell is the mixture over the cell's micro
    states. For IC, each micro state is compared on its own and the worst is kept.

    Args:
        system: The finite abstraction pair.
        family: `'domain'` or `'full'` (see the module docstring).
        n_draws: Draws per cell when there is no `micro_support`.

    Returns:
        A dict with `e_isil`, `e_ic`, `p_hit` and some bookkeeping.
    """
    variables = system.variables
    roots = system.roots
    pool = system.intervention_domain or variables
    restricted = (family == "domain")

    e_isil = e_ic = 0.0
    p_hit = 0.0
    n_cells = 0
    worst_isil: Optional[Dict[str, Any]] = None
    worst_ic: Optional[Dict[str, Any]] = None

    x_pool = pool if restricted else variables
    for size in range(1, len(x_pool) + 1):
        for x_sub in itertools.combinations(x_pool, size):
            # 'domain' fixes the roots too; 'full' marginalizes them
            x_vars = tuple(dict.fromkeys(x_sub + (tuple(roots) if restricted else ())))
            y_vars = [v for v in variables if v not in x_vars]
            if not y_vars:
                continue
            free_roots = [r for r in roots if r not in x_vars]
            sel_prob = _subset_prob(system, x_sub)

            for x_vals in itertools.product(*(system.domains[v] for v in x_vars)):
                assignment_x = dict(zip(x_vars, x_vals))
                p_dist: Dict[Any, float] = {}
                q_dist: Dict[Any, float] = {}
                cell_ic = 0.0
                n_disagree = n_reps = 0

                for u_vals in itertools.product(*([list(system.root_probs[r]) for r in free_roots])):
                    w = 1.0
                    for r, val in zip(free_roots, u_vals):
                        w *= system.root_probs[r][val]
                    assignment = {**assignment_x, **dict(zip(free_roots, u_vals))}
                    out_a, outs_b = system.evaluate(assignment, n_draws)
                    y_a = tuple(out_a.get(v) for v in y_vars)
                    p_dist[y_a] = p_dist.get(y_a, 0.0) + w

                    for out_b in outs_b:
                        y_b = tuple(out_b.get(v) for v in y_vars)
                        q_dist[y_b] = q_dist.get(y_b, 0.0) + w / len(outs_b)
                        n_reps += 1
                        if y_a != y_b:
                            n_disagree += 1
                            cell_ic = 1.0

                d_isil = _sqrt_jsd(p_dist, q_dist)
                n_cells += 1
                if d_isil > e_isil:
                    e_isil = d_isil
                    worst_isil = {"X": list(x_vars), "x": list(map(_jsonable, x_vals)),
                                  "Y": y_vars, "value": d_isil}
                if cell_ic > e_ic:
                    e_ic = cell_ic
                    worst_ic = {"X": list(x_vars), "x": list(map(_jsonable, x_vals)),
                                "Y": y_vars, "value": cell_ic}

                if n_disagree and restricted:
                    prob = sel_prob
                    for v, val in assignment_x.items():
                        probs = system.root_probs.get(v)
                        prob *= (probs.get(val, 0.0) if probs else 1.0 / len(system.domains[v]))
                    p_hit += prob * (n_disagree / max(n_reps, 1))

    n_vars = len(variables)
    return {
        "condition": system.name,
        "family": family,
        "n_macro_vars": n_vars,
        "assessment_pairs": 3 ** n_vars - 2 * 2 ** n_vars + 1,
        "cells": n_cells,
        "e_isil": round(e_isil, 6),
        "e_ic": round(e_ic, 6),
        "p_hit": p_hit,
        "expected_draws_to_sup": (1.0 / p_hit) if p_hit > 0 else float("inf"),
        "finite_micro_support": system.micro_support is not None,
        "worst_isil": worst_isil,
        "worst_ic": worst_ic,
    }


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# System builders


def _engine(hlm, llm, vm, cg, metric="mse"):
    return EvaluationEngine(hlm, llm, vm, cg, EvaluationConfig(metric=metric, seed=0))


def build_logic_circuit() -> List[FiniteSystem]:
    from causal_abstraction import MicroVariableSchema
    lc = load_system("01_logic_circuit.py", "logic_circuit")
    gates, wires = lc.build_2bit_adder()
    schema = MicroVariableSchema.from_names(wires)
    llm = lc.NetlistSimulator(gates, wires)
    cg, vm = lc.build_cg_and_vm(schema)

    domains = {"Operand_A": [0, 1, 2, 3], "Operand_B": [0, 1, 2, 3], "Carry_In": [0, 1],
               "Internal_Carries": [0, 1], "Result_Sum": [0, 1, 2, 3], "Result_Carry": [0, 1]}
    # Wire voltages are thresholded at 0.5, so each label's finite micro state is its
    # bit pattern; the RectSubspace around it only models measurement tolerance
    bits = {1: {0: [[0.0]], 1: [[1.0]]},
            2: {v: [[float(v & 1), float((v >> 1) & 1)]] for v in range(4)}}
    micro_support = {"Operand_A": bits[2], "Operand_B": bits[2], "Result_Sum": bits[2],
                     "Carry_In": bits[1], "Internal_Carries": bits[1], "Result_Carry": bits[1]}
    root_probs = {"Operand_A": {i: 0.25 for i in range(4)},
                  "Operand_B": {i: 0.25 for i in range(4)},
                  "Carry_In": {0: 0.5, 1: 0.5}}

    return [
        FiniteSystem(f"01_logic_circuit/{name}", _engine(hlm, llm, vm, cg),
                     domains, root_probs, max_interventions=2, micro_support=micro_support)
        for name, hlm in (("valid", lc.build_valid_high_level_model()),
                          ("fail", lc.build_failing_high_level_model()),
                          ("inv_internal", lc.build_inverted_internal_high_level_model()))
    ]


def build_exp2() -> List[FiniteSystem]:
    ctrl = load_system("11_controlled_experiments.py", "controlled")
    _Model = ctrl.XORBackupLowLevelModel
    schema = ctrl.build_xor_schema()
    cg = ctrl.build_xor_cg(schema)
    vm = ctrl.build_xor_vm(cg)
    hlm = ctrl.build_xor_high_level_model()
    domains = {"X": [0, 1], "M": [0, 1], "Y": [0, 1]}
    root_probs = {"X": {0: 0.5, 1: 0.5}}
    micro_support = {v: {0: [[0.0]], 1: [[1.0]]} for v in ("X", "M", "Y")}
    return [
        FiniteSystem(f"11_exp2/{name}",
                     _engine(hlm, _Model(xor_active=(name == "xor_leak")), vm, cg),
                     domains, root_probs, max_interventions=4, micro_support=micro_support)
        for name in ("valid", "xor_leak")
    ]


def build_exp5() -> List[FiniteSystem]:
    ctrl = load_system("11_controlled_experiments.py", "controlled")
    schema = ctrl.build_threshold_schema()
    cg = ctrl.build_threshold_cg(schema)
    vm = ctrl.build_threshold_vm(cg)
    llm = ctrl.ThresholdLowLevelModel()
    domains = {"X": [0, 1], "M": [0, 1, 2, 3], "Z": [0, 1]}
    root_probs = {"X": {0: 0.5, 1: 0.5}}
    # x, m, z are declared int-valued in the schema; the unit-wide cells are the
    # library's tolerance around those integers
    micro_support = {"X": {0: [[0.0]], 1: [[1.0]]},
                     "M": {v: [[float(v)]] for v in range(4)},
                     "Z": {0: [[0.0]], 1: [[1.0]]}}
    specs = [("valid", ctrl.build_valid_threshold_high_level_model()),
             ("wrong_threshold", ctrl.build_wrong_threshold_high_level_model())]
    return [
        FiniteSystem(f"11_exp5/{name}", _engine(hlm, llm, vm, cg), domains, root_probs,
                     max_interventions=2, intervention_domain=["M"],
                     micro_support=micro_support)
        for name, hlm in specs
    ]


BUILDERS: Dict[str, Callable[[], List[FiniteSystem]]] = {
    "01": build_logic_circuit,
    "exp2": build_exp2,
    "exp5": build_exp5,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Exact ISIL/IC on the finite systems")
    p.add_argument("--system", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    p.add_argument("--n-ground", type=int, default=32,
                   help="Preimage draws per cell when grounding is not invariant.")
    p.add_argument("--json", type=Path,
                   default=TEST_DIR / "results" / "isil_ic" / "exact_isil_ic.json")
    args = p.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    print(f"{'condition':<34} {'family':<7} {'e_ISIL':>7} {'e_IC':>6} {'|J|':>5} "
          f"{'cells':>7} {'p_hit':>8} {'draws':>7}")
    for key in args.system:
        for system in BUILDERS[key]():
            for family in ("domain", "full"):
                row = analyze(system, family=family, n_draws=args.n_ground)
                rows.append(row)
                draws = row["expected_draws_to_sup"]
                print(f"{row['condition']:<34} {family:<7} {row['e_isil']:>7.3f} "
                      f"{row['e_ic']:>6.3f} {row['assessment_pairs']:>5} {row['cells']:>7} "
                      f"{row['p_hit']:>8.4g} "
                      f"{'-' if not np.isfinite(draws) else format(draws, '.3g'):>7}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(rows, f, indent=2, default=_jsonable)
    print(f"\nWritten to {args.json}")


if __name__ == "__main__":
    main()

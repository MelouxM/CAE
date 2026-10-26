"""
Example 9: Segment polarity gene regulatory network.

Is the textbook Wingless/Hedgehog/Engrailed maintenance loop a valid abstraction of
the multi-cellular logical model of Sanchez, Chaouiya & Thieffry (2008)?

Low-level model:
- The GINsim model `regulatoryGraph.ginml` (same as SP_6cells.zginml): 6 cells x 12
  multi-valued components (Wg, Fz, Dsh, Slp, Nkd, En, Hh, Ci, Ciact, Cirep, Pka, Ptc).
- The pair-rule input (Wg and En per cell) is held for `HOLD_STEPS` synchronous steps,
  then released, and the network runs to its attractor. A knockout clamps a gene to 0
  in every cell.
- Micro-variables: 12 pair-rule inputs, 5 knockout switches, 72 network nodes.
- Logical semantics: a node's basevalue is the parameter of the empty set of active
  interactions; other unlisted sets have parameter 0 (GINsim's default).

High-level model:
- Per cell, steady-state Wg (W_i) and En (E_i) as a function of prepattern and genotype
  (`reduced_steady_state`): En is kept by Wg signaling from the same or a neighboring
  cell, Wg is kept by Hh from a neighboring En cell and repressed by En. It matches the
  full network on all 1024 inputs (see `main`).

Abstraction map:
- Prepattern -> the 12 pair-rule inputs; KO_<gene> -> its knockout switch;
  W_i -> Wg_i >= 1; E_i -> En_i = 1. All other nodes are internal.

Macro domain:
- 32 prepatterns (a Wg stripe next to an En stripe, 1-2 cells each) x any combination
  of wg-, en-, hh-, ci-, dsh- knockouts. ptc- is left out: the full network then
  oscillates with ectopic En, which the reduced model doesn't capture.

Conditions (see test/09_grn.py):
- valid, wrong_map (cell i read from cell 7 - i), wrong_high_level_model
  (cell-autonomous, no signaling), noise.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from causal_abstraction import (
    CausalGraph, LowLevelModel, SystemState,
    MicroVariableSchema, CoarseGrainingMap, ValueMap,
    Variable, RectSubspace,
)
from causal_abstraction.primitives import UNMAPPED


# GINsim XML parser

@dataclass
class Edge:
    """A regulatory edge (interaction) between two nodes."""
    id: str
    source: str
    target: str
    minvalue: int  # Source must be >= this for edge to be "active"
    maxvalue: Optional[int]  # Source must be <= this (None = no upper bound)
    sign: str  # "positive" or "negative" (informational only)


@dataclass
class Node:
    """A node (variable) in the regulatory network."""
    id: str
    name: str  # Short name (e.g., "Wg") without cell suffix
    basevalue: int  # Default value when no parameter matches
    maxvalue: int  # Maximum value this variable can take
    parameters: List[Tuple[Set[str], int]]  # [(set of active edge IDs, output value), ...]


def parse_ginml(filepath: str) -> Tuple[List[str], Dict[str, Node], Dict[str, Edge]]:
    """
    Parse a GINsim .ginml file into nodes and edges.

    Returns:
        node_order: list of node IDs in canonical order
        nodes: dict of node_id -> Node
        edges: dict of edge_id -> Edge
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    graph = root.find('graph')

    # Get canonical node ordering
    node_order = graph.attrib['nodeorder'].split()

    # Parse edges first (we need them to interpret node parameters)
    edges = {}
    for edge_elem in graph.findall('edge'):
        eid = edge_elem.attrib['id']
        maxval_str = edge_elem.attrib.get('maxvalue', None)
        edges[eid] = Edge(
            id=eid,
            source=edge_elem.attrib['from'],
            target=edge_elem.attrib['to'],
            minvalue=int(edge_elem.attrib['minvalue']),
            maxvalue=int(maxval_str) if maxval_str is not None else None,
            sign=edge_elem.attrib['sign']
        )

    # Parse nodes
    nodes = {}
    for node_elem in graph.findall('node'):
        nid = node_elem.attrib['id']
        params = []
        for param_elem in node_elem.findall('parameter'):
            active_ids = set(param_elem.attrib['idActiveInteractions'].split())
            val = int(param_elem.attrib['val'])
            params.append((active_ids, val))

        nodes[nid] = Node(
            id=nid,
            name=node_elem.attrib.get('name', nid),
            basevalue=int(node_elem.attrib.get('basevalue', '0')),
            maxvalue=int(node_elem.attrib.get('maxvalue', '1')),
            parameters=params
        )

    return node_order, nodes, edges


# Multi-valued Boolean network simulator

class SegmentPolarityModel:
    """
    Simulator for the 6-cell segment polarity network.

    This is a multi-valued logical model. Each variable takes discrete values
    in {0, ..., maxvalue}. The update rule for each variable works as follows:

    1. Look at all incoming edges to this variable
    2. Determine which edges are active (source value in [minvalue, maxvalue])
    3. Find the parameter whose set of active edges exactly matches
    4. If found, the variable's next value = that parameter's value
    5. If no parameter matches: next value = basevalue when no incoming interaction is
       active (basevalue is the logical parameter K(empty set)), and 0 otherwise (GINsim's
       default for every logical parameter not listed in the file)

    This is the standard GINsim multi-valued logical semantics.
    """

    def __init__(self, node_order, nodes, edges):
        self.node_order = node_order  # Canonical variable ordering
        self.nodes = nodes
        self.edges = edges
        self.n_vars = len(node_order)

        # Precompute: for each node, which edges target it?
        self.incoming_edges = defaultdict(list)
        for eid, edge in edges.items():
            self.incoming_edges[edge.target].append(edge)

        # Index mapping for fast array access
        self.var_to_idx = {nid: i for i, nid in enumerate(node_order)}

    @classmethod
    def from_ginml(cls, ginml_path: str):
        """Create model from a GINsim .ginml file."""
        node_order, nodes, edges = parse_ginml(ginml_path)
        return cls(node_order, nodes, edges)

    def state_to_array(self, state_dict: Dict[str, int]) -> np.ndarray:
        """Convert a {node_id: value} dict to a numpy array."""
        arr = np.zeros(self.n_vars, dtype=np.int32)
        for nid, val in state_dict.items():
            if nid in self.var_to_idx:
                arr[self.var_to_idx[nid]] = val
        return arr

    def _get_active_edges(self, state: np.ndarray, target_node: str) -> Set[str]:
        """Determine which incoming edges to target_node are active given current state."""
        active = set()
        for edge in self.incoming_edges[target_node]:
            src_idx = self.var_to_idx[edge.source]
            src_val = state[src_idx]
            # Edge is active if source value >= minvalue (and <= maxvalue if specified)
            if src_val >= edge.minvalue:
                if edge.maxvalue is None or src_val <= edge.maxvalue:
                    active.add(edge.id)
        return active

    def compute_target_value(self, state: np.ndarray, node_id: str) -> int:
        """Compute what value a node should take given the current state."""
        node = self.nodes[node_id]
        active_edges = self._get_active_edges(state, node_id)

        # Find matching parameter (exact match of active edge sets)
        for param_active_set, param_val in node.parameters:
            if param_active_set == active_edges:
                return param_val

        # No match: basevalue for the empty set of active interactions, else 0 (GINsim)
        return node.basevalue if not active_edges else 0

    def synchronous_step(self, state: np.ndarray) -> np.ndarray:
        """Compute one synchronous update step (all variables update simultaneously)."""
        new_state = np.zeros_like(state)
        for i, nid in enumerate(self.node_order):
            new_state[i] = self.compute_target_value(state, nid)

        return new_state

# Experimental protocol and macro domain

N_CELLS = 6
HOLD_STEPS = 10      # pair-rule input held for this many synchronous steps, then released
MAX_STEPS = 200      # cap on the release phase (fixed points are reached in < 20 steps)
KO_GENES = ("Wg", "En", "Hh", "Ci", "Dsh")


def stripe_prepatterns() -> List[Tuple[int, ...]]:
    """Pair-rule prepatterns: one contiguous Wg stripe abutting one contiguous En stripe.

    Per cell: 0 = neither, 1 = Wg, 2 = En. Stripes are 1 or 2 cells wide, in either order
    (Wg anterior or posterior to En), anywhere in the 6-cell row: 32 patterns. The wild
    type of Sanchez et al. (2008) (Wg in cells 2-3, En in cells 4-5) is one of them.
    """
    out = set()
    for w in (1, 2):
        for e in (1, 2):
            for start in range(0, N_CELLS - (w + e) + 1):
                for block in ([1] * w + [2] * e, [2] * e + [1] * w):
                    lab = [0] * N_CELLS
                    for k, v in enumerate(block):
                        lab[start + k] = v
                    out.add(tuple(lab))
    return sorted(out)


PREPATTERNS = stripe_prepatterns()


class SegmentPolarityLowLevelModel(LowLevelModel):
    """
    The full 72-node segment polarity network (6 cells x 12 components) under the
    pair-rule protocol of Sanchez et al. (2008).

    The pair-rule input (Wg and En levels per cell) is held for `HOLD_STEPS`
    synchronous steps and then released; the network then runs to its attractor. Gene
    knockouts clamp that gene to 0 in every cell throughout (GINsim mutant semantics).

    Micro-variables:
      Inputs : `pp_Wg{i}` (0-2) and `pp_En{i}` (0-1), the pair-rule input of cell i;
               `ko_{gene}` (0/1) for gene in `KO_GENES`.
      Outputs: every network node (`Wg1` ... `Ptc6`) at the attractor. Fixed points
               are reached on the whole macro domain; if a cycle is reached instead, the
               output is the mean over the cycle.
    """

    INPUTS = ([f"pp_Wg{i}" for i in range(1, N_CELLS + 1)]
              + [f"pp_En{i}" for i in range(1, N_CELLS + 1)]
              + [f"ko_{g}" for g in KO_GENES])

    def __init__(self, model: SegmentPolarityModel):
        self.model = model
        self._cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def attractor_mean(self, pp_wg, pp_en, kos) -> np.ndarray:
        """Mean node values over the attractor reached by the protocol (cached)."""
        key = tuple(pp_wg) + tuple(pp_en) + tuple(kos)
        if key in self._cache:
            return self._cache[key]
        idx = self.model.var_to_idx
        clamps = [idx[f"{g}{i}"] for g, ko in zip(KO_GENES, kos) if ko
                  for i in range(1, N_CELLS + 1)]
        hold = ([(idx[f"Wg{i + 1}"], pp_wg[i]) for i in range(N_CELLS)]
                + [(idx[f"En{i + 1}"], pp_en[i]) for i in range(N_CELLS)])

        def clamp(s):
            s[clamps] = 0
            return s

        state = clamp(np.zeros(self.model.n_vars, dtype=np.int32))
        for _ in range(HOLD_STEPS):
            for j, v in hold:
                state[j] = v
            state = clamp(self.model.synchronous_step(clamp(state)))
        seen, traj = {}, []
        for k in range(MAX_STEPS):
            b = state.tobytes()
            if b in seen:
                break
            seen[b] = k
            traj.append(state)
            state = clamp(self.model.synchronous_step(state))
        cycle = traj[seen[state.tobytes()]:] if state.tobytes() in seen else traj[-1:]
        out = np.mean(np.stack(cycle), axis=0)
        self._cache[key] = out
        return out

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        def col(name, default):
            raw = interventions.get(name, input_state.get(name))
            if raw is None:
                return None, default
            return np.asarray(raw, dtype=float).reshape(-1), default

        cols = {name: col(name, 0.0) for name in self.INPUTS}
        batch = max([len(c[0]) for c in cols.values() if c[0] is not None] or [1])

        def value(name, b, hi):
            arr, default = cols[name]
            v = default if arr is None else arr[min(b, len(arr) - 1)]
            return int(min(max(round(float(v)), 0), hi))

        outputs = {name: [] for name in self.model.node_order}
        inputs = {name: [] for name in self.INPUTS}
        for b in range(batch):
            pp_wg = [value(f"pp_Wg{i}", b, 2) for i in range(1, N_CELLS + 1)]
            pp_en = [value(f"pp_En{i}", b, 1) for i in range(1, N_CELLS + 1)]
            kos = [value(f"ko_{g}", b, 1) for g in KO_GENES]
            final = self.attractor_mean(pp_wg, pp_en, kos)
            for j, name in enumerate(self.model.node_order):
                outputs[name].append([float(final[j])])
            for name, v in zip(self.INPUTS, pp_wg + pp_en + kos):
                inputs[name].append([float(v)])
        values = {k: np.array(v) for k, v in {**inputs, **outputs}.items()}
        return input_state.merge(SystemState(values=values))


# Value map

class SegmentPolarityValueMap(ValueMap):
    """
    tau for the pathway-level abstraction.

    `Prepattern`  <-> the 12 pair-rule inputs (pp_Wg1..6 at level 0/2, pp_En1..6 at 0/1);
                        label k is `PREPATTERNS[k]`.
    `KO_{gene}`   <-> `ko_{Gene}` (0/1).
    `W{i}`        <-> Wg in cell i: 1 if expressed (level >= 1), else 0.
    `E{i}`        <-> En in cell i: 1 if expressed, else 0.

    Read-outs use the logical thresholds (0.5), so measurement noise maps to the nearest
    level; a micro input outside every prepattern abstracts to UNMAPPED.

    Args:
        cell_order: Micro cell feeding macro cell i (`cell_order[i-1]`); the identity
            for the valid map, reversed for the mirrored (wrong) map.
    """

    def __init__(self, cg_map: CoarseGrainingMap, cell_order=tuple(range(1, N_CELLS + 1))):
        tol = 0.4
        specs = {"Prepattern": {
            k: RectSubspace(*([(2.0 * (c == 1) - tol, 2.0 * (c == 1) + tol) for c in pat]
                              + [(1.0 * (c == 2) - tol, 1.0 * (c == 2) + tol) for c in pat]))
            for k, pat in enumerate(PREPATTERNS)}}
        bit = {0: RectSubspace((-tol, tol)), 1: RectSubspace((1 - tol, 1 + tol))}
        for g in KO_GENES:
            specs[f"KO_{g.lower()}"] = bit
        for i in range(1, N_CELLS + 1):
            specs[f"W{i}"] = {0: RectSubspace((-tol, tol)), 1: RectSubspace((2 - tol, 2 + tol))}
            specs[f"E{i}"] = bit
        super().__init__(cg_map, specs)
        self._index = {pat: k for k, pat in enumerate(PREPATTERNS)}

    def abstract(self, name: str, val: np.ndarray):
        val = np.asarray(val, dtype=float).ravel()
        if name == "Prepattern":
            wg, en = val[:N_CELLS], val[N_CELLS:]
            pat = tuple(1 if w > 1.0 else (2 if e > 0.5 else 0) for w, e in zip(wg, en))
            if any(w > 1.0 and e > 0.5 for w, e in zip(wg, en)):
                return UNMAPPED
            return self._index.get(pat, UNMAPPED)
        return int(val[0] > 0.5)


# Schema and CG-map builders

def _base_schema() -> MicroVariableSchema:
    """Pair-rule inputs, knockout switches and the 72 network nodes."""
    names = SegmentPolarityLowLevelModel.INPUTS + [
        f"{g}{i}" for i in range(1, N_CELLS + 1)
        for g in ("Wg", "Fz", "Dsh", "Slp", "Nkd", "En", "Hh", "Ci", "Ciact", "Cirep", "Pka", "Ptc")]
    return MicroVariableSchema([Variable(n, shape=(1,)) for n in names])


def _build_cg_vm(schema: MicroVariableSchema, cell_order):
    mapping = {"Prepattern": [f"pp_Wg{i}" for i in range(1, N_CELLS + 1)]
                             + [f"pp_En{i}" for i in range(1, N_CELLS + 1)]}
    for g in KO_GENES:
        mapping[f"KO_{g.lower()}"] = [f"ko_{g}"]
    for i, c in enumerate(cell_order, start=1):
        mapping[f"W{i}"] = [f"Wg{c}"]
        mapping[f"E{i}"] = [f"En{c}"]
    used = {m for sels in mapping.values() for m in sels}
    # All other network nodes are internal (not Phi)
    internal = sorted(schema.variable_names - used)
    cg = CoarseGrainingMap(schema, mapping, internal_variables=internal)
    return cg, SegmentPolarityValueMap(cg, cell_order)


def build_valid_cg_vm(schema: MicroVariableSchema):
    """Valid map: macro cell i is micro cell i."""
    return _build_cg_vm(schema, tuple(range(1, N_CELLS + 1)))


def build_wrong_cg_vm(schema: MicroVariableSchema):
    """Wrong map: macro cell i read from micro cell 7 - i (anterior/posterior mirrored)."""
    return _build_cg_vm(schema, tuple(range(N_CELLS, 0, -1)))


# High-level models

def reduced_steady_state(prepattern, kos):
    """Steady state of the pathway-level model of the Wg/Hh/En loop.

    Obtained from the 72-node network by eliminating the signaling intermediates at
    their quasi-steady values (a stable-state-preserving reduction in the sense of
    Naldi et al., 2011). Per cell i, with W = Wg expressed and E = En expressed:

        S_i = W_{i-1} or W_i or W_{i+1}       (Wg pathway active: Fz/Dsh; Wg is autocrine
                                               at any level and paracrine at level 2)
        H_i = E_{i-1} or E_{i+1}              (Hh from a neighboring En/Hh cell
                                               inactivates Ptc)
        E_i' = S_i and E_i                    (En is maintained by Wg signaling)
        W_i' = S_i and not E_i and H_i        (Wg is maintained by Hh signaling, via Slp
                                               and Ci; repressed by En)

    Knockouts: wg- -> W = 0, en- -> E = 0, hh- -> H = 0, ci- -> W = 0 (no Ci activator),
    dsh- -> S = 0. Same protocol as the low level: the prepattern is held for
    `HOLD_STEPS` synchronous steps, then released. Returns (W, E) per cell.
    """
    n = N_CELLS
    ko = dict(zip(KO_GENES, kos))

    def step(W, E):
        S = [int(not ko["Dsh"] and (W[i] or (i > 0 and W[i - 1]) or (i < n - 1 and W[i + 1])))
             for i in range(n)]
        H = [int(not ko["Hh"] and ((i > 0 and E[i - 1]) or (i < n - 1 and E[i + 1])))
             for i in range(n)]
        En = [int(not ko["En"] and S[i] and E[i]) for i in range(n)]
        Wn = [int(not (ko["Wg"] or ko["Ci"]) and S[i] and not E[i] and H[i]) for i in range(n)]
        return Wn, En

    W0 = [int(c == 1 and not ko["Wg"]) for c in prepattern]
    E0 = [int(c == 2 and not ko["En"]) for c in prepattern]
    W, E = W0, E0
    for _ in range(HOLD_STEPS):
        W, E = step(W0, E0)
    seen = set()
    while (tuple(W), tuple(E)) not in seen:
        seen.add((tuple(W), tuple(E)))
        W, E = step(W, E)
    return W, E


def cell_autonomous_steady_state(prepattern, kos):
    """Wrong model: no intercellular signaling, each cell keeps its pair-rule identity
    (only the knockout of the gene itself removes it)."""
    ko = dict(zip(KO_GENES, kos))
    W = [int(c == 1 and not ko["Wg"]) for c in prepattern]
    E = [int(c == 2 and not ko["En"]) for c in prepattern]
    return W, E


def _build_high_level_model(steady_fn) -> CausalGraph:
    hlm = CausalGraph()
    hlm.add_variable("Prepattern", lambda: None,
                     distribution={k: 1 / len(PREPATTERNS) for k in range(len(PREPATTERNS))})
    ko_names = [f"KO_{g.lower()}" for g in KO_GENES]
    for name in ko_names:
        hlm.add_variable(name, lambda: None, distribution={0: 0.8, 1: 0.2})
    roots = ["Prepattern"] + ko_names
    cache = {}

    def steady(**kw):
        key = (int(kw["Prepattern"]),) + tuple(int(kw[k]) for k in ko_names)
        if key not in cache:
            cache[key] = steady_fn(PREPATTERNS[key[0]], key[1:])
        return cache[key]

    for i in range(N_CELLS):
        hlm.add_variable(f"W{i + 1}", lambda i=i, **kw: steady(**kw)[0][i], parents=roots,
                         distribution={0: 0.5, 1: 0.5})
        hlm.add_variable(f"E{i + 1}", lambda i=i, **kw: steady(**kw)[1][i], parents=roots,
                         distribution={0: 0.5, 1: 0.5})
    return hlm


def build_valid_high_level_model() -> CausalGraph:
    """Pathway-level model of the Wg/Hh/En loop (`reduced_steady_state`)."""
    return _build_high_level_model(reduced_steady_state)


def build_cell_autonomous_high_level_model() -> CausalGraph:
    """Wrong model: the pair-rule prepattern is simply maintained, cell by cell."""
    return _build_high_level_model(cell_autonomous_steady_state)


OUTPUT_VARS = [f"{x}{i}" for i in range(1, N_CELLS + 1) for x in ("W", "E")]
ROOT_VARS = ["Prepattern"] + [f"KO_{g.lower()}" for g in KO_GENES]


def main():
    """Print the wild-type and mutant stable patterns of the full network and of the
    pathway-level model, and check that they agree on the whole macro domain."""
    import itertools
    grn_dir = os.path.dirname(os.path.abspath(__file__))
    model = SegmentPolarityModel.from_ginml(os.path.join(grn_dir, 'regulatoryGraph.ginml'))
    llm = SegmentPolarityLowLevelModel(model)
    idx = model.var_to_idx
    wt = PREPATTERNS.index((0, 1, 1, 2, 2, 0))
    for name, kos in [("WT", (0, 0, 0, 0, 0)), ("wg-", (1, 0, 0, 0, 0)), ("en-", (0, 1, 0, 0, 0)),
                      ("hh-", (0, 0, 1, 0, 0)), ("ci-", (0, 0, 0, 1, 0)), ("dsh-", (0, 0, 0, 0, 1))]:
        pat = PREPATTERNS[wt]
        final = llm.attractor_mean([2 * (c == 1) for c in pat], [int(c == 2) for c in pat], kos)
        cells = ' '.join(f"W{int(final[idx[f'Wg{i}']] > 0)}E{int(final[idx[f'En{i}']] > 0)}"
                         for i in range(1, N_CELLS + 1))
        print(f"{name:5s} full network: {cells}   pathway model: {reduced_steady_state(pat, kos)}")
    agree = total = 0
    for k, kos in itertools.product(range(len(PREPATTERNS)), itertools.product((0, 1), repeat=5)):
        pat = PREPATTERNS[k]
        final = llm.attractor_mean([2 * (c == 1) for c in pat], [int(c == 2) for c in pat], kos)
        W = [int(final[idx[f'Wg{i}']] > 0) for i in range(1, N_CELLS + 1)]
        E = [int(final[idx[f'En{i}']] > 0) for i in range(1, N_CELLS + 1)]
        agree += (W, E) == tuple(map(list, reduced_steady_state(pat, kos)))
        total += 1
    print(f"pathway model == full network on {agree}/{total} (prepattern, genotype) cells")


if __name__ == '__main__':
    main()

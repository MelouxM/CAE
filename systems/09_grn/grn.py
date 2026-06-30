"""
Example 9: Segment polarity gene regulatory network
We evaluate whether a high-level description of the Wg->Fz paracrine signaling
pathway is a valid causal abstraction of the 6-cell segment polarity network
(Sánchez, Chaouiya & Thieffry, 2008).

Low-level model:
- A linear array of 6 identical "cells" (cells 1 and 6 are boundaries), each with
  12 internal variables (0-1 or 0-2).
- Variables update according to logical rules (multi-valued Boolean network).
- Cells are coupled: Wg (wingless) protein from cell i activates Fz (frizzled)
  in neighboring cells i±1 (paracrine signaling, requires level 2).
- For the evaluation, we apply one synchronous update step.
- Micro-variables: wg_c2 (Wg level in cell 2, input); fz_c1, fz_c3, fz_c4,
  fz_c5 (Fz in various cells, outputs). All other variables start at 0.

High-level model:
- A simple causal rule representing the signaling chain.
- Abstract input: wg_src ∈ {0=low, 1=high} - is Wg at the paracrine level?
- Abstract output: fz_nbr ∈ {0=off, 1=on} - do the target cells express Fz?
- Rule: wg_src=1 -> fz_nbr=1; wg_src=0 -> fz_nbr=0.

Coarse-graining:
- Maps the low-level specific cell variables to high-level abstract variables.
- A valid spatial abstraction must capture the adjacency structure: setting Wg=2
  in cell 2 causes Fz=1 in its true neighbors (cells 1 and 3), but not in
  non-adjacent cells (4 and 5).

Value map:
- Discrete Wg-level / Fz-on labels mapped via RectSubspace.
- wg_src labels: 0 = Wg below paracrine threshold (< 2); 1 = high (= 2).
- fz_tgt labels: 0 = all target cells off; 1 = any target cell on (abstract() uses OR).

Additional details:
- We test four conditions to verify the library detects mis-specifications:
  - Valid: CG map links wg_c2 to fz in cells 1,3 (true neighbors) + correct high-level model.
  - Wrong map: CG map links wg_c2 to fz in cells 4,5 (non-neighbors) + correct high-level model.
  - Reversed: Valid CG map, but high-level model rule is reversed (Wg=high -> Fz=off).
  - Noise: Valid CG map + valid high-level model, but Gaussian noise on Fz output.
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
    EvaluationEngine, EvaluationConfig, TopDownSampler,
    Variable, RectSubspace, NoisyLowLevelModel,
    MSEMetric, L2Metric, R2Metric, IIAMetric,
    MacroscopicInvarianceMetric, ComplexityShiftMetric,
)
from causal_abstraction.tasks import StandardTasks


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
    5. If no parameter matches, next value = basevalue

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

        # No parameter matched -> use base value
        return node.basevalue

    def synchronous_step(self, state: np.ndarray) -> np.ndarray:
        """Compute one synchronous update step (all variables update simultaneously)."""
        new_state = np.zeros_like(state)
        for i, nid in enumerate(self.node_order):
            new_state[i] = self.compute_target_value(state, nid)

        return new_state

# low-level model: one-step segment polarity dynamics

class SegmentPolarityLowLevelModel(LowLevelModel):
    """
    One-step synchronous dynamics of the segment polarity network.

    Micro-variables:
    Input  : wg_c2  - Wg level in cell 2 (shape (1,), range 0-2).
    Outputs: fz_c1, fz_c3, fz_c4, fz_c5 - Fz in cells 1, 3, 4, 5 (shape (1,) each).
    All variables except wg_c2 start at 0 for each evaluation sample.
    """

    def __init__(self, model: SegmentPolarityModel):
        self.model = model

    def forward_with_interventions(self,
                                   input_state: SystemState,
                                   interventions: dict) -> SystemState:
        # Retrieve wg_c2 - shape (batch, 1) or (1,)
        raw = interventions.get('wg_c2', input_state.get('wg_c2', np.zeros((1, 1))))
        wg_c2 = np.asarray(raw, dtype=float)
        batch_size = wg_c2.shape[0]

        results = {k: [] for k in ('wg_c2', 'fz_c1', 'fz_c3', 'fz_c4', 'fz_c5')}

        for b in range(batch_size):
            wg_val = int(round(float(wg_c2[b, 0])))
            wg_val = max(0, min(2, wg_val))   # clamp to valid range

            # Build all-zero initial state, then set Wg2
            init_dict = {nid: 0 for nid in self.model.node_order}
            init_dict['Wg2'] = wg_val

            state = self.model.state_to_array(init_dict)
            next_state = self.model.synchronous_step(state)

            def _get(node): return float(next_state[self.model.var_to_idx[node]])

            results['wg_c2'].append([float(wg_val)])
            results['fz_c1'].append([_get('Fz1')])
            results['fz_c3'].append([_get('Fz3')])
            results['fz_c4'].append([_get('Fz4')])
            results['fz_c5'].append([_get('Fz5')])

        out = {k: np.array(v, dtype=float) for k, v in results.items()}
        return input_state.merge(SystemState(values=out))


# Value map: discrete Wg-level / Fz-on labels
# Wg source: 0 = Wg below paracrine threshold (< 2); 1 = high (= 2)
_WG_SPECS = {
    0: RectSubspace((-0.1, 1.9)),   # Wg in [0, 1]
    1: RectSubspace((1.9, 2.1)),    # Wg = 2 (paracrine-active)
}

# Fz target pair grounding representatives: 0 = all off; 1 = all on
# (abstract() labels a mixed micro-state via OR; these rects are the canonical
#  representatives used for grounding / surjectivity, not the abstraction rule.)
_FZ_PAIR_SPECS = {
    0: RectSubspace((-0.1, 0.1), (-0.1, 0.1)),
    1: RectSubspace((0.9, 1.1), (0.9, 1.1)),
}


class WgFzValueMap(ValueMap):
    """
    ValueMap for the Wg-source / Fz-target causal test.

    wg_src labels  : 0 = low Wg (< 2),  1 = high Wg (= 2, paracrine active).
    fz_tgt labels  : 0 = all target cells off,  1 = any target cell on (OR).
    """

    def __init__(self, cg_map: CoarseGrainingMap):
        super().__init__(cg_map, {
            'wg_src': _WG_SPECS,
            'fz_tgt': _FZ_PAIR_SPECS,
        })

    def abstract(self, name: str, val: np.ndarray):
        val = np.asarray(val, dtype=float).flatten()
        if name == 'wg_src':
            return int(val[0] >= 1.9)       # 1 if Wg = 2, else 0
        else: # fz_tgt
            return int(np.any(val > 0.5))   # 1 if any target cell has Fz = 1

    def ground(self, name: str, label, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        if name == 'wg_src':
            return np.array([0.0] if int(label) == 0 else [2.0], dtype=float)
        else: # fz_tgt
            if int(label) == 0:
                return np.zeros(2, dtype=float)
            return np.ones(2, dtype=float)


# Schema and CG-map builders
def _base_schema() -> MicroVariableSchema:
    """Schema with wg_c2 (input) and Fz in cells 1, 3, 4, 5 (outputs)."""
    return MicroVariableSchema([
        Variable('wg_c2', shape=(1,)),
        Variable('fz_c1', shape=(1,)),
        Variable('fz_c3', shape=(1,)),
        Variable('fz_c4', shape=(1,)),
        Variable('fz_c5', shape=(1,)),
    ])


def build_valid_cg_vm(schema: MicroVariableSchema):
    """
    Valid CG map: wg_c2 -> wg_src, {fz_c1, fz_c3} -> fz_tgt.
    Cells 1 and 3 are the true neighbors of cell 2 in the linear array.
    """
    cg = CoarseGrainingMap(schema, {
        'wg_src': ['wg_c2'],
        'fz_tgt': ['fz_c1', 'fz_c3'],
    })
    vm = WgFzValueMap(cg)
    return cg, vm


def build_wrong_cg_vm(schema: MicroVariableSchema):
    """
    Wrong CG map: wg_c2 -> wg_src, {fz_c4, fz_c5} -> fz_tgt.
    Cells 4 and 5 are not neighbors of cell 2 - Wg2 has no effect on them.
    This tests whether the library detects the mis-attribution of causal influence.
    """
    cg = CoarseGrainingMap(schema, {
        'wg_src': ['wg_c2'],
        'fz_tgt': ['fz_c4', 'fz_c5'],
    })
    vm = WgFzValueMap(cg)
    return cg, vm


# high-level model builders
def _base_high_level_model_inputs() -> CausalGraph:
    high_level_model = CausalGraph()
    high_level_model.add_variable('wg_src', lambda: None, distribution={0: 0.5, 1: 0.5})
    return high_level_model


def build_valid_high_level_model() -> CausalGraph:
    """
    Correct high-level model: wg_src = 1 (high Wg) -> fz_tgt = 1 (Fz on in true neighbors).
    """
    high_level_model = _base_high_level_model_inputs()

    def valid_rule(wg_src):
        return int(wg_src)   # 0 -> 0,  1 -> 1

    high_level_model.add_variable('fz_tgt', valid_rule, parents=['wg_src'],
                     distribution={0: 0.5, 1: 0.5})
    return high_level_model


def build_reversed_high_level_model() -> CausalGraph:
    """
    Wrong high-level model: reverses the prediction - high Wg -> Fz off, low Wg -> Fz on.
    Uses the valid CG map, so the mis-specification is in the causal rule.
    """
    high_level_model = _base_high_level_model_inputs()

    def reversed_rule(wg_src):
        return 1 - int(wg_src)   # reversed

    high_level_model.add_variable('fz_tgt', reversed_rule, parents=['wg_src'],
                     distribution={0: 0.5, 1: 0.5})
    return high_level_model


# Evaluation
def main():
    """
    Evaluate all four conditions of the causal abstraction.
    """

    # Load model
    print("\nLoading model from GINsim files...")
    _grn_dir = os.path.dirname(os.path.abspath(__file__))
    model = SegmentPolarityModel.from_ginml(os.path.join(_grn_dir, 'regulatoryGraph.ginml'))

    print("\nEvaluating abstraction (Wg->Fz paracrine signaling) across 4 conditions...")

    n_samples = 500

    schema = _base_schema()
    low_level_model = SegmentPolarityLowLevelModel(model)
    noisy_low_level_model = NoisyLowLevelModel(low_level_model, noise_std=0.4, noise_type='gaussian')

    cg_valid, vm_valid = build_valid_cg_vm(schema)
    cg_wrong, vm_wrong = build_wrong_cg_vm(schema)

    high_level_model_valid = build_valid_high_level_model()
    high_level_model_rev = build_reversed_high_level_model()

    cfg = EvaluationConfig(metric='mse', seed=0)
    out_vars = ['fz_tgt']
    interv_vars = ['wg_src']

    conditions = [
        ('Valid (adjacent CG map + correct high-level model)',    high_level_model_valid, low_level_model,       cg_valid, vm_valid),
        ('Wrong map (non-adjacent CG map)',          high_level_model_valid, low_level_model,       cg_wrong, vm_wrong),
        ('Reversed high-level model (correct map, wrong rule)',   high_level_model_rev,   low_level_model,       cg_valid, vm_valid),
        ('Noise (σ=0.4 on Fz output)',               high_level_model_valid, noisy_low_level_model, cg_valid, vm_valid),
    ]

    for title, high_level_model, cur_low_level_model, cg, vm in conditions:
        print(f"\n{'=' * 62}")
        print(f"  {title}")
        print(f"{'=' * 62}")

        eng = EvaluationEngine(high_level_model, cur_low_level_model, vm, cg, cfg)
        builder = eng.builder
        sampler = TopDownSampler(vm)

        tasks = [
            StandardTasks.score(builder, sampler=sampler, name='CAE_down'),
            StandardTasks.observational(builder, L2Metric(), sampler=sampler, output_vars=out_vars, name='L2'),
            StandardTasks.observational(builder, MSEMetric(), sampler=sampler, output_vars=out_vars, name='MSE'),
            StandardTasks.observational(builder, R2Metric(), sampler=sampler, output_vars=out_vars, name='R²'),
        ]

        results = eng.run_tasks(
            tasks, n_samples=n_samples, batch_size=1, max_interventions=1, intervention_domain=interv_vars,
        )
        print(results)

        analytical = eng.run_analytical_metrics(
            [
                IIAMetric(inner_metric='mse', output_vars=out_vars, n_pairs=10),
                MacroscopicInvarianceMetric(n_pairs=5, inner_metric='mse'),
                ComplexityShiftMetric(),
            ],
            sampler=sampler,
            n_samples=min(n_samples, 30),
        )
        print('\nAnalytical metrics:')
        for an_name, res in analytical.items():
            summary = {k: f'{v:.4f}' if isinstance(v, float) else str(v)
                       for k, v in res.items() if not isinstance(v, list)}
            print(f'  {an_name}: {summary}')


if __name__ == '__main__':
    main()
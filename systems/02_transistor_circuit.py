"""
Example 2: Hierarchical abstraction from physics to logic

This example demonstrates a "vertical" abstraction layer between continuous physical dynamics
(analog circuit simulation) and discrete digital logic.

We simulate a CMOS half-adder using a SPICE solver (PySpice/Ngspice). The low-level
model operates on voltages and currents through MOSFETs, while the high-level model
operates on boolean logic values.

This example uses internal_variables in the coarse-graining map (a) to mark redundant
internal nodes that are part of the mechanism but should not be intervened on.

Low-level model:
- SPICE simulation of CMOS circuits (NAND, half-adder).
- Variables: Continuous voltage nodes (a, b, sum, carry, vdd, n_1, nand1_out, etc.).

High-level model:
- Causal graph representing Boolean logic.
- Logic: Sum = A XOR B, Carry = A AND B.

Coarse-graining map:
- Maps inputs (A, B) and outputs (Sum, Carry) to their corresponding circuit nodes.
- Identifies internal variables (power rails, intermediate gate wires, transistor junctions)
  to exclude them from the set of unmapped variables (phi).

Value map:
- Maps discrete binary states (0, 1) to continuous voltage regions (0V range, 5V range).
"""
import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

try:
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import u_V, u_F
except ImportError:
    print("WARNING: PySpice not found.")
    exit(0)

from causal_abstraction import (
    SystemState, MicroVariableSchema, LowLevelModel
)

# Circuit builders

def create_base_circuit(name):
    circuit = Circuit(name)
    circuit.model('pmos', 'pmos', Vto=-1, Kp=20e-6)
    circuit.model('nmos', 'nmos', Vto=1, Kp=20e-6)
    circuit.V('dd', 'vdd', circuit.gnd, 5.0 @ u_V)
    return circuit

def add_nand(circuit, name, in1, in2, out):
    circuit.MOSFET(f'p1_{name}', out, in1, 'vdd', 'vdd', model='pmos')
    circuit.MOSFET(f'p2_{name}', out, in2, 'vdd', 'vdd', model='pmos')
    circuit.MOSFET(f'n1_{name}', f'n_{name}', in1, circuit.gnd, circuit.gnd, model='nmos')
    circuit.MOSFET(f'n2_{name}', out, in2, f'n_{name}', circuit.gnd, model='nmos')
    circuit.C(f'load_{name}', out, circuit.gnd, 1e-15 @ u_F)

def add_inverter(circuit, name, in1, out):
    circuit.MOSFET(f'p_{name}', out, in1, 'vdd', 'vdd', model='pmos')
    circuit.MOSFET(f'n_{name}', out, in1, circuit.gnd, circuit.gnd, model='nmos')

def nand_topology(circuit, input_names):
    add_nand(circuit, '1', 'a', 'b', 'output')  # Creates internal node: 'n_1'

def half_adder_topology(circuit, input_names):
    # 4 NANDs + 1 inverter, creates internal nodes n_1 to n_4 from NANDs.
    # add_inverter creates a MOSFET named n_{name} (not a node)
    # Intermediate logic wires: nand1_out, nand2_out, nand3_out
    add_nand(circuit, '1', 'a', 'b', 'nand1_out')
    add_nand(circuit, '2', 'a', 'nand1_out', 'nand2_out')
    add_nand(circuit, '3', 'b', 'nand1_out', 'nand3_out')
    add_nand(circuit, '4', 'nand2_out', 'nand3_out', 'sum')
    add_inverter(circuit, '5', 'nand1_out', 'carry')

# Low-level model

class GenericSpiceModel(LowLevelModel):
    """Low-level model M: a SPICE/Ngspice operating-point simulator over circuit voltage nodes."""

    def __init__(self, build_fn, input_names):
        self.build_fn = build_fn
        self.input_names = input_names

        # Introspection step: build the circuit to discover nodes
        temp = create_base_circuit("Temp")
        self.build_fn(temp, self.input_names)

        # Extract all node names
        self.all_nodes = [str(n).lower() for n in temp.node_names]

        self.schema = MicroVariableSchema.from_names(self.all_nodes)

    def forward_with_interventions(self, input_state: SystemState, interventions):
        current_vals = {}

        # Give a default voltage (0V) to all declared inputs so SPICE never sees a floating input node
        for inp in self.input_names:
            current_vals[inp.lower()] = 0.0

        # Load state (overrides defaults)
        for k in self.all_nodes:
            if k in input_state.values:
                current_vals[k] = input_state[k].item()

        # Apply interventions
        for k, v in interventions.items():
            # Handle potential batch dimension
            if hasattr(v, 'item'):
                val = v.item()
            elif isinstance(v, (list, np.ndarray)):
                val = v[0].item() if isinstance(v[0], (np.ndarray, np.generic)) else v[0]
            else:
                val = v
            current_vals[k.lower()] = val

        # Build simulation circuit
        circuit = create_base_circuit("Sim")
        # Pass the list of intervened keys so the builder knows what is forced
        self.build_fn(circuit, list(interventions.keys()))

        # Drive nodes
        for node, val in current_vals.items():
            if node not in ['0', 'gnd', 'vdd']:  # Don't drive ground/VDD
                try:
                    circuit.V(f'drive_{node}', node, circuit.gnd, float(val) @ u_V)
                except:
                    pass

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        results = {}
        for name in self.all_nodes:
            if name in analysis.nodes:
                val = float(analysis.nodes[name].as_ndarray().item())
                results[name] = np.array([[val]])  # (1, 1)
            elif name in current_vals:
                results[name] = np.array([[current_vals[name]]])  # (1, 1)
            else:
                results[name] = np.zeros((1, 1))  # (1, 1)

        del simulator, analysis, circuit

        return input_state.merge(SystemState(values=results))


def run_full_suite(*args, **kwargs):
    """Deprecated shim. Run the evaluation suite from the command line instead:
    ``python test/02_transistor_circuit.py`` (built on ``test/runner.run_suite``)."""
    raise RuntimeError(
        "run_full_suite is not available here; run the transistor evaluation "
        "suite directly with `python test/02_transistor_circuit.py`."
    )


def main():
    """Simulate NAND gate and half-adder for all binary input combinations."""
    # Logic: 0V = logic 0, 5V = logic 1
    voltages = {0: 0.0, 1: 5.0}

    print("NAND gate simulation:")
    print(f"{'A':<4} {'B':<4} {'Output (V)':<12} {'Logic'}")
    print("-" * 30)
    nand_low_level_model = GenericSpiceModel(nand_topology, ['a', 'b'])
    for a_bit, b_bit in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        state = nand_low_level_model.forward_with_interventions(
            SystemState(values={}),
            {"a": np.array([voltages[a_bit]]), "b": np.array([voltages[b_bit]])},
        )
        out_v = float(np.asarray(state.values.get("output", [0])).ravel()[0])
        out_logic = 0 if out_v < 2.5 else 1
        print(f"{a_bit:<4} {b_bit:<4} {out_v:<12.2f} {out_logic}")

    print("\nHalf-adder simulation:")
    print(f"{'A':<4} {'B':<4} {'Sum (V)':<10} {'Carry (V)':<12} {'Sum':<6} {'Carry'}")
    print("-" * 45)
    ha_low_level_model = GenericSpiceModel(half_adder_topology, ['a', 'b'])
    for a_bit, b_bit in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        state = ha_low_level_model.forward_with_interventions(
            SystemState(values={}),
            {"a": np.array([voltages[a_bit]]), "b": np.array([voltages[b_bit]])},
        )
        sum_v   = float(np.asarray(state.values.get("sum",   [0])).ravel()[0])
        carry_v = float(np.asarray(state.values.get("carry", [0])).ravel()[0])
        sum_l   = 0 if sum_v   < 2.5 else 1
        carry_l = 0 if carry_v < 2.5 else 1
        print(f"{a_bit:<4} {b_bit:<4} {sum_v:<10.2f} {carry_v:<12.2f} {sum_l:<6} {carry_l}")


if __name__ == '__main__':
    main()
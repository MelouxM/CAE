"""
Example 1: 2-bit adder with structural abstraction
Here, we compare a logic simulation of a 2-bit adder against a causal graph representing the same operation.
The key challenge in this abstraction is handling internal mechanisms: intermediate logic gates (wires)
that are essential for computation but are "implementation details" not present in the high-level model.

Low-level model:
- A simulation of a 2-bit adder circuit using logic gates with continuous, noisy voltage states.
- Variables: Inputs (A0-1, B0-1, C0), Output (S0-1, C2), and internal gates (XOR, AND outputs).

High-level model:
- A causal graph implementing 2-bit arithmetic.
- Logic: Sum = A + B + Cin.

Coarse-graining map:
- Maps 2-bit vectors (e.g., A0, A1) to high-level integers (Operand_A).

Value map:
- Inputs/Outputs: Continuous regions around 0.0 (low) and 1.0 (high) mapped to integers.
- We use internal_variables to inform the library of the redundant internal wires.

Experiment structure:
- Valid abstraction: correct high-level model
- Failing abstraction: high-level model uses OR instead of XOR for the sum bit (incorrect logic).
- Inverted internal: high-level model inverts Internal_Carries but compensates downstream,
  so final I/O is correct but intermediate state is wrong.
- Noise experiment: correct high-level model, noisy low-level model (Gaussian noise on wire voltages).
"""
import numpy as np

from causal_abstraction import CausalGraph, LowLevelModel, SystemState, CoarseGrainingMap, ValueMap, RectSubspace

# Low-level: Generic netlist simulator

class NetlistSimulator(LowLevelModel):
    """Low-level model M: a gate-level netlist simulator (the 2-bit adder circuit)."""

    def __init__(self, gates, all_wires):
        self.gates = gates
        self.all_wires = sorted(list(set(all_wires)))
        self.state = {w: False for w in self.all_wires}

    def _evaluate_gate(self, g_type, inputs):
        if g_type == 'AND': return all(inputs)
        if g_type == 'OR':  return any(inputs)
        if g_type == 'XOR': return sum(inputs) % 2 == 1
        return False

    @staticmethod
    def _scalar_to_bool(val, batch_idx):
        """Extract a single boolean from a possibly-batched value."""
        if isinstance(val, np.ndarray):
            if val.ndim >= 2:
                return float(val[batch_idx].ravel()[0]) > 0.5
            elif val.ndim == 1:
                return float(val[batch_idx]) > 0.5
            else:
                return float(val) > 0.5
        if hasattr(val, 'item'):
            return val.item() > 0.5
        return float(val) > 0.5

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        # Determine batch size
        batch_size = 1
        for v in interventions.values():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                batch_size = v.shape[0]
                break

        all_results = {w: [] for w in self.all_wires}

        for b in range(batch_size):
            current_state = {w: False for w in self.all_wires}

            for k, v in input_state.values.items():
                if k in current_state:
                    current_state[k] = self._scalar_to_bool(v, b)

            forced = set()
            for k, v in interventions.items():
                if k in current_state:
                    current_state[k] = self._scalar_to_bool(v, b)
                    forced.add(k)

            # Defensive bound: an acyclic netlist reaches its fixed point in at
            # most len(gates) passes; the cap only prevents a hang on a malformed
            # cyclic netlist. It is never reached for the benchmark circuits, so
            # converged outputs are unchanged.
            for _ in range(len(self.gates) + 2):
                stable = True
                for g in self.gates:
                    out_w = g['out']
                    if out_w in forced:
                        continue
                    in_vals = [current_state[w] for w in g['in']]
                    new_val = self._evaluate_gate(g['type'], in_vals)
                    if current_state[out_w] != new_val:
                        current_state[out_w] = new_val
                        stable = False
                if stable:
                    break

            for w in self.all_wires:
                all_results[w].append(float(current_state[w]))

        final_values = {
            k: np.array(v).reshape(batch_size, 1)
            for k, v in all_results.items()
        }
        return input_state.merge(SystemState(values=final_values))


# Circuit topology

def build_2bit_adder():
    gates = []
    all_wires = ['C0', 'C1', 'C2', 'A0', 'A1', 'B0', 'B1', 'S0', 'S1']  # C0 (in), C1 (internal), C2 (out)

    for i in range(2):
        cin = f'C{i}'
        cout = f'C{i+1}'
        a, b, s = f'A{i}', f'B{i}', f'S{i}'

        xor1 = f'bit{i}_xor1'
        and1 = f'bit{i}_and1'
        and2 = f'bit{i}_and2'
        all_wires.extend([xor1, and1, and2])

        gates.append({'type': 'XOR', 'in': [a, b], 'out': xor1})
        gates.append({'type': 'XOR', 'in': [xor1, cin], 'out': s})
        gates.append({'type': 'AND', 'in': [xor1, cin], 'out': and1})
        gates.append({'type': 'AND', 'in': [a, b], 'out': and2})
        gates.append({'type': 'OR',  'in': [and1, and2], 'out': cout})

    return gates, sorted(list(set(all_wires)))


# Value map

class BinaryValueMap(ValueMap):
    """Value map tau: maps continuous wire voltages (low~0.0, high~1.0) to integer macro-states."""

    def __init__(self, cg, n_bits_map):
        self.n_bits_map = n_bits_map
        specs = {}
        zero_range = (-0.1, 0.1)
        one_range = (0.9, 1.1)

        for var, n_bits in n_bits_map.items():
            var_specs = {}
            for val in range(2**n_bits):
                intervals = []
                for b in range(n_bits):
                    bit = (val >> b) & 1
                    intervals.append(one_range if bit else zero_range)
                var_specs[val] = RectSubspace(*intervals)
            specs[var] = var_specs
        super().__init__(cg, specs)

    def abstract(self, name: str, val: np.ndarray) -> np.ndarray:
        if val.ndim == 1: val = val[np.newaxis, :]
        batch_size, _ = val.shape
        n_bits = self.n_bits_map[name]
        results = []
        for i in range(batch_size):
            vec = val[i]
            int_val = 0
            for b in range(n_bits):
                if vec[b] > 0.5:
                    int_val |= (1 << b)
            results.append(int_val)
        res_array = np.array(results)
        if batch_size == 1:
            return np.array(results[0])
        return res_array


# High-level model builders

def build_valid_high_level_model():
    """Correct 2-bit adder high-level model."""
    high_level_model = CausalGraph()
    # A, B are 2-bit (vals 0-3), Cin is 1-bit
    dist_2bit = {i: 1/4 for i in range(4)}
    dist_1bit = {0: 0.5, 1: 0.5}

    high_level_model.add_variable("Operand_A", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Operand_B", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Carry_In",  lambda: None, distribution=dist_1bit)

    # Internal carries: only C1
    def carries_logic(Operand_A, Operand_B, Carry_In):
        # Calculate C1
        a0 = (Operand_A >> 0) & 1
        b0 = (Operand_B >> 0) & 1
        c0 = Carry_In
        return (a0 & b0) | (c0 & (a0 ^ b0))

    high_level_model.add_variable("Internal_Carries", carries_logic,
                     parents=["Operand_A", "Operand_B", "Carry_In"],
                     distribution=dist_1bit)

    def sum_logic(Operand_A, Operand_B, Carry_In, Internal_Carries):
        c_vec = [Carry_In, Internal_Carries]  # C0, C1
        s = 0
        for i in range(2):
            a = (Operand_A >> i) & 1
            b = (Operand_B >> i) & 1
            c = c_vec[i]
            s |= ((a ^ b ^ c) << i)
        return s

    # C2
    def cout_logic(Operand_A, Operand_B, Internal_Carries):
        c1 = Internal_Carries
        a1 = (Operand_A >> 1) & 1
        b1 = (Operand_B >> 1) & 1
        return (a1 & b1) | (c1 & (a1 ^ b1))

    high_level_model.add_variable("Result_Sum",   sum_logic,  parents=["Operand_A", "Operand_B", "Carry_In", "Internal_Carries"], distribution=dist_2bit)
    high_level_model.add_variable("Result_Carry", cout_logic, parents=["Operand_A", "Operand_B", "Internal_Carries"],             distribution=dist_1bit)
    return high_level_model


def build_failing_high_level_model():
    """
    Wrong high-level model: uses OR instead of XOR for the sum bit.
    """
    high_level_model = CausalGraph()
    dist_2bit = {i: 1/4 for i in range(4)}
    dist_1bit = {0: 0.5, 1: 0.5}

    high_level_model.add_variable("Operand_A", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Operand_B", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Carry_In",  lambda: None, distribution=dist_1bit)

    def carries_logic(Operand_A, Operand_B, Carry_In):
        a0 = (Operand_A >> 0) & 1
        b0 = (Operand_B >> 0) & 1
        return (a0 & b0) | (Carry_In & (a0 | b0))  # OR instead of XOR

    high_level_model.add_variable("Internal_Carries", carries_logic,
                     parents=["Operand_A", "Operand_B", "Carry_In"],
                     distribution=dist_1bit)

    def sum_logic_wrong(Operand_A, Operand_B, Carry_In, Internal_Carries):
        c_vec = [Carry_In, Internal_Carries]
        s = 0
        for i in range(2):
            a = (Operand_A >> i) & 1
            b = (Operand_B >> i) & 1
            c = c_vec[i]
            s |= ((a | b | c) << i)  # OR instead of XOR (wrong)
        return s

    def cout_logic(Operand_A, Operand_B, Internal_Carries):
        c1 = Internal_Carries
        a1 = (Operand_A >> 1) & 1
        b1 = (Operand_B >> 1) & 1
        return (a1 & b1) | (c1 & (a1 ^ b1))

    high_level_model.add_variable("Result_Sum",   sum_logic_wrong, parents=["Operand_A", "Operand_B", "Carry_In", "Internal_Carries"], distribution=dist_2bit)
    high_level_model.add_variable("Result_Carry", cout_logic,      parents=["Operand_A", "Operand_B", "Internal_Carries"],             distribution=dist_1bit)
    return high_level_model


def build_inverted_internal_high_level_model():
    """
    Almost-valid high-level model: Internal_Carries is inverted (NOT of the true carry),
    but all downstream consumers (Result_Sum, Result_Carry) flip it back
    before use. Final I/O is always correct; the intermediate representation
    is wrong. This should be caught by interventional metrics on the
    Internal_Carries node.
    """
    high_level_model = CausalGraph()
    dist_2bit = {i: 1/4 for i in range(4)}
    dist_1bit = {0: 0.5, 1: 0.5}

    high_level_model.add_variable("Operand_A", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Operand_B", lambda: None, distribution=dist_2bit)
    high_level_model.add_variable("Carry_In",  lambda: None, distribution=dist_1bit)

    # Internal carry is INVERTED: returns NOT(c1)
    def carries_logic_inv(Operand_A, Operand_B, Carry_In):
        a0 = (Operand_A >> 0) & 1
        b0 = (Operand_B >> 0) & 1
        c0 = Carry_In
        c1 = (a0 & b0) | (c0 & (a0 ^ b0))
        return 1 - c1  # inverted

    high_level_model.add_variable("Internal_Carries", carries_logic_inv,
                     parents=["Operand_A", "Operand_B", "Carry_In"],
                     distribution=dist_1bit)

    # Sum logic: flip Internal_Carries back before use
    def sum_logic_compensated(Operand_A, Operand_B, Carry_In, Internal_Carries):
        real_c1 = 1 - Internal_Carries  # undo inversion
        c_vec = [Carry_In, real_c1]
        s = 0
        for i in range(2):
            a = (Operand_A >> i) & 1
            b = (Operand_B >> i) & 1
            c = c_vec[i]
            s |= ((a ^ b ^ c) << i)
        return s

    # Cout logic: flip Internal_Carries back before use
    def cout_logic_compensated(Operand_A, Operand_B, Internal_Carries):
        real_c1 = 1 - Internal_Carries  # undo inversion
        a1 = (Operand_A >> 1) & 1
        b1 = (Operand_B >> 1) & 1
        return (a1 & b1) | (real_c1 & (a1 ^ b1))

    high_level_model.add_variable("Result_Sum",   sum_logic_compensated,
                     parents=["Operand_A", "Operand_B", "Carry_In", "Internal_Carries"],
                     distribution=dist_2bit)
    high_level_model.add_variable("Result_Carry", cout_logic_compensated,
                     parents=["Operand_A", "Operand_B", "Internal_Carries"],
                     distribution=dist_1bit)
    return high_level_model


# Helper: build CoarseGrainingMap and ValueMap
def build_cg_and_vm(schema):
    cg_map_dict = {
        "Operand_A":       ["A0", "A1"],
        "Operand_B":       ["B0", "B1"],
        "Carry_In":        ["C0"],
        "Internal_Carries":["C1"],
        "Result_Sum":      ["S0", "S1"],
        "Result_Carry":    ["C2"],
    }
    internal_wires = [
        'bit0_xor1', 'bit0_and1', 'bit0_and2',
        'bit1_xor1', 'bit1_and1', 'bit1_and2',
    ]
    cg = CoarseGrainingMap(schema, cg_map_dict, internal_variables=internal_wires)
    n_bits_config = {
        "Operand_A": 2, "Operand_B": 2, "Carry_In": 1,
        "Internal_Carries": 1, "Result_Sum": 2, "Result_Carry": 1,
    }
    vm = BinaryValueMap(cg, n_bits_config)
    return cg, vm


# Main
def main():
    """Simulate 2-bit adder for all input combinations."""
    gates, all_wires = build_2bit_adder()
    low_level_model = NetlistSimulator(gates, all_wires)

    print("2-bit adder gate simulation (A + B + Cin):")
    print(f"{'A':<5} {'B':<5} {'Cin':<6} {'Sum':<6} {'Cout'}")
    print("-" * 28)
    for a in range(4):
        for b in range(4):
            for cin in range(2):
                state = low_level_model.forward_with_interventions(
                    SystemState(values={}),
                    {
                        "A0": np.array([[float((a >> 0) & 1)]]),
                        "A1": np.array([[float((a >> 1) & 1)]]),
                        "B0": np.array([[float((b >> 0) & 1)]]),
                        "B1": np.array([[float((b >> 1) & 1)]]),
                        "C0": np.array([[float(cin)]]),
                    }
                )
                s0 = int(np.asarray(state.values["S0"]).ravel()[0] > 0.5)
                s1 = int(np.asarray(state.values["S1"]).ravel()[0] > 0.5)
                c2 = int(np.asarray(state.values["C2"]).ravel()[0] > 0.5)
                sum_val = s0 | (s1 << 1)
                expected = a + b + cin
                ok = "✓" if (sum_val == expected % 4 and c2 == int(expected >= 4)) else "✗"
                print(f"{a:<5} {b:<5} {cin:<6} {sum_val:<6} {c2}  {ok}")


if __name__ == "__main__":
    main()
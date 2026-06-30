"""
Example 10: Three-level causal abstraction of the MOS 6502 CPU.

    Level 0 (transistor): perfect6502  ~3500 transistors, half-cycle
    Level 1 (gate):       break6502    decoder PLA + datapath, half-cycle
    Level 2 (ISA):        fake6502     instruction-level, no timing

Bridge design:
All three bridges use the same register-state convention:

  ISA bridge:
    memset(64KB, 0) -> place opcode at $0400 -> set regs directly -> step

  Gate & Transistor bridges:
    memset(64KB, 0) -> write preamble+test+postamble at $0200
    -> reset core via /RES pin -> run until halt -> read from ZP $F0-$F4

P-flag handling:
All outputs mask P to 0xCF (clear bits 4-5).
Bit 4 (B) is push-only; bit 5 is always 1 in hardware.
"""

import ctypes
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from causal_abstraction import (
    AbstractVariable, CausalGraph, ContinuousValueMap, CoarseGrainingMap,
    FullSubspace, LowLevelModel, MicroVariableSchema, RectSubspace, SystemState,
)
from causal_abstraction.sampling import InterventionSampler
from causal_abstraction.schema import Variable

logger = logging.getLogger(__name__)

P_MASK = 0xCF

# Instruction tables

INSTR_LENGTH = [0] * 256
for _op in [0x00,0x08,0x0A,0x18,0x1A,0x28,0x2A,0x38,0x3A,0x40,0x48,0x4A,0x58,
           0x60,0x68,0x6A,0x78,0x88,0x8A,0x98,0x9A,0xA8,0xAA,0xB8,0xBA,0xC8,
           0xCA,0xD8,0xE8,0xEA,0xF8]:
    INSTR_LENGTH[_op] = 1
for _op in [0x01,0x05,0x06,0x09,0x10,0x11,0x15,0x16,0x21,0x24,0x25,0x26,0x29,
           0x30,0x31,0x35,0x36,0x41,0x45,0x46,0x49,0x50,0x51,0x55,0x56,0x61,
           0x65,0x66,0x69,0x70,0x71,0x75,0x76,0x81,0x84,0x85,0x86,0x90,0x91,
           0x94,0x95,0x96,0xA0,0xA1,0xA2,0xA4,0xA5,0xA6,0xA9,0xB0,0xB1,0xB4,
           0xB5,0xB6,0xC0,0xC1,0xC4,0xC5,0xC6,0xC9,0xD0,0xD1,0xD5,0xD6,0xE0,
           0xE1,0xE4,0xE5,0xE6,0xE9,0xEB,0xF0,0xF1,0xF5,0xF6]:
    INSTR_LENGTH[_op] = 2
for _op in [0x0D,0x0E,0x19,0x1D,0x1E,0x20,0x2C,0x2D,0x2E,0x39,0x3D,0x3E,0x4C,
           0x4D,0x4E,0x59,0x5D,0x5E,0x6C,0x6D,0x6E,0x79,0x7D,0x7E,0x8C,0x8D,
           0x8E,0x99,0x9D,0xAC,0xAD,0xAE,0xB9,0xBC,0xBD,0xBE,0xCC,0xCD,0xCE,
           0xD9,0xDD,0xDE,0xEC,0xED,0xEE,0xF9,0xFD,0xFE]:
    INSTR_LENGTH[_op] = 3

# Opcodes safe for single-instruction testing (implied + immediate)
TEST_OPCODES = [
    (0xAA, "TAX", 1), (0xA8, "TAY", 1), (0x8A, "TXA", 1), (0x98, "TYA", 1),
    (0xBA, "TSX", 1), (0x9A, "TXS", 1),
    (0xE8, "INX", 1), (0xC8, "INY", 1), (0xCA, "DEX", 1), (0x88, "DEY", 1),
    (0x18, "CLC", 1), (0x38, "SEC", 1), (0x58, "CLI", 1), (0x78, "SEI", 1),
    (0xD8, "CLD", 1), (0xF8, "SED", 1), (0xB8, "CLV", 1),
    (0x0A, "ASL_A", 1), (0x4A, "LSR_A", 1), (0x2A, "ROL_A", 1), (0x6A, "ROR_A", 1),
    (0x48, "PHA", 1), (0x68, "PLA", 1), (0x08, "PHP", 1), (0x28, "PLP", 1),
    (0xEA, "NOP", 1),
    (0x69, "ADC_imm", 2), (0xE9, "SBC_imm", 2),
    (0x29, "AND_imm", 2), (0x09, "ORA_imm", 2), (0x49, "EOR_imm", 2),
    (0xA9, "LDA_imm", 2), (0xA2, "LDX_imm", 2), (0xA0, "LDY_imm", 2),
    (0xC9, "CMP_imm", 2), (0xE0, "CPX_imm", 2), (0xC0, "CPY_imm", 2),
]

# Schema / maps

ALL_VARS = ["A_in","X_in","Y_in","S_in","P_in",
            "A_out","X_out","Y_out","S_out","P_out",
            "opcode","operand"]
OUTPUTS  = ["A_out","X_out","Y_out","S_out","P_out"]
INPUTS   = ["A_in","X_in","Y_in","S_in","P_in","opcode","operand"]


def build_schema():
    return MicroVariableSchema([Variable(v, shape=(1,)) for v in ALL_VARS])

def build_cg(schema):
    return CoarseGrainingMap(schema, {v: [v] for v in ALL_VARS})


class RegisterValueMap(ContinuousValueMap):
    """
    ContinuousValueMap for integer register values.

    Overrides ground() to always return a 1-d array of shape (1,),
    not a 0-d scalar. This prevents IndexError in _step_abstract
    when it does val.shape[0] on grounded micro-values.
    """

    def ground(self, name, label, rng=None):
        val = np.atleast_1d(np.asarray(label, dtype=float))
        return val

    def abstract(self, name, val):
        arr = np.asarray(val, dtype=float).ravel()
        if len(arr) == 1:
            return float(arr[0])
        return arr.copy()


def build_vm(cg):
    reg = RectSubspace((0.0, 255.0))
    return RegisterValueMap(cg, {v: {0: reg} for v in ALL_VARS})


# Library loader

_LIB_DIR = os.environ.get(
    "CPU6502_LIB_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_cpu_6502_libs"),
)

def _find_lib(name):
    for d in [_LIB_DIR, os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# Helpers

def _val(d, key, default):
    v = d.get(key, default)
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)):
        return np.array(v, dtype=float)
    return np.array([v], dtype=float)

def _i(arr, i):
    if isinstance(arr, np.ndarray):
        flat = arr.ravel()
        return int(flat[min(i, len(flat) - 1)]) & 0xFF
    if isinstance(arr, (list, tuple)):
        return int(arr[min(i, len(arr) - 1)]) & 0xFF
    return int(arr) & 0xFF

def _bs(interventions):
    for v in interventions.values():
        if isinstance(v, np.ndarray):
            return max(1, v.ravel().shape[0])
        if isinstance(v, (list, tuple)):
            return max(1, len(v))
    return 1


# ISA-level model (fake6502)

class ISASimulator(LowLevelModel):
    """Low-level model M at the ISA level (fake6502): instruction-level execution, no timing."""

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = _find_lib("libisa6502.so")
        if not lib_path:
            raise FileNotFoundError("libisa6502.so not found")
        self._lib = ctypes.CDLL(lib_path)
        self._lib.isa_init()
        self._lib.isa_execute_instruction.argtypes = [
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16),
        ]
        self._lib.isa_execute_instruction.restype = None

    def _exec(self, op, op1, ilen, a, x, y, s, p):
        ao, xo, yo, so, po = (ctypes.c_uint8() for _ in range(5))
        pco = ctypes.c_uint16()
        self._lib.isa_execute_instruction(
            op, op1, 0, ilen, a, x, y, s, p,
            ctypes.byref(ao), ctypes.byref(xo), ctypes.byref(yo),
            ctypes.byref(so), ctypes.byref(po), ctypes.byref(pco))
        return ao.value, xo.value, yo.value, so.value, po.value

    def forward_with_interventions(self, input_state, interventions):
        opcode  = _val(interventions, "opcode",  0xEA)
        operand = _val(interventions, "operand", 0)
        a_in = _val(interventions, "A_in", 0)
        x_in = _val(interventions, "X_in", 0)
        y_in = _val(interventions, "Y_in", 0)
        s_in = _val(interventions, "S_in", 0xFD)
        p_in = _val(interventions, "P_in", 0)
        bs = _bs(interventions)

        R = {k: [] for k in ALL_VARS}
        for i in range(bs):
            opi  = _i(opcode, i);  op1i = _i(operand, i)
            ai = _i(a_in, i); xi = _i(x_in, i); yi = _i(y_in, i)
            si = _i(s_in, i); pi = _i(p_in, i)
            il = INSTR_LENGTH[opi] or 1
            ao, xo, yo, so, po = self._exec(opi, op1i, il, ai, xi, yi, si, pi)
            for k, v in [("opcode",opi),("operand",op1i),
                         ("A_in",ai),("X_in",xi),("Y_in",yi),("S_in",si),("P_in",pi),
                         ("A_out",ao),("X_out",xo),("Y_out",yo),("S_out",so),("P_out",po)]:
                R[k].append(float(v))

        return input_state.merge(SystemState(
            values={k: np.array(v).reshape(-1, 1) for k, v in R.items()}))


# Gate-level model (break6502)

class GateSimulator(LowLevelModel):
    """Low-level model M at the gate level (break6502): decoder PLA + datapath, half-cycle."""

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = _find_lib("libgate6502.so")
        if not lib_path:
            raise FileNotFoundError("libgate6502.so not found")
        # break6502's M6502 constructor reads its ~272 MB decode table
        # (Decoder6502.bin) from the current working directory. Rather than
        # copy that blob into the caller's CWD (non-hermetic, pollutes arbitrary
        # directories, ~272 MB per run), chdir into the directory that holds the
        # single canonical copy (staged next to the shared library by
        # build_libs.sh) for the duration of gate_init() only, then restore the
        # original CWD. See THIRD_PARTY for Decoder6502.bin provenance.
        lib_dir = os.path.dirname(os.path.abspath(lib_path))
        decoder_path = os.path.join(lib_dir, "Decoder6502.bin")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(
                f"Decoder6502.bin not found next to libgate6502.so (expected at "
                f"{decoder_path}). It is staged by "
                f"systems/10_cpu_6502_libs/build_libs.sh; see THIRD_PARTY for its provenance."
            )

        self._lib = ctypes.CDLL(lib_path)
        _prev_cwd = os.getcwd()
        try:
            os.chdir(lib_dir)
            self._lib.gate_init()
        finally:
            os.chdir(_prev_cwd)
        self._lib.gate_execute_instruction.argtypes = [
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._lib.gate_execute_instruction.restype = None

    def _exec(self, op, op1, ilen, a, x, y, s, p):
        ao, xo, yo, so, po = (ctypes.c_uint8() for _ in range(5))
        self._lib.gate_execute_instruction(
            op, op1, 0, ilen, a, x, y, s, p,
            ctypes.byref(ao), ctypes.byref(xo), ctypes.byref(yo),
            ctypes.byref(so), ctypes.byref(po))
        return ao.value, xo.value, yo.value, so.value, po.value

    def forward_with_interventions(self, input_state, interventions):
        opcode  = _val(interventions, "opcode",  0xEA)
        operand = _val(interventions, "operand", 0)
        a_in = _val(interventions, "A_in", 0)
        x_in = _val(interventions, "X_in", 0)
        y_in = _val(interventions, "Y_in", 0)
        s_in = _val(interventions, "S_in", 0xFD)
        p_in = _val(interventions, "P_in", 0)
        bs = _bs(interventions)

        R = {k: [] for k in ALL_VARS}
        for i in range(bs):
            opi  = _i(opcode, i); op1i = _i(operand, i)
            ai = _i(a_in, i); xi = _i(x_in, i); yi = _i(y_in, i)
            si = _i(s_in, i); pi = _i(p_in, i)
            il = INSTR_LENGTH[opi] or 1
            ao, xo, yo, so, po = self._exec(opi, op1i, il, ai, xi, yi, si, pi)
            for k, v in [("opcode",opi),("operand",op1i),
                         ("A_in",ai),("X_in",xi),("Y_in",yi),("S_in",si),("P_in",pi),
                         ("A_out",ao),("X_out",xo),("Y_out",yo),("S_out",so),("P_out",po)]:
                R[k].append(float(v))

        return input_state.merge(SystemState(
            values={k: np.array(v).reshape(-1, 1) for k, v in R.items()}))


# Transistor-level model (perfect6502)

class TransistorSimulator(LowLevelModel):
    """Low-level model M at the transistor level (perfect6502): ~3500 transistors, half-cycle."""

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = _find_lib("libtransistor6502.so")
        if not lib_path:
            raise FileNotFoundError("libtransistor6502.so not found")
        self._lib = ctypes.CDLL(lib_path)
        self._lib.transistor_init()
        self._lib.transistor_execute_instruction.argtypes = [
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._lib.transistor_execute_instruction.restype = None

    def _exec(self, op, op1, ilen, a, x, y, s, p):
        ao, xo, yo, so, po = (ctypes.c_uint8() for _ in range(5))
        self._lib.transistor_execute_instruction(
            op, op1, 0, ilen, a, x, y, s, p,
            ctypes.byref(ao), ctypes.byref(xo), ctypes.byref(yo),
            ctypes.byref(so), ctypes.byref(po))
        return ao.value, xo.value, yo.value, so.value, po.value

    def forward_with_interventions(self, input_state, interventions):
        opcode  = _val(interventions, "opcode",  0xEA)
        operand = _val(interventions, "operand", 0)
        a_in = _val(interventions, "A_in", 0)
        x_in = _val(interventions, "X_in", 0)
        y_in = _val(interventions, "Y_in", 0)
        s_in = _val(interventions, "S_in", 0xFD)
        p_in = _val(interventions, "P_in", 0)
        bs = _bs(interventions)

        R = {k: [] for k in ALL_VARS}
        for i in range(bs):
            opi  = _i(opcode, i); op1i = _i(operand, i)
            ai = _i(a_in, i); xi = _i(x_in, i); yi = _i(y_in, i)
            si = _i(s_in, i); pi = _i(p_in, i)
            il = INSTR_LENGTH[opi] or 1
            ao, xo, yo, so, po = self._exec(opi, op1i, il, ai, xi, yi, si, pi)
            for k, v in [("opcode",opi),("operand",op1i),
                         ("A_in",ai),("X_in",xi),("Y_in",yi),("S_in",si),("P_in",pi),
                         ("A_out",ao),("X_out",xo),("Y_out",yo),("S_out",so),("P_out",po)]:
                R[k].append(float(v))

        return input_state.merge(SystemState(
            values={k: np.array(v).reshape(-1, 1) for k, v in R.items()}))


# Broken gate simulator

class StuckA7Simulator(LowLevelModel):
    """Wraps any simulator with A[7] stuck at 0 (bit 7 cleared)."""
    def __init__(self, base: LowLevelModel):
        self.base = base

    def forward_with_interventions(self, input_state, interventions):
        result = self.base.forward_with_interventions(input_state, interventions)
        if "A_out" in result.values:
            arr = result.values["A_out"].copy().astype(float)
            arr = np.where(arr >= 128.0, arr - 128.0, arr)
            result.values["A_out"] = arr
        return result

# Backward compat
BrokenGateSimulator = StuckA7Simulator


# high-level model builders

def _make_high_level_model_from_sim(sim, label=""):
    high_level_model = CausalGraph()
    reg = RectSubspace((0.0, 255.0))
    for v in INPUTS:
        high_level_model.add_variable(v, lambda: None, domain=reg)

    parents = list(INPUTS)

    def _eq(idx):
        def fn(opcode, operand, A_in, X_in, Y_in, S_in, P_in):
            op  = int(opcode)  & 0xFF
            op1 = int(operand) & 0xFF
            il  = INSTR_LENGTH[op] or 1
            r = sim._exec(op, op1, il,
                          int(A_in) & 0xFF, int(X_in) & 0xFF, int(Y_in) & 0xFF,
                          int(S_in) & 0xFF, int(P_in) & 0xFF)
            return float(r[idx])
        return fn

    for i, name in enumerate(OUTPUTS):
        high_level_model.add_variable(name, _eq(i), parents=parents, domain=reg)
    return high_level_model


def build_isa_high_level_model(isa_sim):   return _make_high_level_model_from_sim(isa_sim, "ISA")
def build_gate_high_level_model(gate_sim): return _make_high_level_model_from_sim(gate_sim, "Gate")


# Sampler

class InstructionSampler(InterventionSampler):
    """Sampler that draws 6502 opcodes and operands as interventions."""

    def __init__(self, value_map, opcodes=None):
        super().__init__(value_map)
        self.opcodes = opcodes or TEST_OPCODES

    def sample_intervention(self, variables, batch_size=1, max_interventions=None,
                            force_all=False, rng=None):
        rng = self._get_rng(rng)
        vnames = {v.name for v in variables}
        spec = {}

        idx = rng.integers(0, len(self.opcodes))
        op_code, _, op_len = self.opcodes[idx]

        if "opcode" in vnames:
            spec["opcode"] = {
                "labels": [float(op_code)] * batch_size,
                "micro_values": None,
            }

        if "operand" in vnames:
            if op_len >= 2:
                ops = [float(rng.integers(0, 256)) for _ in range(batch_size)]
            else:
                ops = [0.0] * batch_size
            spec["operand"] = {"labels": ops, "micro_values": None}

        for reg in ["A_in", "X_in", "Y_in", "S_in"]:
            if reg in vnames:
                spec[reg] = {
                    "labels": [float(rng.integers(0, 256)) for _ in range(batch_size)],
                    "micro_values": None,
                }

        if "P_in" in vnames:
            spec["P_in"] = {
                "labels": [float((int(rng.integers(0, 256)) & P_MASK) | 0x20)
                           for _ in range(batch_size)],
                "micro_values": None,
            }

        return spec


# Verification

def verify_bridges(sim_a, sim_b, label="", n_tests=200, seed=42):
    rng = np.random.default_rng(seed)
    matches = total = 0
    failures = []

    for _ in range(n_tests):
        idx = rng.integers(0, len(TEST_OPCODES))
        opcode, name, ilen = TEST_OPCODES[idx]
        op1 = int(rng.integers(0, 256))
        ai  = int(rng.integers(0, 256))
        xi  = int(rng.integers(0, 256))
        yi  = int(rng.integers(0, 256))
        si  = int(rng.integers(0, 256))
        pi  = (int(rng.integers(0, 256)) & P_MASK) | 0x20

        try:
            r_a = sim_a._exec(opcode, op1, ilen, ai, xi, yi, si, pi)
            r_b = sim_b._exec(opcode, op1, ilen, ai, xi, yi, si, pi)
        except Exception as e:
            failures.append((name, f"Exception: {e}"))
            total += 1
            continue

        ok = r_a == r_b
        if ok:
            matches += 1
        else:
            failures.append((name, f"A:{ai:02X} X:{xi:02X} Y:{yi:02X} S:{si:02X} P:{pi:02X} | "
                                   f"a={tuple(f'{v:02X}' for v in r_a)} "
                                   f"b={tuple(f'{v:02X}' for v in r_b)}"))
        total += 1

    pct = 100.0 * matches / total if total else 0.0
    status = "PASS" if matches == total else "WARN"
    print(f"[{status}] {label}: {matches}/{total} ({pct:.1f}%)")

    if failures:
        for name, detail in failures[:10]:
            print(f"  FAIL {name}: {detail}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return matches, total, failures


def main():
    print("6502 Bridge Smoke Test\n")
    isa = ISASimulator()
    gate = GateSimulator()
    verify_bridges(isa, gate, label="ISA vs Gate", n_tests=500, seed=42)

    try:
        transistor = TransistorSimulator()
        print()
        verify_bridges(gate, transistor, label="Gate vs Transistor", n_tests=100, seed=42)
    except FileNotFoundError:
        print("\n(libtransistor6502.so not found)")


if __name__ == "__main__":
    main()
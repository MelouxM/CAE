"""
Shared pytest fixtures for the ``causal_abstraction`` unit tests.

These unit tests live alongside the runnable evaluation suites in ``test/`` but
are collected separately via the ``test_*.py`` naming
convention configured in ``pyproject.toml`` ([tool.pytest.ini_options]), so the
``NN_<system>.py`` / ``power_*.py`` suites and figure generators are never
picked up as unit tests.
"""
import os
import sys

import pytest

# Make the shared test helpers importable; importing ``utils`` then puts the repo
# root and ``systems/`` on sys.path (mirrors the eval suites).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import load_system


def _load_logic_circuit():
    """Load ``systems/01_logic_circuit.py`` by path (mirrors the eval suites)."""
    return load_system("01_logic_circuit.py", "logic_circuit_sys")


@pytest.fixture(scope="session")
def logic_system():
    """A small, fast, deterministic ``(M, E, tau)`` triple: the 2-bit adder.

    Reused as a fixture for the sampler and engine tests so they exercise the
    real schema / coarse-graining map / value map rather than mocks. The objects
    are not mutated by evaluation, so a single session-scoped instance is safe.
    """
    from causal_abstraction import MicroVariableSchema

    lc = _load_logic_circuit()
    gates, all_wires = lc.build_2bit_adder()
    schema = MicroVariableSchema.from_names(all_wires)
    low_level = lc.NetlistSimulator(gates, all_wires)
    cg, vm = lc.build_cg_and_vm(schema)
    return {
        "module": lc,
        "schema": schema,
        "low_level": low_level,
        "cg": cg,
        "vm": vm,
        "valid": lc.build_valid_high_level_model(),
        "failing": lc.build_failing_high_level_model(),
    }

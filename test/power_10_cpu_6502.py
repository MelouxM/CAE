"""
Statistical power / convergence study for the MOS 6502 three-level abstraction.
"""
from pathlib import Path

from utils import load_system

_cpu = load_system("10_cpu_6502.py", "cpu_6502")

from causal_abstraction import EvaluationConfig, EvaluationEngine, BottomUpSampler
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = Path(__file__).parent / "results" / "power_10_cpu_6502.json"
N_GRID       = [2, 4, 6, 8, 10, 15, 20, 30]
TARGET_RUNS  = 50
CONDITIONS   = ["gate_vs_isa", "transistor_vs_gate", "broken_gate_vs_isa"]
INPUTS       = _cpu.INPUTS

# Shared objects (lazy init, verified once)

_SCHEMA   = _cpu.build_schema()
_CG       = _cpu.build_cg(_SCHEMA)
_VM       = _cpu.build_vm(_CG)
_ISA      = None
_GATE     = None
_TRANS    = None
_VERIFIED = False


def _isa():
    global _ISA
    if _ISA is None:
        _ISA = _cpu.ISASimulator()
    return _ISA


def _gate():
    global _GATE
    if _GATE is None:
        _GATE = _cpu.GateSimulator()
    return _GATE


def _transistor():
    global _TRANS
    if _TRANS is None:
        try:
            _TRANS = _cpu.TransistorSimulator()
        except FileNotFoundError:
            print("WARNING: libtransistor6502.so not found; using gate sim as proxy.")
            _TRANS = _gate()
    return _TRANS


def _verify_once():
    global _VERIFIED
    if _VERIFIED:
        return
    print("\n--> Bridge verification")
    _cpu.verify_bridges(_isa(), _gate(), "ISA vs Gate", n_tests=300, seed=123)
    if _transistor() is not _gate():
        _cpu.verify_bridges(_gate(), _transistor(), "Gate vs Transistor",
                            n_tests=100, seed=456)
    print("--> End verification\n")
    _VERIFIED = True


def _build_tasks(builder):
    s  = _cpu.InstructionSampler(_VM)
    bu = BottomUpSampler(_VM)
    return [
        StandardTasks.score(builder, sampler=s,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=s,  name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    _verify_once()
    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg  = EvaluationConfig(metric="mse", seed=seed)
    if cond == "gate_vs_isa":
        high_level_model = _cpu.build_isa_high_level_model(_isa())
        low_level_model = _gate()
    elif cond == "transistor_vs_gate":
        high_level_model = _cpu.build_gate_high_level_model(_gate())
        low_level_model = _transistor()
    else:
        high_level_model = _cpu.build_isa_high_level_model(_isa())
        low_level_model = _cpu.BrokenGateSimulator(_gate())
    engine = EvaluationEngine(high_level_model, low_level_model, _VM, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder),
        analytical=[],
        sampler=_cpu.InstructionSampler(_VM),
        task_kwargs=dict(batch_size=1, max_interventions=len(INPUTS),
                         intervention_domain=INPUTS),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="MOS 6502 three-level abstraction - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

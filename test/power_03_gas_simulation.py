"""
Statistical power / convergence study for the Lennard-Jones gas.
A fresh UniversalGasModel is created per build_condition call to prevent
stale particle state leaking between conditions.
"""
import os
from pathlib import Path

from utils import load_system

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = Path(__file__).parent

os.chdir(os.path.join(_REPO_ROOT, "systems"))

_gs = load_system("03_gas_simulation.py", "gas_simulation")

# Import helpers from test module (runs calibration once, uses disk cache).
_gas_test = load_system(TEST_DIR / "03_gas_simulation.py", "gas_test")

from causal_abstraction import EvaluationConfig, EvaluationEngine
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, parse_power_args
from runner import Condition

# Config

RESULTS_FILE = TEST_DIR / "results" / "power_03_gas_simulation.json"
N_GRID       = [2, 4, 6, 8, 12, 20]
TARGET_RUNS  = 20
CONDITIONS   = _gas_test.CONDITIONS
COND_SPECS   = _gas_test.COND_SPECS
_CG          = _gas_test._CG
_A_VDW       = _gas_test._A_VDW
_B_VDW       = _gas_test._B_VDW

COND_SPECS = {
    "valid_ideal":     ((_gas_test.LOW_RHO, _gas_test.MED_TEMP),  _gas_test.ALPHA_VALID, False),
    "valid_vdw":       ((_gas_test.LOW_RHO, _gas_test.MED_TEMP),  None,        True),
    "invalid_alpha":   ((_gas_test.LOW_RHO, _gas_test.MED_TEMP),  _gas_test.ALPHA_WRONG, False),
    "invalid_density": ((0.25,    _gas_test.MED_TEMP),  _gas_test.ALPHA_VALID, False),  # was 0.70
    "invalid_temp":    ((_gas_test.LOW_RHO, 1.8),       _gas_test.ALPHA_VALID, False),  # was 1.0
}


def _build_tasks(builder, vm, bu, high_level_model):
    from causal_abstraction.tasks import StandardTasks
    td = _gs.ThermodynamicSampler(vm)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
    ]


def build_condition(cond: str, run_index: int) -> Condition:
    (rho, t_center), alpha, use_vdw = COND_SPECS[cond]
    v_sp, t_sp, p_sp = _gas_test._build_domain(rho, t_center)
    vm  = _gas_test._build_vm(_CG, v_sp, t_sp, p_sp)
    high_level_model = (_gas_test._make_vdw_high_level_model(_A_VDW, _B_VDW, v_sp, t_sp) if use_vdw
           else _gas_test._make_ideal_high_level_model(alpha, v_sp, t_sp))
    low_level_model = _gs.UniversalGasModel()  # fresh per call
    bu  = _gas_test.GasBottomUpSampler(
        vm,
        rho_range=(rho * 0.7, rho * 1.3),
        t_range=(t_center * 0.7, t_center * 1.3),
    )
    seed   = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg    = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, _CG, cfg)
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, vm, bu, high_level_model),
        analytical=[],
        sampler=bu,
        task_kwargs=dict(batch_size=1, max_interventions=2),
    )


if __name__ == "__main__":
    args = parse_power_args()
    run_power_suite(
        title="Gas simulation (LJ fluid vs EOS) - power/convergence",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        build_condition=build_condition,
        n_grid=N_GRID,
        target_runs=TARGET_RUNS,
        args=args,
    )

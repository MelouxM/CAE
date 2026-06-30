"""
Statistical power / convergence study for the six controlled experiments.
One results file per experiment; can be run individually or all together.

Usage:
    python power_11_controlled.py                  # all experiments
    python power_11_controlled.py --experiment 2   # single experiment
    python power_11_controlled.py --runs 50        # stop after 50 rounds
"""
import argparse
from pathlib import Path

from utils import load_system

TEST_DIR = Path(__file__).parent

_ctrl = load_system(TEST_DIR / "11_controlled_experiments.py", "controlled_test")

from causal_abstraction import BottomUpSampler
from causal_abstraction.tasks import StandardTasks
from power_runner import run_power_suite, POWER_METRICS
from runner import Condition

# n grids: exp 1 uses a real ABM, rest are synthetic

N_GRID_MEDIUM = [2, 4, 6, 8, 10, 15, 20, 30]
N_GRID_FAST   = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TARGET_RUNS   = 100


def _add_cae_up_faith(condition: Condition) -> Condition:
    """Inject CAE_up (bottom-up + faithfulness) into an existing Condition."""
    bu       = BottomUpSampler(condition.engine.value_map)
    cae_up_faith = StandardTasks.score(
        condition.engine.builder, sampler=bu, name="CAE_up",
        include_faithfulness=True)
    return Condition(
        engine=condition.engine,
        tasks=condition.tasks + [cae_up_faith],
        analytical=condition.analytical,
        sampler=condition.sampler,
        task_kwargs=condition.task_kwargs,
        analytical_n_samples=condition.analytical_n_samples,
    )


def _wrap(build_fn):
    def _build(cond, run_idx):
        return _add_cae_up_faith(build_fn(cond, run_idx))
    return _build


# Experiment registry

POWER_EXPERIMENTS = {
    1: dict(
        title="Exp 1 - Hidden Environmental Confounder (predator-prey)",
        results_file=TEST_DIR / "results" / "power_11_exp1_hidden_confounder.json",
        conditions=["valid", "confounder"],
        build_condition=_wrap(_ctrl.build_exp1),
        n_grid=N_GRID_MEDIUM,
    ),
    2: dict(
        title="Exp 2 - XOR-Masked Redundant Pathway",
        results_file=TEST_DIR / "results" / "power_11_exp2_xor_backup.json",
        conditions=["valid", "xor_leak"],
        build_condition=_wrap(_ctrl.build_exp2),
        n_grid=[2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500],
    ),
    3: dict(
        title="Exp 3 - Wrong Intermediate Representation",
        results_file=TEST_DIR / "results" / "power_11_exp3_wrong_intermediate.json",
        conditions=["valid", "wrong_intermediate"],
        build_condition=_wrap(_ctrl.build_exp3),
        n_grid=N_GRID_FAST,
    ),
    4: dict(
        title="Exp 4 - Spurious Mediator (fork vs chain)",
        results_file=TEST_DIR / "results" / "power_11_exp4_spurious_mediator.json",
        conditions=["valid_fork", "spurious_chain"],
        build_condition=_wrap(_ctrl.build_exp4),
        n_grid=N_GRID_FAST,
    ),
    5: dict(
        title="Exp 5 - Unreachable Intermediate States",
        results_file=TEST_DIR / "results" / "power_11_exp5_unreachable_states.json",
        conditions=["valid", "wrong_threshold"],
        build_condition=_wrap(_ctrl.build_exp5),
        n_grid=N_GRID_FAST,
    ),
    6: dict(
        title="Exp 6 - Wrong Causal Direction (chain vs fork)",
        results_file=TEST_DIR / "results" / "power_11_exp6_wrong_direction.json",
        conditions=["valid_chain", "wrong_fork"],
        build_condition=_wrap(_ctrl.build_exp6),
        n_grid=N_GRID_FAST,
    ),
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Power/convergence experiments for controlled abstractions")
    p.add_argument("--experiment", nargs="+", type=int, default=None,
                   help="Experiment numbers to run (1-6). Default: all.")
    p.add_argument("--runs", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    exp_ids = args.experiment or sorted(POWER_EXPERIMENTS.keys())

    for eid in exp_ids:
        if eid not in POWER_EXPERIMENTS:
            print(f"Unknown experiment {eid}. Valid: {sorted(POWER_EXPERIMENTS.keys())}")
            continue
        exp = POWER_EXPERIMENTS[eid]
        run_power_suite(
            title=exp["title"],
            results_file=exp["results_file"],
            conditions=exp["conditions"],
            build_condition=exp["build_condition"],
            n_grid=exp["n_grid"],
            target_runs=TARGET_RUNS,
            args=args,
        )

"""
Controlled experiments for metric failure mode analysis.

Six experiments, each targeting a specific gap that the CAE metric
addresses but that observational and/or existing causal metrics miss.

Exp 1 - Hidden confounder          (faithfulness / Gap 3)
Exp 2 - XOR backup pathway         (faithfulness / Gap 3)
Exp 3 - Wrong intermediate repr.   (intermediate checking / Gap 5)
Exp 4 - Spurious mediator          (domain-wide + intermediate / Gaps 2,5)
Exp 5 - Interaction effects        (wrong high-level model structure, calibrated marginals)
Exp 6 - Wrong causal direction     (intermediate checking / Gap 5)

Usage:
    python test/11_controlled_experiments.py                       # all, run forever
    python test/11_controlled_experiments.py --experiment 1        # only exp 1
    python test/11_controlled_experiments.py --experiment 1 4 6    # subset
    python test/11_controlled_experiments.py --runs 10             # 10 runs then stop
    python test/11_controlled_experiments.py --experiment 2 --metrics CAE_down IIAMetric
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils import load_system

TEST_DIR = Path(__file__).parent


# Lazy system loaders
_pp = None
_ctrl = None


def _get_pp():
    global _pp
    if _pp is None:
        _pp = load_system("04_predator_prey.py", "predator_prey")
    return _pp


def _get_ctrl():
    global _ctrl
    if _ctrl is None:
        _ctrl = load_system("11_controlled_experiments.py", "controlled")
    return _ctrl


# Library imports
from causal_abstraction import (
    CausalGraph, CoarseGrainingMap, ContinuousValueMap,
    EvaluationConfig, EvaluationEngine,
    FullSubspace, MicroVariableSchema, RectSubspace,
    TopDownSampler, L2Metric, VarianceDecompositionMetric,
    IBLagrangianMetric, CIBLagrangianMetric,
    ComplexityShiftMetric, SobolSensitivityMetric,
    IIAMetric, BCCMetric, ProbingMetric, InfidelityMetric,
    SymbionMetric, MMDMetric, ConditionalIndependenceMetric,
    RelationalFidelityMetric, MacroscopicInvarianceMetric,
    RMSEMetric, BottomUpSampler,
)
from causal_abstraction.schema import Variable
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite


# Shared metric builders

def _continuous_tasks(builder, sampler, out):
    td = sampler
    bu = BottomUpSampler(sampler.value_map)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
        StandardTasks.observational_r2(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_mse(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_kl(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_jsd(builder, sampler=bu, output_vars=out),
        StandardTasks.observational(builder, RMSEMetric(), sampler=bu,
                                    output_vars=out, name="RMSE"),
        StandardTasks.observational(builder, L2Metric(), sampler=bu,
                                    output_vars=out, name="L2"),
        StandardTasks.observational(builder, MMDMetric(), sampler=bu,
                                    output_vars=out, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(),
                                    sampler=bu, output_vars=out, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),
                                    sampler=bu, output_vars=out, name="VarDecomp"),
    ]


def _continuous_analytical(out, n):
    return [
        IIAMetric(inner_metric="l2", output_vars=out, n_pairs=n),
        BCCMetric(n_pairs=n, inner_metric="l2"),
        MacroscopicInvarianceMetric(n_pairs=min(n, 10), inner_metric="l2"),
        RelationalFidelityMetric(output_vars=out, n_pairs=max(n, 50)),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        ComplexityShiftMetric(output_vars=out),
        SobolSensitivityMetric(n_samples=n, output_vars=out),
        InfidelityMetric(output_vars=out),
        ProbingMetric(n_train=max(n * 4, 80), inner_metric="l2"),
    ]


def _discrete_tasks(builder, sampler, out):
    td = sampler
    bu = BottomUpSampler(sampler.value_map)
    return [
        StandardTasks.score(builder, sampler=td, name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu, name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down",
                            include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up",
                            include_faithfulness=True),
        StandardTasks.observational_r2(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_mse(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_nmse(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_kl(builder, sampler=bu, output_vars=out),
        StandardTasks.observational_jsd(builder, sampler=bu, output_vars=out),
        StandardTasks.observational(builder, MMDMetric(), sampler=bu,
                                    output_vars=out, name="MMD"),
        StandardTasks.observational(builder, ConditionalIndependenceMetric(),
                                    sampler=bu, output_vars=out, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),
                                    sampler=bu, output_vars=out, name="VarDecomp")
    ]


def _discrete_analytical(out, n):
    return [
        IIAMetric(inner_metric="mse", output_vars=out, n_pairs=n),
        BCCMetric(n_pairs=n, inner_metric="mse"),
        MacroscopicInvarianceMetric(n_pairs=min(n, 10), inner_metric="mse"),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        ComplexityShiftMetric(output_vars=out),
        SobolSensitivityMetric(n_samples=n, output_vars=out),
        InfidelityMetric(output_vars=out),
        SymbionMetric(output_vars=out),
        ProbingMetric(n_train=n, inner_metric="mse"),
        RelationalFidelityMetric(output_vars=out),
    ]


#  EXPERIMENT 1 - Hidden Environmental Confounder
#
# Gap tested: Faithfulness (Gap 3).
# A resource variable modulates prey survival but is unmapped (phi).
# Under normal operation (resource=0), the system behaves identically
# to standard PP.  Under phi noise, the resource shifts prey counts.
#
# Expected: all non-faithfulness metrics see valid ~ confounder.
#           CAE_down detects the confounder.

_EXP1 = dict(
    file=TEST_DIR / "results" / "11_exp1_hidden_confounder_results.json",
    n=50, outputs=["final_populations"],
    conditions=["valid", "confounder"],
    comparisons=[("valid", "confounder", "valid -> confounder")],
)

_exp1_objects = None


def _exp1_init():
    global _exp1_objects
    if _exp1_objects is not None:
        return _exp1_objects
    pp = _get_pp()

    base_cfg = dict(prey_reproduce_prob=0.1, predation_prob=0.001,
                    predator_reproduce_prob=0.5, predator_starvation_prob=0.05,
                    num_steps=50, grid_size=20,
                    prey_age_death_prob=0.02, predator_age_death_prob=0.04,
                    spatial=False, stochastic_reproduction=False, agent_aging=False)
    base_abm = pp.AgentBasedModel(base_cfg)

    search = dict(alpha=np.linspace(0.08, 0.12, 3),
                  beta=np.linspace(0.0008, 0.0012, 3),
                  delta=np.linspace(0.0004, 0.0006, 3),
                  gamma=np.linspace(0.04, 0.06, 3))
    high_level_model_params = pp.calibrate_high_level_model(base_abm, search)

    prey_dom = RectSubspace((50, 250))
    pred_dom = RectSubspace((20, 100))
    out_dom  = FullSubspace(2)

    schema = MicroVariableSchema([
        Variable("prey_t"), Variable("predator_t"),
        Variable("resource_level"), Variable("final_populations"),
    ])
    cg = CoarseGrainingMap(schema, {
        "prey_t": ["prey_t"], "predator_t": ["predator_t"],
        "final_populations": ["final_populations"],
    })
    vm = ContinuousValueMap(cg, {
        "prey_t": {0: prey_dom}, "predator_t": {0: pred_dom},
        "final_populations": {0: out_dom},
    })

    _exp1_objects = dict(pp=pp, base_abm=base_abm, high_level_model_params=high_level_model_params,
                         prey_dom=prey_dom, pred_dom=pred_dom, out_dom=out_dom,
                         schema=schema, cg=cg, vm=vm)
    return _exp1_objects


def build_exp1(cond, run_index):
    o = _exp1_init()
    pp, ctrl = _get_pp(), _get_ctrl()
    e = _EXP1

    effect = 0.0 if cond == "valid" else 0.5
    low_level_model = ctrl.ConfounderPPModel(o["base_abm"], resource_effect=effect)

    high_level_model = CausalGraph()
    high_level_model.add_variable("prey_t", lambda: None, domain=o["prey_dom"])
    high_level_model.add_variable("predator_t", lambda: None, domain=o["pred_dom"])
    high_level_model.add_variable("final_populations",
                     lambda prey_t, predator_t: pp.solve_lotka_volterra(
                         prey_t, predator_t, o["high_level_model_params"]),
                     parents=["prey_t", "predator_t"], domain=o["out_dom"])

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed, phi_noise_std=1.0)
    engine = EvaluationEngine(high_level_model, low_level_model, o["vm"], o["cg"], cfg)
    sampler = TopDownSampler(o["vm"])

    return Condition(
        engine=engine,
        tasks=_continuous_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_continuous_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=2,
                         intervention_domain=["prey_t", "predator_t"]),
    )


#  EXPERIMENT 2 - XOR-Masked Redundant Pathway
#
# Gap tested: Faithfulness (Gap 3).
# Two phi variables (b1, b2) are harmless when equal but activate
# a dormant OR-path when independently perturbed.
#
# Expected: every non-faithfulness metric gives identical scores.
#           Only CAE_down detects the leak.

_EXP2 = dict(
    file=TEST_DIR / "results" / "11_exp2_xor_backup_results.json",
    n=500, outputs=["Y"],
    conditions=["valid", "xor_leak"],
    comparisons=[("valid", "xor_leak", "valid -> xor_leak")],
)


def build_exp2(cond, run_index):
    ctrl = _get_ctrl()
    e = _EXP2

    schema = ctrl.build_xor_schema()
    cg = ctrl.build_xor_cg(schema)
    vm = ctrl.build_xor_vm(cg)
    high_level_model = ctrl.build_xor_high_level_model()
    low_level_model = ctrl.XORBackupLowLevelModel(xor_active=(cond == "xor_leak"))

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed, phi_noise_std=1.0)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    sampler = TopDownSampler(vm)

    return Condition(
        engine=engine,
        tasks=_discrete_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_discrete_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=4),
    )


#  EXPERIMENT 3 - Wrong Intermediate Representation
#
# Gap tested: Intermediate variable checking (Gap 5).
#
# low-level model: x -> m = 2x -> z = m + 3     (chain, z = 2x + 3)
# high-level model valid:   same equations
# high-level model wrong:   m = 3x, z = (2/3)m + 3   (z = 2x + 3 still correct)
#
# Both high-level models produce identical z for any x.
# The intermediate m is wrong (3x vs 2x).
#
# IIA also misses this because the compensating z equation
# produces matching counterfactual outputs when m is interchanged.
# Only CAE_down_nf/CAE_up_nf detect it by directly checking m's value.
#
# Expected:
#   Observational metrics (output_vars=["z"]): no difference
#   IIA: no difference (compensating equations mask the error)

_EXP3 = dict(
    file=TEST_DIR / "results" / "11_exp3_wrong_intermediate_results.json",
    n=200, outputs=["z"],
    conditions=["valid", "wrong_intermediate"],
    comparisons=[("valid", "wrong_intermediate",
                  "valid -> wrong intermediate")],
)


def build_exp3(cond, run_index):
    ctrl = _get_ctrl()
    e = _EXP3

    low_level_model = ctrl.IntermediateChainLowLevelModel()
    schema = ctrl.build_intermediate_schema()
    cg = ctrl.build_intermediate_cg(schema)
    vm = ctrl.build_intermediate_vm(cg)

    if cond == "valid":
        high_level_model = ctrl.build_valid_intermediate_high_level_model()
    else:
        high_level_model = ctrl.build_wrong_intermediate_high_level_model()

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    sampler = TopDownSampler(vm)

    return Condition(
        engine=engine,
        tasks=_continuous_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_continuous_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        # Intervene on x and m to test both root and intermediate
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=2,
                         intervention_domain=["x", "m"]),
    )


#  EXPERIMENT 4 - Spurious Mediator
#
# Gap tested: Domain-wide interventions + intermediate checking (Gaps 2, 5).
# low-level model is a fork: x -> y, x -> z independently.
# Wrong high-level model claims a chain: x -> y -> z.
# Under observation both produce identical z = 3x + 2.
# Under intervention on y, chain high-level model predicts z = 1.5y + 0.5,
# but the low-level model's z is always 3x + 2 regardless of y.
#
# Expected: observational metrics see no difference.
#           CAE_down_nf/CAE_up_nf detect mismatch at z under y-interventions.

_EXP4 = dict(
    file=TEST_DIR / "results" / "11_exp4_spurious_mediator_results.json",
    n=200, outputs=["z"],
    conditions=["valid_fork", "spurious_chain"],
    comparisons=[("valid_fork", "spurious_chain", "fork -> chain")],
)


def build_exp4(cond, run_index):
    ctrl = _get_ctrl()
    e = _EXP4

    low_level_model = ctrl.ForkLowLevelModel()
    schema = ctrl.build_fork_schema()
    cg = ctrl.build_fork_cg(schema)
    vm = ctrl.build_fork_vm(cg)

    high_level_model = ctrl.build_fork_high_level_model() if cond == "valid_fork" else ctrl.build_chain_high_level_model()

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    sampler = TopDownSampler(vm)

    return Condition(
        engine=engine,
        tasks=_continuous_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_continuous_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=2,
                         intervention_domain=["x", "y"]),
    )


#  EXPERIMENT 5 - Unreachable Intermediate States
#
# Gap tested: Reachability-limited interventions (Gap 2).
#
# low-level model: x in {0,1} -> m = 2x in {0,2} -> z = (m >= 2) ? 1 : 0
# M has domain {0, 1, 2, 3} but only {0, 2} are input-reachable.
#
# Valid high-level model:  Z = (M >= 2) ? 1 : 0    M=0->0, M=1->0, M=2->1, M=3->1
# Wrong high-level model:  Z = (M >= 1) ? 1 : 0    M=0->0, M=1->1!, M=2->1, M=3->1
#
# Both high-level models agree on input-reachable M in {0, 2}.
# They disagree only at M=1 (unreachable from any input).
#
# IIA: only swaps M values produced by inputs -> tests M in {0,2} -> passes.
# Observational metrics: identical I/O for all inputs -> score 0.
# CAE_down_nf: samples M from full domain {0,1,2,3} -> tests M=1 -> detects.

_EXP5 = dict(
    file=TEST_DIR / "results" / "11_exp5_unreachable_states_results.json",
    n=500, outputs=["Z"],
    conditions=["valid", "wrong_threshold"],
    comparisons=[("valid", "wrong_threshold", "valid -> wrong threshold")],
)


def build_exp5(cond, run_index):
    ctrl = _get_ctrl()
    e = _EXP5

    low_level_model = ctrl.ThresholdLowLevelModel()
    schema = ctrl.build_threshold_schema()
    cg = ctrl.build_threshold_cg(schema)
    vm = ctrl.build_threshold_vm(cg)

    if cond == "valid":
        high_level_model = ctrl.build_valid_threshold_high_level_model()
    else:
        high_level_model = ctrl.build_wrong_threshold_high_level_model()

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    sampler = TopDownSampler(vm)

    return Condition(
        engine=engine,
        tasks=_discrete_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_discrete_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        # Intervene on M only: the violation lives at M=1, which is
        # unreachable from X. Top-down sampling over M's full domain
        # {0,1,2,3} is what exposes it.
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=2,
                         intervention_domain=["M"]),
    )


#  EXPERIMENT 6 - Wrong Causal Direction
#
# Gap tested: Intermediate variable checking (Gap 5).
# low-level model is a chain: x -> y -> z  (z depends on y, not x).
# Wrong high-level model claims a fork: x -> y, x -> z  (ignores y -> z).
# Under observation both produce identical z = 3x + 2.
# Under intervention on y, the fork high-level model still computes z from x,
# but the low-level model computes z from the intervened y.
#
# Expected: observational metrics identical. CAE_down_nf/CAE_up_nf detect mismatch.

_EXP6 = dict(
    file=TEST_DIR / "results" / "11_exp6_wrong_direction_results.json",
    n=200, outputs=["z"],
    conditions=["valid_chain", "wrong_fork"],
    comparisons=[("valid_chain", "wrong_fork", "chain -> fork")],
)


def build_exp6(cond, run_index):
    ctrl = _get_ctrl()
    e = _EXP6

    low_level_model = ctrl.ChainLowLevelModel()
    schema = ctrl.build_fork_schema()
    cg = ctrl.build_fork_cg(schema)
    vm = ctrl.build_fork_vm(cg)

    high_level_model = ctrl.build_chain_high_level_model() if cond == "valid_chain" \
        else ctrl.build_wrong_fork_high_level_model()

    seed = run_index * 2 + e["conditions"].index(cond)
    cfg = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, cg, cfg)
    sampler = TopDownSampler(vm)

    return Condition(
        engine=engine,
        tasks=_continuous_tasks(engine.builder, sampler, e["outputs"]),
        analytical=_continuous_analytical(e["outputs"], e["n"]),
        sampler=sampler,
        task_kwargs=dict(n_samples=e["n"], batch_size=1, max_interventions=2,
                         intervention_domain=["x", "y"]),
    )


#  Experiment registry

EXPERIMENTS = {
    1: dict(title="Exp 1: Hidden Environmental Confounder (PP)",
            results_file=_EXP1["file"], conditions=_EXP1["conditions"],
            comparisons=_EXP1["comparisons"],
            build_condition=build_exp1, print_table_every=5),
    2: dict(title="Exp 2: XOR-Masked Redundant Pathway",
            results_file=_EXP2["file"], conditions=_EXP2["conditions"],
            comparisons=_EXP2["comparisons"],
            build_condition=build_exp2, print_table_every=10),
    3: dict(title="Exp 3: Wrong Intermediate Representation",
            results_file=_EXP3["file"], conditions=_EXP3["conditions"],
            comparisons=_EXP3["comparisons"],
            build_condition=build_exp3, print_table_every=10),
    4: dict(title="Exp 4: Spurious Mediator (fork vs chain)",
            results_file=_EXP4["file"], conditions=_EXP4["conditions"],
            comparisons=_EXP4["comparisons"],
            build_condition=build_exp4, print_table_every=10),
    5: dict(title="Exp 5: Unreachable Intermediate States",
            results_file=_EXP5["file"], conditions=_EXP5["conditions"],
            comparisons=_EXP5["comparisons"],
            build_condition=build_exp5, print_table_every=10),
    6: dict(title="Exp 6: Wrong Causal Direction (chain vs fork)",
            results_file=_EXP6["file"], conditions=_EXP6["conditions"],
            comparisons=_EXP6["comparisons"],
            build_condition=build_exp6, print_table_every=10),
}


#  CLI

def parse_controlled_args(argv=None):
    p = argparse.ArgumentParser(
        description="Controlled experiments for metric failure mode analysis")
    p.add_argument("--experiment", nargs="+", type=int, default=None,
                   help="Experiment numbers to run (1-6). Default: all.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Only (re)compute these metrics.")
    p.add_argument("--runs", type=int, default=None,
                   help="Number of runs per experiment.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_controlled_args()
    exp_ids = args.experiment or sorted(EXPERIMENTS.keys())
    runner_args = SimpleNamespace(metrics=args.metrics, runs=100)

    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"Unknown experiment {eid}. Valid: {sorted(EXPERIMENTS.keys())}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {EXPERIMENTS[eid]['title']}")
        print(f"{'=' * 70}\n")

        run_suite(**EXPERIMENTS[eid], args=runner_args)
"""
Full evaluation suite for the Gas Law system (Lennard-Jones MD vs EOS high-level models).
Five conditions: valid_ideal, valid_vdw, invalid_alpha, invalid_density, invalid_temp.
"""
import os
from pathlib import Path

import numpy as np

from utils import load_system

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(_REPO_ROOT, 'systems'))

_gs = load_system("03_gas_simulation.py", "gas_simulation")

from causal_abstraction import (
    CausalGraph, MicroVariableSchema, CoarseGrainingMap, ContinuousValueMap,
    EvaluationConfig, EvaluationEngine, BottomUpSampler,
    RectSubspace, FullSubspace,
    VarianceDecompositionMetric, CIBLagrangianMetric, IBLagrangianMetric,
    ComplexityShiftMetric, SobolSensitivityMetric, IIAMetric, BCCMetric,
    ProbingMetric, InfidelityMetric, MMDMetric, ConditionalIndependenceMetric,
    RelationalFidelityMetric, MacroscopicInvarianceMetric, RMSEMetric, L2Metric,
)
from causal_abstraction.analytical_metrics import (
    StructuralDeviationMetric, CausalSensitivityIndexMetric, MallowsCpMetric,
)
from causal_abstraction.tasks import StandardTasks
from runner import Condition, run_suite

RESULTS_FILE = Path(__file__).parent / "results" / "03_gas_simulation_results.json"
N_SAMPLES    = 10
CONDITIONS   = ["valid_ideal", "valid_vdw", "invalid_alpha", "invalid_density", "invalid_temp"]
OUTPUTS      = ["pressure"]
COMPARISONS  = [
    ("valid_ideal", "invalid_alpha",   "ideal gas -> wrong alpha"),
    ("valid_ideal", "invalid_density", "ideal gas -> high density"),
    ("valid_vdw",   "invalid_density", "VdW -> high density"),
    ("valid_ideal", "invalid_temp",    "ideal gas -> low temp"),
]

N_PARTICLES  = _gs.N_PARTICLES
T_BOYLE      = _gs.T_BOYLE
LOW_RHO, HIGH_RHO = 0.05, 0.70
MED_TEMP, LOW_TEMP = T_BOYLE, 1.0
ALPHA_VALID, ALPHA_WRONG = 1.0, 0.5
ALL_MICRO    = ["volume", "temperature", "pressure", "positions", "velocities", "box_len"]

COND_SPECS = {
    "valid_ideal":     ((LOW_RHO, MED_TEMP),  ALPHA_VALID, False),
    "valid_vdw":       ((LOW_RHO, MED_TEMP),  None,        True),
    "invalid_alpha":   ((LOW_RHO, MED_TEMP),  ALPHA_WRONG, False),
    "invalid_density": ((HIGH_RHO, MED_TEMP), ALPHA_VALID, False),
    "invalid_temp":    ((LOW_RHO, LOW_TEMP),  ALPHA_VALID, False),
}

_SCHEMA = MicroVariableSchema.from_names(ALL_MICRO)
_CG     = CoarseGrainingMap(_SCHEMA, {k: [k] for k in ALL_MICRO})


class GasBottomUpSampler(BottomUpSampler):
    def __init__(self, value_map, rho_range, t_range, n_particles=N_PARTICLES):
        super().__init__(value_map)
        self.rho_range, self.t_range, self.n_particles = rho_range, t_range, n_particles

    def sample_intervention(self, variables, batch_size=1, max_interventions=None,
                            force_all=False, rng=None):
        rng = self._get_rng(rng)
        pos_l, vel_l, box_l, vol_l, temp_l = [], [], [], [], []
        for _ in range(batch_size):
            rho = float(rng.uniform(*self.rho_range))
            t   = float(rng.uniform(*self.t_range))
            box = float((self.n_particles / rho) ** (1.0 / 3.0))
            k   = int(np.ceil(self.n_particles ** (1.0 / 3.0)))
            s   = box / k
            grd = np.arange(k) * s
            x, y, z = np.meshgrid(grd, grd, grd)
            pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:self.n_particles]
            pos += rng.normal(0.0, s * 0.05, size=pos.shape)
            pos = _gs.minimize_energy(pos, self.n_particles, box) % box
            vel = rng.normal(0.0, np.sqrt(t), size=(self.n_particles, 3))
            vel -= vel.mean(axis=0)
            pos_l.append(pos); vel_l.append(vel); box_l.append(box)
            vol_l.append(box ** 3); temp_l.append(t)
        none_lbls = [None] * batch_size
        return {
            "positions":   {"micro_values": np.stack(pos_l),  "labels": none_lbls},
            "velocities":  {"micro_values": np.stack(vel_l),  "labels": none_lbls},
            "box_len":     {"micro_values": np.array(box_l),  "labels": none_lbls},
            "volume":      {"micro_values": np.array(vol_l),  "labels": vol_l},
            "temperature": {"micro_values": np.array(temp_l), "labels": temp_l},
        }


def _build_domain(rho, t_center, width=0.30):
    v_target = N_PARTICLES / rho
    return (RectSubspace((v_target * (1 - width), v_target * (1 + width))),
            RectSubspace((t_center * (1 - width), t_center * (1 + width))),
            FullSubspace(1))


def _build_vm(_CG, v_sp, t_sp, p_sp):
    return ContinuousValueMap(_CG, {"volume": {0: v_sp}, "temperature": {0: t_sp}, "pressure": {0: p_sp}})


def _make_ideal_high_level_model(alpha, v_sp, t_sp):
    g = CausalGraph()
    g.add_variable("volume", lambda: None, domain=v_sp)
    g.add_variable("temperature", lambda: None, domain=t_sp)
    g.add_variable("pressure",
                   lambda volume, temperature: float(N_PARTICLES * (float(temperature) ** alpha) / float(volume)),
                   parents=["volume", "temperature"], domain=FullSubspace(1))
    return g


def _make_vdw_high_level_model(a, b, v_sp, t_sp):
    def p_fn(volume, temperature):
        rho = N_PARTICLES / float(volume)
        d = 1.0 - b * rho
        return (rho * float(temperature)) / d - a * rho ** 2 if d > 1e-4 else 0.0
    g = CausalGraph()
    g.add_variable("volume", lambda: None, domain=v_sp)
    g.add_variable("temperature", lambda: None, domain=t_sp)
    g.add_variable("pressure", p_fn, parents=["volume", "temperature"], domain=FullSubspace(1))
    return g


def _build_tasks(builder, vm, bu, high_level_model):
    td = _gs.ThermodynamicSampler(vm)
    td_roots = _gs.ThermodynamicSampler(vm)
    td_roots._root_vars_only = [v for n, v in high_level_model.variables.items()
                                 if not high_level_model._nodes.get(n, {}).get("parents")]
    return [
        StandardTasks.score(builder, sampler=td,  name="CAE_down_nf"),
        StandardTasks.score(builder, sampler=bu,  name="CAE_up_nf"),
        StandardTasks.score(builder, sampler=td, name="CAE_down", include_faithfulness=True),
        StandardTasks.score(builder, sampler=bu, name="CAE_up", include_faithfulness=True),
        StandardTasks.observational_r2(  builder, sampler=td_roots, output_vars=OUTPUTS),
        StandardTasks.observational_mse( builder, sampler=td_roots, output_vars=OUTPUTS),
        StandardTasks.observational_nmse(builder, sampler=td_roots, output_vars=OUTPUTS),
        StandardTasks.observational_kl(  builder, sampler=td_roots, output_vars=OUTPUTS),
        StandardTasks.observational_jsd( builder, sampler=td_roots, output_vars=OUTPUTS),
        StandardTasks.observational(builder, L2Metric(),    sampler=td_roots, output_vars=OUTPUTS, name="L2"),
        StandardTasks.observational(builder, RMSEMetric(),  sampler=td_roots, output_vars=OUTPUTS, name="RMSE"),
        StandardTasks.observational(builder, MMDMetric(),   sampler=td_roots, output_vars=OUTPUTS, name="MMD"),
        #StandardTasks.observational(builder, ConditionalIndependenceMetric(), sampler=td_roots, output_vars=OUTPUTS, name="HSIC"),
        StandardTasks.observational(builder, VarianceDecompositionMetric(),   sampler=td_roots, output_vars=OUTPUTS, name="VarDecomp"),
    ]


def _build_analytical(cond, a_vdw, b_vdw, v_sp, t_sp):
    if cond == "valid_vdw":
        pn, nom = ["a", "b"], {"a": a_vdw, "b": b_vdw}
        mk = lambda p: _make_vdw_high_level_model(p["a"], p["b"], v_sp, t_sp)
        n_params = 2
    else:
        alpha_nom = ALPHA_WRONG if cond == "invalid_alpha" else ALPHA_VALID
        pn, nom = ["alpha"], {"alpha": alpha_nom}
        mk = lambda p: _make_ideal_high_level_model(p["alpha"], v_sp, t_sp)
        n_params = 1
    return [
        IIAMetric(inner_metric="mse", output_vars=OUTPUTS, n_pairs=N_SAMPLES),
        BCCMetric(n_pairs=N_SAMPLES, inner_metric="mse"),
        #ProbingMetric(inner_metric="mse"),
        IBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        CIBLagrangianMetric(beta=1.0, n_bins=20, inner_metric="mse"),
        InfidelityMetric(output_vars=OUTPUTS),
        RelationalFidelityMetric(output_vars=OUTPUTS),
        #ComplexityShiftMetric(output_vars=OUTPUTS),
        SobolSensitivityMetric(n_samples=N_SAMPLES, output_vars=OUTPUTS),
        MacroscopicInvarianceMetric(n_pairs=min(N_SAMPLES, 5), inner_metric="mse"),
        #MallowsCpMetric(n_params=n_params, output_vars=OUTPUTS),
        StructuralDeviationMetric(param_names=pn, nominal_params=nom, make_high_level_model=mk, inner_metric="mse"),
        CausalSensitivityIndexMetric(param_names=pn, nominal_params=nom, make_high_level_model=mk, inner_metric="mse"),
    ]


# Calibrate VdW once
print("Calibrating VdW parameters (loads cached result if available)...")
_A_VDW, _B_VDW = _gs.perform_full_calibration(_gs.UniversalGasModel())
print(f"  VdW: a={_A_VDW:.4f}, b={_B_VDW:.4f}\n")


def build_condition(cond: str, run_index: int) -> Condition:
    (rho, t_center), alpha, use_vdw = COND_SPECS[cond]
    v_sp, t_sp, p_sp = _build_domain(rho, t_center)
    vm = _build_vm(_CG, v_sp, t_sp, p_sp)
    high_level_model = (_make_vdw_high_level_model(_A_VDW, _B_VDW, v_sp, t_sp) if use_vdw
            else _make_ideal_high_level_model(alpha, v_sp, t_sp))

    low_level_model = _gs.UniversalGasModel()

    seed = run_index * len(CONDITIONS) + CONDITIONS.index(cond)
    cfg  = EvaluationConfig(metric="mse", seed=seed)
    engine = EvaluationEngine(high_level_model, low_level_model, vm, _CG, cfg)

    bu = GasBottomUpSampler(vm, rho_range=(rho * 0.7, rho * 1.3),
                            t_range=(t_center * 0.7, t_center * 1.3))
    return Condition(
        engine=engine,
        tasks=_build_tasks(engine.builder, vm, bu, high_level_model),
        analytical=_build_analytical(cond, _A_VDW, _B_VDW, v_sp, t_sp),
        sampler=bu,
        task_kwargs=dict(n_samples=N_SAMPLES, batch_size=1, max_interventions=2),
    )


if __name__ == "__main__":
    run_suite(
        title="Gas (LJ fluid vs EOS high-level models)",
        results_file=RESULTS_FILE,
        conditions=CONDITIONS,
        comparisons=COMPARISONS,
        build_condition=build_condition,
        print_table_every=5,
    )
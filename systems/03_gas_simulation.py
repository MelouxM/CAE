"""
Example 3: Gas law abstractions from a realistic Lennard-Jones fluid simulation.

We evaluate the validity of two macroscopic gas laws (ideal gas vs. Van der Waals)
as abstractions of an underlying microscopic Lennard-Jones (LJ) molecular dynamics system.
We work in reduced units (epsilon=1, sigma=1, m=1, kB=1).

Ideal gas law: PV = NkT (PV = NT in reduced units)
Van der Waals law: P = (rho * T) / (1 - b*rho) - a*rho^2  (where rho = N/V)

Low-level model:
- A N-body molecular dynamics simulation using the 12-6 Lennard-Jones potential in a cubic periodic box (N = 128 particles)
  - Integration uses the velocity Verlet algorithm (dt = 0.002)
  - Thermostat: Langevin dynamics provide a stochastic heat bath (canonical ensemble)
  - Barostat: A Berendsen barostat is employed for isobaric (NPT) transitions.
  - Corrections: Long-range tail corrections are applied to the pressure virial to
    account for the potential truncation at r_c=3.0.
- The state variables are the continuous positions and velocities of the N particles.
- Energy minimization is applied before sampling to prevent steric clashes.

High-level models:
- Variables: The three continuous macroscopic thermodynamic quantities (P, V, T).
- Ideal gas law:
    - (V, T) -> P : The forward macroscopic mapping (NVT ensemble).
    - (P, T) -> V : The isobaric response (NPT ensemble).
    - (P, V) -> T : The inverse problem ("PVT"), solved via a proportional-gain
                    temperature controller to find the steady-state T that satisfies the EOS.
- Van der Waals law: The corresponding density-dependent formulations.

Coarse-graining map:
- We map the microscopic configuration to macroscopic observables by averaging
  statistical mechanics quantities (T via kinetic energy, P via the virial theorem, V = Box_length^3).

Value map:
- A continuous identity mapping (P_micro -> P_macro).
- We use RectSubspaces to sample valid intervention ranges for the independent variables.

Additional notes:
- The Van der Waals constants (a, b) are empirically determined via an isothermal sweep at the Boyle temperature
- The system is simulated for over 10,000 steps per sample (minimization + 5,000x equilibration + 5,000x measurement).
- We evaluate the laws in a local region around the starting values of T, V and P.
"""


import os
import pickle
from typing import Optional

import numpy as np
import numba
from scipy.optimize import curve_fit, root_scalar

from causal_abstraction import (
    LowLevelModel, SystemState, TopDownSampler
)

# Anchor the calibration cache to this module's own directory so it always
# resolves to the trusted, repo-local asset (systems/gas_law_calibration.pkl)
# rather than a pickle picked up from an attacker-controlled current working
# directory. pickle.load below executes arbitrary code during deserialization,
# so the source of this file must be controlled.
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gas_law_calibration.pkl")
N_PARTICLES = 128
DT = 0.002
CUTOFF = 3.0
T_BOYLE = 3.417927982  # Boyle temperature for LJ fluid

# Numba physics engine

@numba.njit
def compute_forces(pos, n, box, cutoff_sq):
    """
    Computes Lennard-Jones forces and virial stress.
    Standard LJ potential: 4*eps*((sig/r)^12 - (sig/r)^6)
    Reduced units: eps=1.0, sig=1.0.
    """
    forces = np.zeros_like(pos)
    virial = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            # Periodic boundary conditions (minimum image convention)
            dr = dr - box * np.round(dr / box)
            r2 = np.sum(dr ** 2)

            if r2 < cutoff_sq:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3
                # LJ Force: F = 24 * epsilon * (2*inv_r12 - inv_r6) / r2 * r_vec
                f_scalar = 24.0 * inv_r2 * (2.0 * inv_r6 ** 2 - inv_r6)
                f_vec = f_scalar * dr
                forces[i] += f_vec
                forces[j] -= f_vec

                virial += np.dot(f_vec, dr)  # Virial for pressure calculation: sum(r . f)
    return forces, virial


@numba.njit
def apply_langevin(vel, dt, target_t, gamma=1.0):
    """
    Applies friction and noise to velocities.
    """
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1.0 - c1 ** 2) * np.sqrt(target_t)  # target_t = kB*T/m here

    for i in range(vel.shape[0]):
        for j in range(3):
            vel[i, j] = c1 * vel[i, j] + c2 * np.random.normal(0.0, 1.0)
    return vel


@numba.njit
def get_p_tail_corr(n, vol, cutoff):
    """
    Lennard-Jones tail correction for pressure (analytical integral for LJ 12-6 from r_c to infinity)
    """
    rho = n / vol
    sig, eps = 1.0, 1.0
    term1 = (2.0 / 3.0) * (sig / cutoff) ** 9
    term2 = (sig / cutoff) ** 3
    return (16.0 / 3.0) * np.pi * (rho ** 2) * eps * (sig ** 3) * (term1 - term2)


@numba.njit
def minimize_energy(pos, n, box, max_steps=500, tol=1.0):
    """
    Steepest descent minimization. Removes high-energy overlaps
    after random initialization or volume compression before starting dynamics.
    """
    cutoff_sq = CUTOFF ** 2
    dt_desc = 0.001  # Time step for minimization

    for _ in range(max_steps):
        forces, _ = compute_forces(pos, n, box, cutoff_sq)

        # Cap huge forces to prevent flying off during minimization
        max_f = 0.0
        for i in range(n):
            f_norm = np.sqrt(np.sum(forces[i] ** 2))
            if f_norm > max_f:
                max_f = f_norm

        pos += forces * dt_desc  # Move down gradient
        pos = pos % box  # Wrap

        if max_f < tol:
            break

    return pos

@numba.njit
def run_equilibration(pos, vel, n, box, dt, steps, mode, target_1, target_2):
    """
    Universal molecular dynamics loop.
    Mode 0 (NVT): target_1=T, output=P
    Mode 1 (NPT): target_1=T, target_2=P, output=V
    Mode 2 (PVT): target_1=P, output=T
    """
    cutoff_sq = CUTOFF ** 2
    forces, _ = compute_forces(pos, n, box, cutoff_sq)

    tau_p = 200 * dt
    gamma = 1.0  # Friction coefficient for Langevin
    # Barostat gain: scale_p below DIVIDES the pressure error by this (~x20 coupling).
    # This is a deliberate aggressive-but-clamped gain, not the textbook Berendsen
    # isothermal compressibility kappa_T (which multiplies the pressure error). The
    # hard clamp to [0.995, 1.005] keeps the controller a slow, stable relaxation.
    compressibility = 0.05
    smoothed_p = 0.0

    # In PVT, start with a guess for T and move it to find P
    bath_temp = target_1 if (mode == 0 or mode == 1) else T_BOYLE

    for step in range(steps):
        # Velocity Verlet step 1
        pos += vel * dt + 0.5 * forces * dt ** 2

        if mode == 1 and step % 10 == 0:  # NPT barostat
            curr_v = box ** 3
            curr_rho = n / curr_v

            _, curr_vir = compute_forces(pos, n, box, cutoff_sq)
            curr_k_temp = np.sum(vel ** 2) / (3 * n)  # Kinetic temp; full 3N DOF (no COM removal; ~0.8% bias at N=128)
            curr_p = (curr_rho * curr_k_temp) + (curr_vir / (3.0 * curr_v)) + get_p_tail_corr(n, curr_v, CUTOFF)

            scale_p = (1.0 - (dt / tau_p) * (target_2 - curr_p) / compressibility) ** (1.0 / 3.0)
            scale_p = min(max(scale_p, 0.995), 1.005)
            box *= scale_p
            pos *= scale_p

        pos = pos % box

        # Velocity Verlet step 2
        forces_new, virial = compute_forces(pos, n, box, cutoff_sq)
        vel += 0.5 * (forces + forces_new) * dt
        forces = forces_new

        if mode == 2:  # PVT: Adjust the bath temp to hit a pressure
            curr_v = box ** 3
            curr_rho = n / curr_v
            curr_k_temp = np.sum(vel ** 2) / (3 * n)
            curr_p = (curr_rho * curr_k_temp) + (virial / (3.0 * curr_v)) + get_p_tail_corr(n, curr_v, CUTOFF)

            if step == 0:
                smoothed_p = curr_p
            else:
                smoothed_p = 0.9 * smoothed_p + 0.1 * curr_p

            # Proportional control: If P is too low, raise bath T
            p_err = target_1 - smoothed_p
            bath_temp = bath_temp * (1.0 + 0.005 * p_err)
            bath_temp = max(0.1, min(bath_temp, 10.0))  # Safety clamps

        vel = apply_langevin(vel, dt, bath_temp, gamma)

    return pos, vel, box, bath_temp


def run_measurement(pos, vel, n, box, dt, steps, temperature):
    cutoff_sq = CUTOFF ** 2
    vol = box ** 3
    gamma = 1.0
    p_accum, t_accum = 0.0, 0.0
    forces, _ = compute_forces(pos, n, box, cutoff_sq)

    for step in range(steps):
        pos += vel * dt + 0.5 * forces * dt ** 2  # Verlet step 1
        pos = pos % box

        forces_new, virial_new = compute_forces(pos, n, box, cutoff_sq)
        vel += 0.5 * (forces + forces_new) * dt  # Verlet step 2
        forces = forces_new

        vel = apply_langevin(vel, dt, temperature, gamma)  # Thermostat

        # Measurement
        curr_t = np.sum(vel ** 2) / (3 * n)
        measured_p = (n / vol * curr_t) + (virial_new / (3.0 * vol))
        p_corr = get_p_tail_corr(n, vol, CUTOFF)

        p_accum += measured_p + p_corr
        t_accum += curr_t

    return p_accum / steps, t_accum / steps, box ** 3, pos, vel

# Physics model

class UniversalGasModel(LowLevelModel):
    """Low-level model M: an N-body Lennard-Jones molecular-dynamics simulation."""

    def __init__(self, equil_steps=5000, measure_steps=5000):
        self.N = N_PARTICLES
        self.equil_steps = equil_steps
        self.measure_steps = measure_steps
        # Internal state cache
        self.pos = None
        self.vel = None
        self.box = 10.0

    def _init_lattice(self, box):
        if self.pos is None or self.pos.shape[0] != self.N:  # Reset if dimensions change drastically/uninitialized
            k = int(np.ceil(self.N ** (1 / 3.0)))
            s = box / k
            grid = np.arange(k) * s
            x, y, z = np.meshgrid(grid, grid, grid)
            self.pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:self.N]
            self.pos += (np.random.rand(self.N, 3) - 0.5) * (s * 0.1)
            self.vel = np.random.randn(self.N, 3)
            self.pos = minimize_energy(self.pos, self.N, box)  # Remove overlaps immediately
        return self.pos, self.vel

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        def resolve(key, i):
            # Check interventions
            if key in interventions:
                val = interventions[key]
                if hasattr(val, '__len__') and not isinstance(val, str):
                    if isinstance(val, np.ndarray) and val.ndim == 0:
                        return float(val)
                    return float(val[i]) if i < len(val) else float(val[0])
                return float(val)

            # Check input_state
            val = input_state.get(key)
            if val is not None:
                if hasattr(val, '__len__') and not isinstance(val, str):
                    if isinstance(val, np.ndarray) and val.ndim == 0:
                        return float(val)
                    if isinstance(val, dict) and 'micro_values' in val:
                        return val['micro_values'][i]
                    return float(val[i]) if i < len(val) else float(val[0])
                return float(val)
            return None

        # Determine batch size
        batch_size = 1
        for k, v in interventions.items():
            if hasattr(v, '__len__') and not isinstance(v, str):
                batch_size = len(v)
                break

        # Output arrays
        p_res, t_res, v_res = np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)
        pos_out_list, vel_out_list, box_out_list = [], [], []

        for i in range(batch_size):
            # Resolve targets
            vol_in = resolve("volume", i) or 1000.0
            temp_in = resolve("temperature", i) or 2.0
            press_in = resolve("pressure", i) or 0.1

            # Load microstate if available, else random init
            pos_data = input_state.get("positions")
            vel_data = input_state.get("velocities")
            box_data = input_state.get("box_len")

            # Unwrap dicts if coming from sampler
            if isinstance(pos_data, dict) and 'micro_values' in pos_data: pos_data = pos_data['micro_values']
            if isinstance(vel_data, dict) and 'micro_values' in vel_data: vel_data = vel_data['micro_values']
            if isinstance(box_data, dict) and 'micro_values' in box_data: box_data = box_data['micro_values']

            curr_box = 10.0
            if box_data is not None and len(box_data) > i:
                curr_box = float(box_data[i])

            if pos_data is not None and len(pos_data) > i:
                curr_pos = pos_data[i].copy()
                curr_vel = vel_data[i].copy()
            else:
                curr_pos, curr_vel = self._init_lattice(curr_box)
                curr_pos = minimize_energy(curr_pos, self.N, curr_box)

            # Intervention logic & equilibration
            active_vars = set(interventions.keys())

            mode = 0
            target_1 = temp_in
            target_2 = 0.0

            # Scenario A: DO(Volume), DO(Temperature) -> Standard NVT
            if "volume" in active_vars and "temperature" in active_vars:
                mode = 0 # NVT
                target_box = vol_in ** (1.0 / 3.0)
                if abs(target_box - curr_box) > 1e-4:
                    curr_pos *= target_box / curr_box
                    curr_box = target_box
                target_1 = temp_in

            # Scenario B: DO(Pressure), DO(Temperature) -> NPT equilibration -> NVT measure
            elif "pressure" in active_vars and "temperature" in active_vars:
                mode = 1 # NPT
                target_1 = temp_in
                target_2 = press_in
                if curr_box < 1.0:
                    curr_box = 10.0  # Use current box as starting guess, or default if uninitialized

            # Scenario C: DO(Pressure), DO(Volume) -> Find T (inverse problem)
            elif "pressure" in active_vars and "volume" in active_vars:
                mode = 2 # PVT
                target_box = vol_in ** (1.0 / 3.0)
                if abs(target_box - curr_box) > 1e-4:
                    curr_pos *= target_box / curr_box
                    curr_box = target_box
                target_1 = press_in # Target P

            # Run physics
            new_pos, new_vel, fin_box, found_temp = run_equilibration(
                curr_pos, curr_vel, self.N, curr_box, DT,
                self.equil_steps, mode, target_1, target_2
            )

            p_avg, t_avg, v_avg, fin_pos, fin_vel = run_measurement(
                new_pos, new_vel, self.N, fin_box, DT, self.measure_steps,
                found_temp if mode == 2 else temp_in
            )

            p_res[i] = p_avg
            t_res[i] = t_avg
            v_res[i] = v_avg

            # Store microstates
            pos_out_list.append(fin_pos)
            vel_out_list.append(fin_vel)
            box_out_list.append(fin_box)

        return input_state.merge(SystemState(values={
            "pressure": p_res.reshape(-1, 1),
            "temperature": t_res.reshape(-1, 1),
            "volume": v_res.reshape(-1, 1),
            "positions": np.stack(pos_out_list),
            "velocities": np.stack(vel_out_list),
            "box_len": np.array(box_out_list).reshape(-1, 1),
        }))

# Sampler

class ThermodynamicSampler(TopDownSampler):
    """TopDownSampler (CAE_down) that generates lattice microstates, deferring equilibration to M."""

    def __init__(self, value_map):
        super().__init__(value_map)
        self._root_vars_only  = None

    def sample_intervention(self, variables, batch_size=1, max_interventions=None, force_all=False,
                            rng: Optional[np.random.Generator] = None):
        if hasattr(self, '_root_vars_only') and self._root_vars_only is not None:
            variables = [v for v in variables if v.name in
                         {rv.name for rv in self._root_vars_only}]

        rng = self._get_rng(rng)
        # Get labels (targets)
        base_spec = super().sample_intervention(variables, batch_size, max_interventions, force_all, rng=rng)
        if not base_spec: return {}

        # Generate generic microstates (lattice); leave equilibration to the low-level model
        pos_list, vel_list, box_list = [], [], []

        for i in range(batch_size):
            # Default start
            box = 10.0

            # If volume is a label, use it for the box
            if "volume" in base_spec and base_spec["volume"]["labels"][i] is not None:
                v_target = base_spec["volume"]["labels"][i]
                box = float(v_target) ** (1.0 / 3.0)

            # Lattice init
            k = int(np.ceil(N_PARTICLES ** (1 / 3.0)))
            s = box / k
            grid = np.arange(k) * s
            x, y, z = np.meshgrid(grid, grid, grid)
            pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:N_PARTICLES]
            vel = rng.normal(size=(N_PARTICLES, 3))

            # We do not minimize: the low-level model handles the physics validity.
            # This just provides raw data structures.

            pos_list.append(pos)
            vel_list.append(vel)
            box_list.append(box)

        none_lbls = [None] * batch_size
        result = {
            "positions": {'micro_values': np.stack(pos_list), 'labels': none_lbls},
            "velocities": {'micro_values': np.stack(vel_list), 'labels': none_lbls},
            "box_len": {'micro_values': np.array(box_list), 'labels': none_lbls}
        }
        result.update(base_spec)
        return result

# Calibration and equations

def get_ideal_laws(N):
    return (
        lambda temperature, volume: (N * temperature) / volume,
        lambda pressure, temperature: (N * temperature) / pressure,
        lambda pressure, volume: (pressure * volume) / N
    )


def get_vdw_laws(a, b, N):
    def vdw_P(temperature, volume):
        rho = N / volume
        return (rho * temperature) / (1 - b * rho) - a * rho ** 2

    def vdw_V(pressure, temperature):
        def f(v):
            if v <= b * N + 1.0: return 1e6
            rho = N / v
            return (rho * temperature) / (1 - b * rho) - a * rho ** 2 - pressure

        guess = (N * temperature) / pressure
        try:
            res = root_scalar(f, x0=guess, x1=guess * 1.1)
            return res.root
        except:
            return guess

    def vdw_T(pressure, volume):
        rho = N / volume
        return (pressure + a * rho ** 2) * (1 - b * rho) / rho

    return vdw_P, vdw_V, vdw_T

def perform_full_calibration(low_level_model, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    np.random.seed(rng.integers(2 ** 31))
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "rb") as f: return pickle.load(f)
    print("Calibrating (this may take a minute)...")

    # Run calibration NVT sweeps, sampling different densities at a fixed temp
    rhos = np.linspace(0.05, 0.40, 8)
    data_rho, data_p = [], []

    for r in rhos:
        v = N_PARTICLES / r
        state = SystemState()
        # forward_with_interventions handles the physics loop (minimize -> NVT)
        res = low_level_model.forward_with_interventions(state, {
            "volume": [v], "temperature": [T_BOYLE]
        })
        p_val = res['pressure'][0]
        data_rho.append(r)
        data_p.append(p_val)
        print(f"  Calib: Rho={r:.3f}, P={p_val:.4f}")

    # Fit Van der Waals equation of state
    def vdw_iso(rho, a, b):
        # P = rho*T / (1 - b*rho) - a*rho^2
        denom = 1 - b * rho
        # Numerical stability for fitting only
        denom[denom < 0.01] = 0.01
        return (rho * T_BOYLE) / denom - a * (rho ** 2)

    popt, _ = curve_fit(vdw_iso, np.array(data_rho), np.array(data_p),
                        p0=[4.0, 1.0], bounds=([0, 0.1], [20, 2]))

    print(f"  Result: a={popt[0]:.2f}, b={popt[1]:.2f}")
    with open(CALIB_FILE, "wb") as f:
        pickle.dump(tuple(popt), f)
    return popt
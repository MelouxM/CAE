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
- The Van der Waals constants (a, b) are fitted jointly on NVT sweeps over three supercritical
  isotherms (T = 2, T_Boyle, 5; rho in [0.05, 0.40]). A single isotherm at the Boyle temperature
  cannot identify them, since the second virial coefficient b - a/T vanishes there.
- The system is simulated for over 10,000 steps per sample (minimization + 5,000x equilibration + 5,000x measurement).
- We evaluate the laws in a local region around the starting values of T, V and P.
"""


import json
import os
from typing import Optional

import numpy as np
import numba
from scipy.optimize import curve_fit, root_scalar

from causal_abstraction import (
    LowLevelModel, SystemState, TopDownSampler
)

N_PARTICLES = 128
DT = 0.002
CUTOFF = 3.0
T_BOYLE = 3.417927982  # Boyle temperature for LJ fluid

# Numba physics engine

@numba.njit
def seed_numba(seed):
    """Seed the generator used inside Numba-compiled code (separate from NumPy's)."""
    np.random.seed(seed)


@numba.njit
def _lj_pair(dr, r2):
    """LJ force vector on i from j for separation dr = r_i - r_j (reduced units)."""
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    # F = 24 * epsilon * (2*inv_r12 - inv_r6) / r2 * r_vec
    return 24.0 * inv_r2 * (2.0 * inv_r6 ** 2 - inv_r6) * dr


@numba.njit
def _compute_forces_all_pairs(pos, n, box, cutoff_sq):
    forces = np.zeros_like(pos)
    virial = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            # Periodic boundary conditions (minimum image convention)
            dr = dr - box * np.round(dr / box)
            r2 = np.sum(dr ** 2)
            if r2 < cutoff_sq:
                f_vec = _lj_pair(dr, r2)
                forces[i] += f_vec
                forces[j] -= f_vec
                virial += np.dot(f_vec, dr)  # Virial for pressure calculation: sum(r . f)
    return forces, virial


@numba.njit
def _compute_forces_cells(pos, n, box, cutoff_sq, nc):
    """Same as the all-pairs loop, restricted to the 27 neighboring cells (nc >= 3 per side)."""
    forces = np.zeros_like(pos)
    virial = 0.0
    cs = box / nc
    head = -np.ones(nc * nc * nc, dtype=np.int64)
    nxt = -np.ones(n, dtype=np.int64)
    cell_of = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        for d in range(3):
            cell_of[i, d] = int(np.floor((pos[i, d] % box) / cs)) % nc
        c = (cell_of[i, 0] * nc + cell_of[i, 1]) * nc + cell_of[i, 2]
        nxt[i] = head[c]
        head[c] = i
    for i in range(n):
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    cx = (cell_of[i, 0] + ox) % nc
                    cy = (cell_of[i, 1] + oy) % nc
                    cz = (cell_of[i, 2] + oz) % nc
                    j = head[(cx * nc + cy) * nc + cz]
                    while j >= 0:
                        if j > i:
                            dr = pos[i] - pos[j]
                            dr = dr - box * np.round(dr / box)
                            r2 = np.sum(dr ** 2)
                            if r2 < cutoff_sq:
                                f_vec = _lj_pair(dr, r2)
                                forces[i] += f_vec
                                forces[j] -= f_vec
                                virial += np.dot(f_vec, dr)
                        j = nxt[j]
    return forces, virial


@numba.njit
def compute_forces(pos, n, box, cutoff_sq):
    """
    Computes Lennard-Jones forces and virial stress.
    Standard LJ potential: 4*eps*((sig/r)^12 - (sig/r)^6)
    Reduced units: eps=1.0, sig=1.0.

    Uses cell lists (O(N)) when the box holds at least 3 cells of the cutoff size per side,
    and the all-pairs loop otherwise; both visit exactly the pairs within the cutoff.
    """
    nc = int(box / np.sqrt(cutoff_sq))
    if nc >= 3:
        return _compute_forces_cells(pos, n, box, cutoff_sq, nc)
    return _compute_forces_all_pairs(pos, n, box, cutoff_sq)


VERLET_SKIN = 0.4


@numba.njit
def _build_pairs(pos, n, box, rlist_sq):
    """All pairs closer than the list radius (minimum image), as two index arrays."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dr = dr - box * np.round(dr / box)
            if np.sum(dr ** 2) < rlist_sq:
                count += 1
    pi = np.empty(count, dtype=np.int64)
    pj = np.empty(count, dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dr = dr - box * np.round(dr / box)
            if np.sum(dr ** 2) < rlist_sq:
                pi[k] = i
                pj[k] = j
                k += 1
    return pi, pj


@numba.njit
def forces_with_list(pos, n, box, cutoff_sq, pi, pj, pos_ref, box_ref):
    """LJ forces and virial, using a Verlet neighbor list when cell lists cannot help.

    With at least 4 cutoff cells per side this just calls `compute_forces`. Otherwise
    the pairs within cutoff + skin are listed once and reused until a particle has
    moved more than half the skin or the box has changed.

    Returns:
        (forces, virial, pi, pj, pos_ref, box_ref): the list state to pass back in.
    """
    if int(box / np.sqrt(cutoff_sq)) >= 4:
        f, v = compute_forces(pos, n, box, cutoff_sq)
        return f, v, pi, pj, pos_ref, box_ref
    rebuild = box != box_ref
    if not rebuild:
        half_skin_sq = (0.5 * VERLET_SKIN) ** 2
        for i in range(n):
            dr = pos[i] - pos_ref[i]
            dr = dr - box * np.round(dr / box)
            if np.sum(dr ** 2) > half_skin_sq:
                rebuild = True
                break
    if rebuild:
        rlist = np.sqrt(cutoff_sq) + VERLET_SKIN
        pi, pj = _build_pairs(pos, n, box, rlist * rlist)
        pos_ref = pos.copy()
        box_ref = box
    forces = np.zeros_like(pos)
    virial = 0.0
    for k in range(pi.shape[0]):
        i = pi[k]
        j = pj[k]
        dr = pos[i] - pos[j]
        dr = dr - box * np.round(dr / box)
        r2 = np.sum(dr ** 2)
        if r2 < cutoff_sq:
            f_vec = _lj_pair(dr, r2)
            forces[i] += f_vec
            forces[j] -= f_vec
            virial += np.dot(f_vec, dr)
    return forces, virial, pi, pj, pos_ref, box_ref


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

        # Cap the displacement at 0.05 sigma per iteration so that the huge forces of
        # overlapping pairs cannot fling particles across the box
        step_scale = dt_desc
        if max_f * dt_desc > 0.05:
            step_scale = 0.05 / max_f
        pos += forces * step_scale  # Move down gradient
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
    pi = np.empty(0, dtype=np.int64)
    pj = np.empty(0, dtype=np.int64)
    pos_ref = pos.copy()
    box_ref = -1.0
    forces, _, pi, pj, pos_ref, box_ref = forces_with_list(pos, n, box, cutoff_sq, pi, pj, pos_ref, box_ref)

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

            _, curr_vir, pi, pj, pos_ref, box_ref = forces_with_list(
                pos, n, box, cutoff_sq, pi, pj, pos_ref, box_ref)
            curr_k_temp = np.sum(vel ** 2) / (3 * n)  # Kinetic temp; full 3N DOF (no COM removal; ~0.8% bias at N=128)
            curr_p = (curr_rho * curr_k_temp) + (curr_vir / (3.0 * curr_v)) + get_p_tail_corr(n, curr_v, CUTOFF)

            scale_p = (1.0 - (dt / tau_p) * (target_2 - curr_p) / compressibility) ** (1.0 / 3.0)
            scale_p = min(max(scale_p, 0.995), 1.005)
            box *= scale_p
            pos *= scale_p

        pos = pos % box

        # Velocity Verlet step 2
        forces_new, virial, pi, pj, pos_ref, box_ref = forces_with_list(
            pos, n, box, cutoff_sq, pi, pj, pos_ref, box_ref)
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
    nl = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), pos.copy(), -1.0)
    forces, _, *nl = forces_with_list(pos, n, box, cutoff_sq, *nl)

    for step in range(steps):
        pos += vel * dt + 0.5 * forces * dt ** 2  # Verlet step 1
        pos = pos % box

        forces_new, virial_new, *nl = forces_with_list(pos, n, box, cutoff_sq, *nl)
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

def make_initial_positions(n, box, rng):
    """Random subset of a jittered cubic lattice with k^3 >= n sites, filling the whole box."""
    k = int(np.ceil(n ** (1 / 3.0)))
    s = box / k
    grid = np.arange(k) * s
    x, y, z = np.meshgrid(grid, grid, grid)
    sites = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    pos = sites[rng.choice(len(sites), size=n, replace=False)].copy()
    pos += (rng.random((n, 3)) - 0.5) * (s * 0.1)
    return minimize_energy(pos % box, n, box)


# Physics model

class UniversalGasModel(LowLevelModel):
    """Low-level model M: an N-body Lennard-Jones molecular-dynamics simulation."""

    def __init__(self, equil_steps=5000, measure_steps=5000, n_particles=N_PARTICLES):
        self.N = n_particles
        self.equil_steps = equil_steps
        self.measure_steps = measure_steps

    def forward_with_interventions(self, input_state: SystemState, interventions: dict,
                                   rng: Optional[np.random.Generator] = None) -> SystemState:
        gen = rng if rng is not None else np.random.default_rng()
        seed_numba(int(gen.integers(2 ** 31 - 1)))

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
                curr_pos = np.array(pos_data[i], dtype=float)  # copy: never alias the caller's state
                curr_vel = np.array(vel_data[i], dtype=float)
            else:
                curr_pos = make_initial_positions(self.N, curr_box, gen)
                curr_vel = gen.normal(0.0, np.sqrt(temp_in), size=(self.N, 3))

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

            # Remove overlaps left by the starting configuration or by compressing the box
            # to the target volume (no-op when the forces are already small)
            curr_pos = minimize_energy(curr_pos % curr_box, self.N, curr_box)

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

    def __init__(self, value_map, n_particles=N_PARTICLES):
        super().__init__(value_map)
        self._root_vars_only  = None
        self.n_particles = n_particles

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

            pos = make_initial_positions(self.n_particles, box, rng)
            vel = rng.normal(size=(self.n_particles, 3))

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

CALIB_TEMPS = (2.0, T_BOYLE, 5.0)
CALIB_RHOS = tuple(np.linspace(0.05, 0.40, 8))


def calibration_file(n_particles):
    """Repo-local JSON cache of the fitted (a, b) for a given particle number."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"gas_vdw_calibration_N{n_particles}.json")


def perform_full_calibration(low_level_model, rng: Optional[np.random.Generator] = None):
    """Fit the van der Waals constants (a, b) to the simulated fluid.

    NVT runs on a grid of supercritical isotherms (T = 2, T_Boyle, 5) and densities
    rho in [0.05, 0.40], then a joint least-squares fit of P = rho T / (1 - b rho) - a rho^2.
    Several isotherms are needed: on the Boyle isotherm alone the second virial
    coefficient b - a/T vanishes, so a and b are not separately identifiable.
    The result is cached per particle number.
    """
    path = calibration_file(low_level_model.N)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["a"], d["b"]
    rng = rng if rng is not None else np.random.default_rng(0)
    print("Calibrating van der Waals constants (NVT sweep over 3 isotherms)...")

    data_t, data_rho, data_p = [], [], []
    for t in CALIB_TEMPS:
        for r in CALIB_RHOS:
            res = low_level_model.forward_with_interventions(SystemState(), {
                "volume": [low_level_model.N / r], "temperature": [t]
            }, rng=rng)
            p_val = float(res['pressure'][0, 0])
            data_t.append(t); data_rho.append(r); data_p.append(p_val)
            print(f"  Calib: T={t:.3f} rho={r:.3f} P={p_val:.4f}")

    def vdw_surface(x, a, b):
        t, rho = x
        denom = np.maximum(1 - b * rho, 0.01)  # numerical stability for fitting only
        return (rho * t) / denom - a * (rho ** 2)

    popt, _ = curve_fit(vdw_surface, (np.array(data_t), np.array(data_rho)), np.array(data_p),
                        p0=[4.0, 1.0], bounds=([0, 0.1], [20, 2]))
    a, b = float(popt[0]), float(popt[1])
    print(f"  Result: a={a:.3f}, b={b:.3f}")
    with open(path, "w") as f:
        json.dump({"a": a, "b": b, "n_particles": low_level_model.N,
                   "temps": list(CALIB_TEMPS), "rhos": list(CALIB_RHOS),
                   "points": [list(z) for z in zip(data_t, data_rho, data_p)]}, f, indent=1)
    return a, b

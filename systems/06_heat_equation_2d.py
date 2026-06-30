"""
Example 6: From phonons to the heat equation (spatial abstraction).
We demonstrate that the continuous heat equation is a valid causal abstraction of
discrete phonon scattering in a crystal lattice, but only under specific physical regimes.

Low-level model:
- A 2D mass-spring lattice representing atoms in a crystal.
- Physics: Newton's laws + anharmonic scattering (random velocity rotation).
- State: Positions and velocities of N*N atoms.
- Observable: Kinetic energy density (Temperature map).

High-level model:
- The 2D heat equation with Neumann boundary conditions.
- Physics: Continuum diffusion (dT/dt = alpha * Laplacian(T)).
- State: Continuous 2D temperature field.

Coarse-graining:
- Micro -> Macro: Spatial smoothing (Gaussian filter) of atomic kinetic energies.
- This creates a continuous field from discrete point masses.

Value map:
- Identity mapping for control parameters (Source X, Y, Energy).
- Continuous field mapping for the output temperature map.

Conditions evaluated (see test/06_heat_equation_2d.py):
- Valid (high scattering): Valid abstraction. Atoms thermalize locally.
- Fail (E uses 10x the diffusion coefficient): Invalid abstraction. E over-diffuses
  relative to the abstracted lattice output.
- Noise (Gaussian output noise on the kinetic-energy map): Noisy abstraction.
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from causal_abstraction import (
    SystemState, LowLevelModel, ContinuousValueMap,
)

try:
    from numba import njit
except ImportError:
    def njit(func): return func

logger = logging.getLogger(__name__)


#Physics engine (Numba)
@njit
def compute_forces_lattice(pos, shape, k_spring):
    H, W = shape
    N = H * W
    pos_grid = pos.reshape((H, W, 2))
    force_grid = np.zeros((H, W, 2))

    for r in range(H):
        for c in range(W):
            p0 = pos_grid[r, c]
            neighbors = []
            if r > 0:
                neighbors.append((pos_grid[r - 1, c], np.array([c, r - 1])))
            if r < H - 1:
                neighbors.append((pos_grid[r + 1, c], np.array([c, r + 1])))
            if c > 0:
                neighbors.append((pos_grid[r, c - 1], np.array([c - 1, r])))
            if c < W - 1:
                neighbors.append((pos_grid[r, c + 1], np.array([c + 1, r])))

            f_acc = np.zeros(2)
            for p_neigh, rest_neigh in neighbors:
                curr_diff = p0 - p_neigh
                # rest_neigh is float64 to match p_neigh
                rest_diff = np.array([float(c), float(r)]) - rest_neigh
                displacement = curr_diff - rest_diff
                f_acc -= k_spring * displacement
            force_grid[r, c] = f_acc

    return force_grid.reshape((N, 2))

@njit
def run_scattering_steps(pos, vel, n_steps, dt, k_spring, p_scatter, mass, shape):
    for _ in range(n_steps):
        forces = compute_forces_lattice(pos, shape, k_spring)
        vel += 0.5 * (forces / mass) * dt
        pos += vel * dt

        forces = compute_forces_lattice(pos, shape, k_spring)
        vel += 0.5 * (forces / mass) * dt

        # Phonon scattering (random velocity rotation)
        if p_scatter > 0:
            n_particles = len(vel)
            for i in range(n_particles):
                if np.random.rand() < p_scatter:
                    vx, vy = vel[i]
                    speed = np.sqrt(vx*vx + vy*vy)
                    if speed > 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        vel[i, 0] = speed * np.cos(angle)
                        vel[i, 1] = speed * np.sin(angle)
    return pos, vel

@njit
def solve_heat_pde_neumann(grid, alpha, dt, steps, dx=1.0):
    curr = grid.copy()
    # Stability check for explicit Euler
    stability_criterion = alpha * dt / (dx ** 2)

    if stability_criterion > 0.2:
        sub_steps = int(np.ceil(stability_criterion / 0.2))
        dt_sub = dt / sub_steps
    else:
        sub_steps = 1
        dt_sub = dt

    for _ in range(steps * sub_steps):
        # Neumann boundaries (derivative = 0)
        curr[0, :] = curr[1, :]
        curr[-1, :] = curr[-2, :]
        curr[:, 0] = curr[:, 1]
        curr[:, -1] = curr[:, -2]

        laplacian = np.zeros_like(curr)
        laplacian[1:-1, 1:-1] = (
            curr[2:, 1:-1] + curr[:-2, 1:-1] +
            curr[1:-1, 2:] + curr[1:-1, :-2] -
            4 * curr[1:-1, 1:-1]
        )
        curr += (alpha * dt_sub / (dx ** 2)) * laplacian

    return curr


# Low-level model
class PhononLatticeModel(LowLevelModel):
    """Low-level model M: a 2D lattice of coupled oscillators (phonons) with scattering."""

    def __init__(self, grid_h=32, grid_w=32, k_spring=100.0, p_scatter=0.1, dt=0.02, sim_time=20.0, n_averages=50):
        self.H = grid_h
        self.W = grid_w
        self.k = k_spring
        self.p_scatter = p_scatter
        self.dt = dt
        self.sim_time = sim_time
        self.mass = 1.0
        self.n_averages = n_averages

        self.pos0 = np.zeros((self.H * self.W, 2))
        for r in range(self.H):
            for c in range(self.W):
                self.pos0[r * self.W + c] = [c, r]

    def measure_temperature_map(self, vel):
        # T ~ kinetic energy
        ke = 0.5 * self.mass * np.sum(vel ** 2, axis=1)
        return ke.reshape((self.H, self.W))

    def _run_single_instance(self, source_x, source_y, source_E):
        pos = self.pos0.copy()
        vel = np.zeros_like(pos)

        # Inject energy ("kick" atoms at source)
        if source_x >= 0:
            cx = int(source_x * self.W)
            cy = int(source_y * self.H)
            cx = np.clip(cx, 3, self.W - 4)
            cy = np.clip(cy, 3, self.H - 4)
            idx = cy * self.W + cx

            angle = np.random.uniform(0, 2 * np.pi)
            v_mag = np.sqrt(2 * source_E / self.mass)
            vel[idx] += np.array([np.cos(angle), np.sin(angle)]) * v_mag

        # Run dynamics
        n_steps = int(self.sim_time / self.dt)
        pos, vel = run_scattering_steps(
            pos, vel, n_steps, self.dt,
            self.k, self.p_scatter, self.mass, (self.H, self.W)
        )
        return self.measure_temperature_map(vel)

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        # Standardize inputs to batch arrays
        def get_batch(name, default):
            val = interventions.get(name, input_state.get(name, default))
            return np.atleast_1d(val)

        sx_in = get_batch('source_x', -1.0)
        sy_in = get_batch('source_y', -1.0)
        se_in = get_batch('source_E', 0.0)

        batch_size = len(sx_in)
        results = []

        for i in range(batch_size):
            # Avoid numpy deprecation warnings
            sx = sx_in[i].item() if hasattr(sx_in[i], 'item') else sx_in[i]
            sy = sy_in[i].item() if hasattr(sy_in[i], 'item') else sy_in[i]
            se = se_in[i].item() if hasattr(se_in[i], 'item') else se_in[i]

            # Run N averages to get a statistically stable heatmap
            accum_map = np.zeros((self.H, self.W))
            for _ in range(self.n_averages):
                accum_map += self._run_single_instance(sx, sy, se)

            avg_map = accum_map / self.n_averages
            results.append(avg_map)

        # Stack to (batch, H, W)
        final_tensor = np.stack(results, axis=0)

        return input_state.merge(SystemState(values={
            'source_x': sx_in,
            'source_y': sy_in,
            'source_E': se_in,
            'final_temp_map': final_tensor
        }))


# Maps and calibration
class SpatialValueMap(ContinuousValueMap):
    """Value map tau: Gaussian-smooths the micro temperature field into the coarse macro field."""

    def __init__(self, cg, specs, grid_shape):
        super().__init__(cg, specs)
        self.H, self.W = grid_shape
        # Dynamic smoothing sigma: 5% of grid size (acts as the coarse-graining filter)
        self.sigma = max(0.1, self.H * 0.05)

    def ground(self, abstract_var_name: str, label: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        # Grounding scalar parameters (Source X, Y, E)
        if isinstance(label, np.ndarray):
            return label
        if isinstance(label, (float, int, np.number)):
            return np.array([label], dtype=float)
        return super().ground(abstract_var_name, label, rng=rng)

    def abstract(self, abstract_var_name: str, micro_values: np.ndarray) -> np.ndarray:
        # Abstracting the heatmap
        if abstract_var_name == "final_temp_map":
            # micro_values is (batch, H, W) or (H, W)
            # Gaussian filter handles batch dimension automatically if axis is specified,
            # but simpler to loop for explicit normalization.

            if micro_values.ndim == 3:
                res = []
                for i in range(micro_values.shape[0]):
                    smoothed = gaussian_filter(micro_values[i], sigma=self.sigma)
                    total = np.sum(smoothed)
                    if total > 1e-9: smoothed /= total
                    res.append(smoothed)
                return np.stack(res)
            else:
                smoothed = gaussian_filter(micro_values, sigma=self.sigma)
                total = np.sum(smoothed)
                if total > 1e-9: smoothed /= total
                return smoothed

        return micro_values

def calibrate_alpha_opt(grid_size, p_scatter, sim_time, sigma_smooth, n_calib_averages=20):
    """Fits the diffusion coefficient (alpha) of the heat equation to the simulation."""
    print(f"Calibrating (Size={grid_size}, P_scat={p_scatter})...")
    low_level_model = PhononLatticeModel(grid_size, grid_size, p_scatter=p_scatter, sim_time=sim_time, n_averages=n_calib_averages)

    state = SystemState()
    # Run one intervention at center to get diffusion profile
    res = low_level_model.forward_with_interventions(state, {
        "source_x": np.array([0.5]), "source_y": np.array([0.5]), "source_E": np.array([1000.0])
    })
    target_map = gaussian_filter(res['final_temp_map'][0], sigma=sigma_smooth)
    total = np.sum(target_map)
    if total > 1e-9: target_map /= total

    def objective(alpha):
        grid = np.zeros((grid_size, grid_size))
        cx, cy = int(0.5 * grid_size), int(0.5 * grid_size)
        grid[cy, cx] = 1.0
        # Smooth initial condition to match the abstraction map's view
        grid = gaussian_filter(grid, sigma=sigma_smooth)

        pred_map = solve_heat_pde_neumann(grid, alpha, dt=0.2, steps=int(sim_time/0.2))

        ptotal = np.sum(pred_map)
        if ptotal > 1e-9: pred_map /= ptotal
        return np.linalg.norm(pred_map - target_map)

    res = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
    print(f"  -> optimized alpha: {res.x:.4f} (Error: {res.fun:.4f})")
    return res.x


#Main
def run_full_suite(*args, **kwargs):
    """Deprecated shim. Run the evaluation suite from the command line instead:
    ``python test/06_heat_equation_2d.py`` (built on ``test/runner.run_suite``)."""
    raise RuntimeError(
        "run_full_suite is not available here; run the 2D heat-equation evaluation "
        "suite directly with `python test/06_heat_equation_2d.py`."
    )


def main():
    """Simulate phonon diffusion for a small grid and display the final temperature map."""
    logging.basicConfig(level=logging.INFO)
    grid_size = 16
    p_scatter = 0.2
    sim_time = 5.0

    print(f"Phonon lattice simulation (grid={grid_size}x{grid_size}, p_scatter={p_scatter}, sim_time={sim_time})")

    low_level_model = PhononLatticeModel(
        grid_h=grid_size, grid_w=grid_size,
        p_scatter=p_scatter, sim_time=sim_time,
        n_averages=5,
    )
    # Use a fixed source in the center
    state = low_level_model.forward_with_interventions(
        SystemState(values={}),
        {"source_x": np.array([[0.5]]), "source_y": np.array([[0.5]]), "source_E": np.array([[750.0]])},
    )
    temp_map = np.asarray(state.values.get("final_temp_map", [[0]])).ravel()
    total_energy = temp_map.sum()
    peak = temp_map.max()
    print(f"  Total energy: {total_energy:.4f}  Peak temperature: {peak:.4f}")

if __name__ == "__main__":
    main()
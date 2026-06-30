"""
Example 5: From brownian motion to the heat equation.
We demonstrate that the heat equation (a continuous partial differential equation)
is the emergent limit of a system of discrete Brownian particles as N -> infinity.

Low-level model:
- 1D Brownian motion of N particles in a box [0, L].
- Particles evolve stochastically: x(t+dt) = x(t) + N(0, 2D*dt).
- Reflective boundary conditions.
- State: Position vector of N particles.

High-level model:
- The heat equation: dT/dt = D * d^2T/dx^2.
- Solved numerically using finite differences on a grid.
- State: Temperature profile T(x) discretized into bins.

Coarse-graining:
- Micro-state (Positions) -> Macro-state (Histogram).
- The continuous positions of N particles are mapped into a spatial histogram (concentration profile).

Value map:
- Abstraction: Binning particle positions.
- Grounding: Sampling particle positions from a probability density function (PDF).
"""
import logging
from typing import Optional

import numpy as np
import torch

from causal_abstraction import (
    LowLevelModel, SystemState, InterventionSampler, ContinuousValueMap,
)

try:
    from numba import njit
except ImportError:
    def njit(func): return func


@njit
def run_brownian_steps(positions, n_steps, step_size, box_length):
    """Evolves particle positions with reflective boundaries."""
    n_particles = positions.shape[0]
    for _ in range(n_steps):
        steps = np.random.normal(0, step_size, size=n_particles)
        positions += steps
        # Reflective boundary conditions
        for i in range(n_particles):
            while positions[i] < 0 or positions[i] > box_length:
                if positions[i] < 0:
                    positions[i] = -positions[i]
                elif positions[i] > box_length:
                    positions[i] = 2 * box_length - positions[i]
    return positions


# Low-level model

class BrownianParticleSystem(LowLevelModel):
    """Low-level model M: 1D Brownian motion of N particles in a box with reflective walls."""

    def __init__(self, n_particles=1000, n_steps=100, diff_coeff=0.1, box_len=1.0, dt=0.001):
        self.N = n_particles
        self.L = box_len
        self.alpha = diff_coeff
        self.dt = dt
        self.step_size = np.sqrt(2 * self.alpha * self.dt)
        self.n_steps = n_steps

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        # Resolve input (initial positions): check interventions, then state, then random initialization
        if 'T_initial' in interventions:
            positions = interventions['T_initial']
        elif 'T_initial' in input_state.values:
            positions = input_state['T_initial']
        else:
            positions = np.random.uniform(0, self.L, (1, self.N))

        # Ensure batch dimension (batch, N)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]

        batch_size, current_n = positions.shape

        # Resample if particle count doesn't match (e.g. if grounding used a different N)
        if current_n != self.N:
            new_pos = np.zeros((batch_size, self.N))
            for i in range(batch_size):
                indices = np.random.choice(current_n, self.N, replace=True)
                new_pos[i] = positions[i, indices]
            positions = new_pos

        # Run simulation
        flat_pos = positions.flatten()  # For Numba
        final_flat = run_brownian_steps(flat_pos.copy(), self.n_steps, self.step_size, self.L)

        final_pos = final_flat.reshape(batch_size, self.N)  # Reshape from Numba

        return input_state.merge(SystemState(values={
            'T_initial': positions,
            'T_final': final_pos
        }))


# High-level model

class HeatEquationSolver:
    """High-level model E: a finite-difference solver for the 1D heat equation."""

    def __init__(self, n_bins=50, diff_coeff=0.1, box_len=1.0, dt=0.001, n_steps=100):
        self.dx = box_len / n_bins
        self.dt = dt
        self.alpha = (diff_coeff * dt) / (self.dx ** 2)
        self.n_steps = n_steps

    def solve(self, initial_profile):
        """
        Solve the 1D heat equation using finite differences.

        Args:
            initial_profile: Initial heat profile, shape (batch, bins) or (bins,).

        Returns:
            The evolved profile with the same shape as ``initial_profile``.
        """
        u = np.array(initial_profile, dtype=float)
        was_1d = (u.ndim == 1)
        if was_1d: u = u[np.newaxis, :]

        # Time-stepping
        for _ in range(self.n_steps):
            # Vectorized finite difference
            u_left = np.roll(u, 1, axis=1)
            u_right = np.roll(u, -1, axis=1)

            # Neumann boundaries (derivative = 0) -> Neighbors match edge
            u_left[:, 0] = u[:, 0]
            u_right[:, -1] = u[:, -1]

            u = u + self.alpha * (u_left + u_right - 2*u)

        return u[0] if was_1d else u


# Maps and sampler

class BinningValueMap(ContinuousValueMap):
    """Value map tau: abstract() bins particle positions into a histogram; ground() resamples positions."""

    def __init__(self, cg_map, specs, n_bins, box_len, n_particles):
        super().__init__(cg_map, specs)
        self.n_bins = n_bins
        self.L = box_len
        self.N = n_particles
        self.edges = np.linspace(0, self.L, self.n_bins + 1)

    def abstract(self, var_name, micro_val):
        """Maps particle positions (N,) to histogram (bins,)."""
        if micro_val.ndim == 2:
            batch_size = micro_val.shape[0]
            res = []
            for i in range(batch_size):
                counts, _ = np.histogram(micro_val[i], bins=self.edges)
                res.append(counts.astype(float) / len(micro_val[i]))
            return np.stack(res)
        else:
            counts, _ = np.histogram(micro_val, bins=self.edges)
            return counts.astype(float) / len(micro_val)

    def ground(self, var_name, macro_val, rng: Optional[np.random.Generator] = None):
        """Map a histogram (bins,) back to particle positions (N,) by resampling the PDF."""
        if rng is None:
            rng = np.random.default_rng()
        pdf = np.maximum(macro_val, 0)
        total = np.sum(pdf)  # Normalize to get PDF
        if total == 0:
            pdf = np.ones_like(pdf)
            total = np.sum(pdf)
        pdf = pdf / total

        bin_indices = rng.choice(self.n_bins, size=self.N, p=pdf)  # Sample bin indices based on PDF

        # Uniformly distribute within bins (use the passed seeded rng for
        # cross-process reproducibility, not the global NumPy RNG).
        bin_width = self.L / self.n_bins
        offsets = rng.random(self.N) * bin_width
        positions = self.edges[bin_indices] + offsets

        return positions

class SmoothProfileSampler(InterventionSampler):
    """
    Generates smooth Gaussian temperature profiles for the heat equation.
    """
    def __init__(self, value_map, n_bins):
        super().__init__(value_map)
        self.n_bins = n_bins
        self.x_grid = np.linspace(0, 1, n_bins)

    def sample_intervention(self, variables, batch_size=1, max_interventions=None, force_all=False, rng: Optional[np.random.Generator] = None):
        rng = self._get_rng(rng)
        spec = {}
        target_vars = [v.name for v in variables]

        # Only intervene on initial temperature
        if 'T_initial' in target_vars:
            batch_labels = []
            batch_micro = []

            for _ in range(batch_size):
                # Generate random Gaussian bump
                mu = rng.uniform(0.3, 0.7)
                sigma = rng.uniform(0.05, 0.15)
                amp = 100.0  # Arbitrary magnitude

                profile_pdf = amp * np.exp(-0.5 * ((self.x_grid - mu) / sigma)**2)

                # Ground to micro-state, then abstract back to get the "canonical" macro-state for the high-level model
                micro = self.value_map.ground('T_initial', profile_pdf)
                high_level_model_input = self.value_map.abstract('T_initial', micro)

                batch_labels.append(high_level_model_input)
                batch_micro.append(micro)

            spec['T_initial'] = {
                'labels': batch_labels,
                'micro_values': np.stack(batch_micro)
            }

        return spec


def main():
    """Simulate Brownian particle diffusion for various N and show the final spread."""
    L, alpha = 1.0, 0.1
    T_max, STEPS = 0.2, 200
    dt = T_max / STEPS
    rng = np.random.default_rng(0)

    particle_counts = [100, 1000, 10000]
    print(f"Brownian diffusion demo (alpha={alpha}, L={L}, steps={STEPS})")
    print(f"{'N':<10} | {'mean pos':<12} | {'std pos':<12}")
    print("-" * 38)

    for n in particle_counts:
        low_level_model = BrownianParticleSystem(
            n_particles=n, n_steps=STEPS, diff_coeff=alpha, box_len=L, dt=dt
        )
        initial_pos = rng.uniform(0.4, 0.6, (1, n))
        state = low_level_model.forward_with_interventions(
            SystemState(values={}),
            {"T_initial": initial_pos},
        )
        final_pos = np.asarray(state.values["T_final"]).ravel()
        print(f"{n:<10} | {final_pos.mean():<12.4f} | {final_pos.std():<12.4f}")


if __name__ == '__main__':
    main()
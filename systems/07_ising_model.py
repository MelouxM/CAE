"""
Example 7: Evaluating the Ising model as an Abstraction.
We test whether the classic rigid-lattice Ising model is a valid abstraction of a
more realistic physical system where atoms vibrate and interactions depend on distance.

Low-level model:
- A hybrid molecular dynamics (MD) + Monte Carlo (MC) simulation.
- Atoms exist in continuous 2D space and are given an initial thermal velocity from the
  vibrational temperature (microcanonical / NVE: a one-time Maxwell-Boltzmann seed, no thermostat).
- Spin interactions decay with distance: J_eff = J0 * exp(-decay * |r - r0|)
  (symmetric in |r - r0|; a phenomenological simplification, since real exchange is monotone in r).
- State: Continuous positions and discrete spins.

High-level model:
- The textbook 2D Ising model.
- Atoms are frozen on a rigid grid.
- Constant neighbor interaction J.
- State: Discrete spins only.

Coarse-graining:
- Maps the complex, vibrating micro-state to the idealized thermodynamic variables.
- We validate the abstraction by checking if the magnetization curve (phase transition)
  predicted by the rigid model holds for the vibrating system.

Hypothesis:
- At low vibrational temperature: The atoms stay near grid points. The abstraction is valid.
- At high vibrational temperature, atoms drift, changing J_eff. The rigid assumption breaks.
"""

import logging

import numpy as np

from causal_abstraction import LowLevelModel, SystemState

# Guard Numba
try:
    from numba import njit
except ImportError:
    def njit(func): return func

logger = logging.getLogger(__name__)


#Core simulation logic (Numba-optimized)
@njit
def _compute_forces(positions, grid_size, r0, spring_constant):
    """Calculates pairwise harmonic spring forces between nearby atoms (within 1.5*r0).

    These are pairwise springs between current near-neighbors, not tethers to fixed
    lattice sites, so under periodic boundaries global translation/shear are
    zero-energy modes (atoms are not rigidly anchored to their sites).
    """
    n_atoms = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dr = positions[i] - positions[j]
            # Periodic boundary conditions
            for dim in range(2):
                if dr[dim] > grid_size / 2: dr[dim] -= grid_size
                if dr[dim] < -grid_size / 2: dr[dim] += grid_size

            dist = np.sqrt(np.sum(dr**2))
            # Harmonic pairwise spring between nearby atoms (not a fixed-site tether)
            if dist < 1.5 * r0:
                force_mag = -spring_constant * (dist - r0)
                force_vec = force_mag * (dr / (dist + 1e-9))
                forces[i] += force_vec
                forces[j] -= force_vec
    return forces

@njit
def _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side):
    """Metropolis sweep where J depends on instantaneous distance."""
    n_atoms = spins.shape[0]
    for _ in range(n_atoms):
        i = np.random.randint(n_atoms)

        # Identify neighbors on the theoretical grid
        row, col = i // grid_side, i % grid_side
        neighbors = [
            ((row + 1) % grid_side) * grid_side + col,
            ((row - 1 + grid_side) % grid_side) * grid_side + col,
            row * grid_side + ((col + 1) % grid_side),
            row * grid_side + ((col - 1 + grid_side) % grid_side),
        ]

        neighbor_interaction_energy = 0
        for j in neighbors:
            dr = positions[i] - positions[j]
            # Periodic boundary conditions
            for dim in range(2):
                if dr[dim] > (grid_side * r0) / 2: dr[dim] -= (grid_side * r0)
                if dr[dim] < -(grid_side * r0) / 2: dr[dim] += (grid_side * r0)

            dist = np.sqrt(np.sum(dr**2))

            # Physical coupling: interaction strength decays with distance
            J_eff = J0 * np.exp(-j_decay * np.abs(dist - r0))
            neighbor_interaction_energy += -J_eff * spins[j]

        dE = 2 * spins[i] * (-neighbor_interaction_energy + field)

        if magnetic_temp > 1e-9 and (dE < 0 or np.random.rand() < np.exp(-dE / magnetic_temp)):
            spins[i] *= -1
        elif dE < 0:
            spins[i] *= -1

@njit
def _run_md_mc_simulation(grid_side, vibrational_temp, magnetic_temp, field, seed,
                          J0, r0, j_decay, mass, spring_constant, dt, k_boltzmann,
                          equil_sweeps, measure_sweeps):
    """
    Interleaves molecular dynamics (vibration) and Monte Carlo (Spin Flip).
    """
    np.random.seed(seed)
    n_atoms = grid_side * grid_side

    spins = np.ones(n_atoms, dtype=np.int8)

    positions = np.zeros((n_atoms, 2))
    for i in range(n_atoms):  # Initialize lattice positions
        positions[i] = np.array([(i % grid_side) * r0, (i // grid_side) * r0])

    np.random.seed(seed + 1)
    velocities = np.zeros_like(positions)

    # Initialize velocities if Vib > 0. This consumes RNG and creates a divergence compared to Vib=0 (intended)
    if vibrational_temp > 1e-9:
        velocities = np.random.randn(n_atoms, 2) * np.sqrt(k_boltzmann * vibrational_temp / mass)

    # Equilibration
    for _ in range(equil_sweeps):
        if vibrational_temp > 1e-9:
            forces = _compute_forces(positions, grid_side * r0, r0, spring_constant)
            positions += velocities * dt + 0.5 * (forces / mass) * dt**2
            new_forces = _compute_forces(positions, grid_side * r0, r0, spring_constant)
            velocities += 0.5 * ((forces + new_forces) / mass) * dt
        _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side)

    # Measurement
    mags = np.empty(measure_sweeps, dtype=np.float64)
    for i in range(measure_sweeps):
        if vibrational_temp > 1e-9:
            forces = _compute_forces(positions, grid_side * r0, r0, spring_constant)
            positions += velocities * dt + 0.5 * (forces / mass) * dt**2
            new_forces = _compute_forces(positions, grid_side * r0, r0, spring_constant)
            velocities += 0.5 * ((forces + new_forces) / mass) * dt
        _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side)
        mags[i] = np.abs(np.mean(spins))

    return np.mean(mags)


@njit
def _run_rigid_ising_sweep(spins, temp, field, J):
    """Standard Metropolis sweep for a rigid lattice (the high-level model logic)."""
    side_length = spins.shape[0]
    n_atoms = side_length * side_length
    for _ in range(n_atoms):
        idx = np.random.randint(n_atoms)
        x, y = idx // side_length, idx % side_length

        current_spin = spins[x, y]
        # Periodic boundaries
        neighbor_sum = (spins[(x + 1) % side_length, y] +
                        spins[(x - 1) % side_length, y] +
                        spins[x, (y + 1) % side_length] +
                        spins[x, (y - 1) % side_length])

        dE = 2 * current_spin * (J * neighbor_sum + field)

        if temp > 1e-9 and (dE < 0 or np.random.rand() < np.exp(-dE / temp)):
            spins[x, y] = -current_spin
        elif dE < 0:
            spins[x, y] = -current_spin

@njit
def _run_rigid_simulation(grid_size, temp, field, J, equil_sweeps, measure_sweeps, seed):
    """Runs the ideal (rigid) simulation."""
    np.random.seed(seed)
    spins = np.ones((grid_size, grid_size), dtype=np.int8)

    for _ in range(equil_sweeps):
        _run_rigid_ising_sweep(spins, temp, field, J)

    mags = np.empty(measure_sweeps, dtype=np.float64)
    for i in range(measure_sweeps):
        _run_rigid_ising_sweep(spins, temp, field, J)
        mags[i] = np.abs(np.mean(spins))
    return np.mean(mags)


# Low-level model
class MolecularDynamicsIsingModel(LowLevelModel):
    """Low-level model M: an Ising spin lattice with molecular-dynamics (vibrational) updates."""

    def __init__(self, grid_side=8, vibrational_temp=1.0):
        self.grid_side = grid_side
        self.vibrational_temp = vibrational_temp

        # Simulation parameters
        self.params = {
            'equil_sweeps': 2000,
            'measure_sweeps': 1000,
            'J0': 1.0, 'r0': 1.0, 'j_decay': 2.0,
            'mass': 1.0, 'spring_constant': 50.0, 'dt': 0.01, 'k_boltzmann': 1.0
        }

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        def get(name, default):
            val = interventions.get(name, input_state.get(name, default))
            return val.item() if hasattr(val, 'item') else val

        t = get('Temperature', 0.0)
        h = get('ExternalField', 0.0)

        # Deterministic seed based on input conditions for reproducibility (common
        # random numbers across M and E). This depends only on t+h, so distinct
        # interventions with equal Temperature+ExternalField collide to the same seed.
        seed = int(round(t, 8) * 1e8) + int(round(h + 10, 8) * 1e8)

        # Run hybrid conditions
        avg_mag = _run_md_mc_simulation(
            grid_side=self.grid_side,
            vibrational_temp=self.vibrational_temp,
            magnetic_temp=t,
            field=h,
            seed=seed,
            J0=self.params['J0'], r0=self.params['r0'], j_decay=self.params['j_decay'],
            mass=self.params['mass'], spring_constant=self.params['spring_constant'],
            dt=self.params['dt'], k_boltzmann=self.params['k_boltzmann'],
            equil_sweeps=self.params['equil_sweeps'],
            measure_sweeps=self.params['measure_sweeps']
        )

        return input_state.merge(SystemState(values={
            'Temperature': np.array([t]),
            'ExternalField': np.array([h]),
            'PredictedMagnetization': np.array([avg_mag])
        }))


#Main
def main():
    """Simulate the Ising model for various vibrational temperatures."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    vibrational_temps = [0, 0.05, 0.1, 0.2]
    grid_side = 8
    sample_temps = [1.5, 2.269, 3.5]  # Below, at, and above Curie temperature

    print(f"Ising model simulation (grid={grid_side}x{grid_side})")
    print(f"{'vib_temp':<12} | {'T=1.5 mag':<12} | {'T=2.27 mag':<12} | {'T=3.5 mag':<12}")
    print("-" * 55)

    for vib_temp in vibrational_temps:
        low_level_model = MolecularDynamicsIsingModel(grid_side=grid_side, vibrational_temp=vib_temp)
        mags = []
        for temp in sample_temps:
            state = low_level_model.forward_with_interventions(
                SystemState(values={}),
                {"Temperature": np.array([[temp]]), "ExternalField": np.array([[0.0]])},
            )
            mag = float(np.asarray(state.values.get("PredictedMagnetization", [[0]])).ravel()[0])
            mags.append(mag)
        print(f"{vib_temp:<12.3f} | {mags[0]:<12.4f} | {mags[1]:<12.4f} | {mags[2]:<12.4f}")


if __name__ == '__main__':
    main()
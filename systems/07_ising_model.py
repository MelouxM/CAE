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
def _lattice_forces(positions, side, r0, spring_constant):
    """Harmonic springs of a 2D square lattice, fixed by the lattice topology.

    Each atom is bonded to its 4 nearest neighbors (rest length r0) and its 4 diagonal
    neighbors (rest length sqrt(2) r0), with periodic boundaries. Without the diagonals
    the lattice would have no shear stiffness.
    """
    n_atoms = positions.shape[0]
    box = side * r0
    forces = np.zeros_like(positions)
    diag = np.sqrt(2.0) * r0
    for i in range(n_atoms):
        row, col = i // side, i % side
        # each bond once: right, down, down-right, down-left
        for k in range(4):
            if k == 0:
                j = row * side + (col + 1) % side
                rest = r0
            elif k == 1:
                j = ((row + 1) % side) * side + col
                rest = r0
            elif k == 2:
                j = ((row + 1) % side) * side + (col + 1) % side
                rest = diag
            else:
                j = ((row + 1) % side) * side + (col - 1 + side) % side
                rest = diag
            dr = positions[i] - positions[j]
            for dim in range(2):
                dr[dim] -= box * np.round(dr[dim] / box)
            dist = np.sqrt(np.sum(dr ** 2))
            force_vec = -spring_constant * (dist - rest) * (dr / (dist + 1e-12))
            forces[i] += force_vec
            forces[j] -= force_vec
    return forces


@njit
def _nn_distance(positions, i, j, side, r0):
    box = side * r0
    dr = positions[i] - positions[j]
    for dim in range(2):
        dr[dim] -= box * np.round(dr[dim] / box)
    return np.sqrt(np.sum(dr ** 2))


@njit
def _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side):
    """Metropolis sweep where J depends on the instantaneous nearest-neighbor distance."""
    n_atoms = spins.shape[0]
    for _ in range(n_atoms):
        i = np.random.randint(n_atoms)

        # Neighbors on the lattice topology
        row, col = i // grid_side, i % grid_side
        neighbors = (
            ((row + 1) % grid_side) * grid_side + col,
            ((row - 1 + grid_side) % grid_side) * grid_side + col,
            row * grid_side + ((col + 1) % grid_side),
            row * grid_side + ((col - 1 + grid_side) % grid_side),
        )

        local_field = 0.0
        for j in neighbors:
            dist = _nn_distance(positions, i, j, grid_side, r0)
            # Physical coupling: interaction strength decays with distance
            J_eff = J0 * np.exp(-j_decay * np.abs(dist - r0))
            local_field += J_eff * spins[j]

        # H = -sum J_ij s_i s_j - h sum s_i  ->  flip cost 2 s_i (sum_j J_ij s_j + h)
        dE = 2 * spins[i] * (local_field + field)

        if dE < 0 or (magnetic_temp > 1e-9 and np.random.rand() < np.exp(-dE / magnetic_temp)):
            spins[i] *= -1


@njit
def _md_steps(positions, velocities, n_steps, side, r0, spring_constant, mass, dt, vib_temp, gamma):
    """Velocity Verlet + Langevin thermostat at the vibrational temperature."""
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 ** 2) * vib_temp / mass)
    forces = _lattice_forces(positions, side, r0, spring_constant)
    for _ in range(n_steps):
        velocities += 0.5 * (forces / mass) * dt
        positions += velocities * dt
        forces = _lattice_forces(positions, side, r0, spring_constant)
        velocities += 0.5 * (forces / mass) * dt
        for a in range(velocities.shape[0]):
            for d in range(2):
                velocities[a, d] = c1 * velocities[a, d] + c2 * np.random.normal(0.0, 1.0)


@njit
def _run_md_mc_simulation(grid_side, vibrational_temp, magnetic_temp, field, seed,
                          J0, r0, j_decay, mass, spring_constant, dt, md_steps_per_sweep,
                          gamma, equil_sweeps, measure_sweeps):
    """
    Interleaves molecular dynamics (thermostatted lattice vibration) and Monte Carlo
    (spin flips). Returns the mean |magnetization| over the measurement sweeps and the
    mean nearest-neighbor coupling the spins actually saw.
    """
    np.random.seed(seed)
    n_atoms = grid_side * grid_side
    spins = np.ones(n_atoms, dtype=np.int8)

    positions = np.zeros((n_atoms, 2))
    for i in range(n_atoms):  # Initialize lattice positions
        positions[i, 0] = (i % grid_side) * r0
        positions[i, 1] = (i // grid_side) * r0
    velocities = np.zeros_like(positions)
    vibrating = vibrational_temp > 1e-9
    if vibrating:
        velocities = np.random.randn(n_atoms, 2) * np.sqrt(vibrational_temp / mass)
        # thermalize the lattice before the spins start
        _md_steps(positions, velocities, 20 * md_steps_per_sweep, grid_side, r0,
                  spring_constant, mass, dt, vibrational_temp, gamma)

    for _ in range(equil_sweeps):
        if vibrating:
            _md_steps(positions, velocities, md_steps_per_sweep, grid_side, r0,
                      spring_constant, mass, dt, vibrational_temp, gamma)
        _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side)

    mag_sum = 0.0
    j_sum = 0.0
    for _ in range(measure_sweeps):
        if vibrating:
            _md_steps(positions, velocities, md_steps_per_sweep, grid_side, r0,
                      spring_constant, mass, dt, vibrational_temp, gamma)
        _run_mc_sweep_distance_dependent(positions, spins, magnetic_temp, field, J0, r0, j_decay, grid_side)
        mag_sum += np.abs(np.mean(spins))
        # mean coupling over the right/down bonds (each bond once)
        acc = 0.0
        for i in range(n_atoms):
            row, col = i // grid_side, i % grid_side
            for j in (row * grid_side + (col + 1) % grid_side, ((row + 1) % grid_side) * grid_side + col):
                acc += J0 * np.exp(-j_decay * np.abs(_nn_distance(positions, i, j, grid_side, r0) - r0))
        j_sum += acc / (2 * n_atoms)

    return mag_sum / measure_sweeps, j_sum / measure_sweeps


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


def input_seed(*values):
    """Deterministic seed from the exact float inputs."""
    words = np.frombuffer(np.asarray(values, dtype=np.float64).tobytes(), dtype=np.uint32)
    return int(np.random.SeedSequence(words.tolist()).generate_state(1)[0] % (2 ** 31 - 1))


# Low-level model
class MolecularDynamicsIsingModel(LowLevelModel):
    """Low-level model M: an Ising spin lattice on a vibrating, thermostatted atomic lattice.

    Args:
        grid_side: Side of the square lattice (grid_side^2 atoms/spins).
        vibrational_temp: Temperature of the lattice heat bath (fixed; the magnetic
            temperature is the intervened `Temperature`). 0 freezes the lattice, which
            makes the model identical to the rigid Ising model.
    """

    def __init__(self, grid_side=8, vibrational_temp=1.0):
        self.grid_side = grid_side
        self.vibrational_temp = vibrational_temp

        # Simulation parameters
        self.params = {
            'equil_sweeps': 2000,
            'measure_sweeps': 1000,
            'J0': 1.0, 'r0': 1.0, 'j_decay': 2.0,
            'mass': 1.0, 'spring_constant': 50.0, 'dt': 0.02,
            'md_steps_per_sweep': 5, 'gamma': 1.0,
        }

    def _simulate(self, t, h, seed):
        return _run_md_mc_simulation(
            grid_side=self.grid_side,
            vibrational_temp=self.vibrational_temp,
            magnetic_temp=t,
            field=h,
            seed=seed,
            J0=self.params['J0'], r0=self.params['r0'], j_decay=self.params['j_decay'],
            mass=self.params['mass'], spring_constant=self.params['spring_constant'],
            dt=self.params['dt'], md_steps_per_sweep=self.params['md_steps_per_sweep'],
            gamma=self.params['gamma'],
            equil_sweeps=self.params['equil_sweeps'],
            measure_sweeps=self.params['measure_sweeps'],
        )

    def effective_coupling(self, n_sweeps=500, seed=0):
        """Mean nearest-neighbor coupling <J0 exp(-decay |d - r0|)> on the vibrating lattice.

        The lattice doesn't feel the spins, so this only depends on the lattice. It is
        the coupling the rigid model should use if vibrations are fast.
        """
        saved = dict(self.params)
        self.params['equil_sweeps'], self.params['measure_sweeps'] = 0, n_sweeps
        try:
            _, j_eff = self._simulate(2.0, 0.0, seed)
        finally:
            self.params = saved
        return float(j_eff)

    def forward_with_interventions(self, input_state: SystemState, interventions: dict,
                                   rng=None) -> SystemState:
        def get(name, default):
            val = interventions.get(name, input_state.get(name, default))
            return np.atleast_1d(np.asarray(val, dtype=float)).ravel()

        # One simulation per batch element
        ts, hs = get('Temperature', 0.0), get('ExternalField', 0.0)
        n = max(len(ts), len(hs))
        ts, hs = np.broadcast_to(ts, n), np.broadcast_to(hs, n)
        mags = np.empty(n)
        for i, (t, h) in enumerate(zip(ts, hs)):
            seed = int(rng.integers(2 ** 31 - 1)) if rng is not None else input_seed(t, h)
            mags[i], _ = self._simulate(float(t), float(h), seed)

        return input_state.merge(SystemState(values={
            'Temperature': np.array(ts, dtype=float),
            'ExternalField': np.array(hs, dtype=float),
            'PredictedMagnetization': mags
        }))


#Main
def main():
    """Effective coupling and magnetization for a few vibrational temperatures."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    grid_side = 16
    sample_temps = [1.5, 2.0, 2.5, 3.5]
    print(f"Ising model on a vibrating lattice (grid={grid_side}x{grid_side})")
    for vib_temp in [0.0, 0.25, 0.5, 1.0]:
        m = MolecularDynamicsIsingModel(grid_side=grid_side, vibrational_temp=vib_temp)
        m.params['equil_sweeps'], m.params['measure_sweeps'] = 500, 1000
        j_eff = m.effective_coupling()
        mags = []
        for t in sample_temps:
            ll = m._simulate(t, 0.0, input_seed(t, 0.0))[0]
            hl = _run_rigid_simulation(grid_side, t, 0.0, j_eff, 500, 1000, input_seed(t, 1.0))
            mags.append(f"T={t}: {ll:.3f}/{hl:.3f}")
        print(f"vib_temp={vib_temp:<5} J_eff={j_eff:.3f}  (vibrating / rigid with J_eff)  " + "  ".join(mags))


if __name__ == '__main__':
    main()

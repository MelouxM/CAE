"""
Example 4: Evaluating an abstraction under varying low-level realism.
We quantify how "true" the Lotka-Volterra equations are as an explanation for
Agent-Based Models (ABMs) of varying complexity.

Low-level models:
- A suite of 8 Agent-Based Models of predator-prey dynamics.
- They vary across 3 axes of realism:
  1. Spatial: Particles moving on a grid vs. global mixing
  2. Stochastic: Probabilistic birth/death vs. deterministic rates
  3. Aging: Agents die from old age vs. constant death rate

High-level model:
- The Lotka-Volterra (LV) differential equations.
- Variables: Continuous populations of Prey and Predators.

Coarse-graining:
- The high-dimensional micro-state (the set of all individual agents, their locations,
and status) are mapped to low-dimensional macro-variables (total prey/predator population).
- In spatial variants, this marginalizes out the spatial distribution and treats
the system as a mean field.

Value map:
- Continuous identity mapping.
- Discrete integer agent counts are treated as continuous real-valued variables (N -> N.0).
  This relies on the law of large numbers to approximate the continuum assumption
  of the differential equations.

Additional details:
- We calibrate the LV parameters (alpha, beta, gamma, delta) once against the
  simplest, most idealized ABM (Non-spatial, Deterministic, No Aging).
- We then freeze this high-level model and evaluate its error against all 8 ABM variants.
- This measures how much the textbook abstraction degrades as real-world
  complexities (space, noise, biology) are introduced.
"""

import itertools

import numpy as np
import numba
from tqdm import tqdm

from causal_abstraction import LowLevelModel, SystemState


# Simulation kernels (Numba-accelerated)

@numba.njit
def _run_nonspatial_step(prey, pred, params, use_stochastic, use_aging):
    # Unpack params for speed
    r_prey = params[0] # prey_reproduce
    p_pred = params[1] # predation
    r_pred = params[2] # pred_reproduce
    s_pred = params[3] # pred_starve
    d_prey_age = params[4]
    d_pred_age = params[5]

    if prey < 1 or pred < 1:
        return 0.0, 0.0

    if use_stochastic:
        births = np.random.binomial(int(prey), r_prey)  # Stochastic (binomial) dynamics

        # Predation interactions (capped below at the number of prey)
        interactions = int(prey) * int(pred)

        eaten = np.random.binomial(interactions, p_pred)
        if eaten > prey:  # Cap eaten at number of prey
            eaten = int(prey)

        pred_births = np.random.binomial(eaten, r_pred)
        pred_starve = np.random.binomial(int(pred), s_pred)

        age_death_prey = 0
        age_death_pred = 0
        if use_aging:
            age_death_prey = np.random.binomial(int(prey - eaten), d_prey_age)
            age_death_pred = np.random.binomial(int(pred), d_pred_age)

        new_prey = prey + births - eaten - age_death_prey
        new_pred = pred + pred_births - pred_starve - age_death_pred

    else:
        # Deterministic (float) dynamics
        births = prey * r_prey
        eaten = prey * pred * p_pred
        pred_births = eaten * r_pred
        pred_starve = pred * s_pred

        age_death_prey = 0.0
        age_death_pred = 0.0
        if use_aging:
            age_death_prey = (prey - eaten) * d_prey_age
            age_death_pred = pred * d_pred_age

        new_prey = prey + births - eaten - age_death_prey
        new_pred = pred + pred_births - pred_starve - age_death_pred

    return max(0.0, new_prey), max(0.0, new_pred)

@numba.njit
def _run_spatial_simulation(steps, init_prey, init_pred, grid_size, params_array, use_aging):
    """
    Grid-based simulation. State: 0=Empty, 1=Prey, 2=Predator
    """
    # Unpack
    r_prey = params_array[0]
    # p_pred is not used directly in spatial loop (interaction implies predation)
    r_pred = params_array[2]
    s_pred = params_array[3]
    d_prey_age = params_array[4]
    d_pred_age = params_array[5]

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)  # Initialize grid

    # Random spawn using rejection sampling (fast enough)
    placed = 0
    while placed < init_prey:
        r, c = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        if grid[r, c] == 0:
            grid[r, c] = 1
            placed += 1

    placed = 0
    while placed < init_pred:
        r, c = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        if grid[r, c] == 0:
            grid[r, c] = 2
            placed += 1

    # Run loop
    for _ in range(steps):
        # We iterate over a shuffled list of occupied cells
        pop_indices = []
        for r in range(grid_size):  # Numba doesn't support np.argwhere nicely with shuffle
            for c in range(grid_size):
                if grid[r, c] != 0:
                    pop_indices.append((r, c))

        # Create random order
        n_pop = len(pop_indices)
        if n_pop == 0:
            break

        order = np.random.permutation(n_pop)

        prey_birth_queue = 0
        pred_birth_queue = 0

        for idx in order:
            r, c = pop_indices[idx]
            val = grid[r, c]
            if val == 0:
                continue  # Might have been eaten/died this turn

            is_prey = (val == 1)

            # Death logic
            died = False
            if is_prey:
                if use_aging and np.random.rand() < d_prey_age:
                    grid[r, c] = 0; died = True
            else:
                # Predators starve or die of age
                prob_death = s_pred + (d_pred_age if use_aging else 0)
                if np.random.rand() < prob_death:
                    grid[r, c] = 0; died = True

            if died:
                continue

            # Reproduction accumulator
            if is_prey:
                if np.random.rand() < r_prey:
                    prey_birth_queue += 1

            # Movement / interaction
            # Pick random neighbor
            dr = np.random.randint(-1, 2) # -1, 0, 1
            dc = np.random.randint(-1, 2)
            if dr == 0 and dc == 0:
                continue  # No move

            nr, nc = (r + dr) % grid_size, (c + dc) % grid_size
            target = grid[nr, nc]

            if target == 0:  # Move to empty
                grid[nr, nc] = val
                grid[r, c] = 0
            elif is_prey and target == 2:  # Prey walks into predator -> Eaten
                grid[r, c] = 0
                if np.random.rand() < r_pred: pred_birth_queue += 1
            elif not is_prey and target == 1:  # Predator walks into prey -> Eat
                grid[nr, nc] = 2  # Predator takes spot
                grid[r, c] = 0
                if np.random.rand() < r_pred: pred_birth_queue += 1

        # Resolve births by spawning into globally-selected empty cells
        empties = []
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == 0:
                    empties.append((r, c))

        n_empty = len(empties)
        if n_empty > 0:
            spawn_order = np.random.permutation(n_empty)
            ptr = 0

            # Spawn prey
            to_spawn = min(prey_birth_queue, n_empty)
            for _ in range(to_spawn):
                rr, cc = empties[spawn_order[ptr]]
                grid[rr, cc] = 1
                ptr += 1

            # Spawn predators
            remaining = n_empty - ptr
            to_spawn = min(pred_birth_queue, remaining)
            for _ in range(to_spawn):
                rr, cc = empties[spawn_order[ptr]]
                grid[rr, cc] = 2
                ptr += 1

    # Count
    c_prey = 0
    c_pred = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 1: c_prey += 1
            elif grid[r, c] == 2: c_pred += 1

    return float(c_prey), float(c_pred)


# Low-level model wrappers

class AgentBasedModel(LowLevelModel):
    """Low-level model M: a predator-prey agent-based model with configurable realism axes."""

    def __init__(self, config_dict):
        self.cfg = config_dict
        # Prepare parameters as array for Numba
        # [r_prey, p_pred, r_pred, s_pred, d_prey_age, d_pred_age]
        self.params_arr = np.array([
            self.cfg['prey_reproduce_prob'],
            self.cfg['predation_prob'],
            self.cfg['predator_reproduce_prob'],
            self.cfg['predator_starvation_prob'],
            self.cfg.get('prey_age_death_prob', 0),
            self.cfg.get('predator_age_death_prob', 0)
        ], dtype=np.float64)

        self.steps = self.cfg['num_steps']
        self.grid_size = self.cfg['grid_size']

        self.spatial = self.cfg.get('spatial', False)
        self.stochastic = self.cfg.get('stochastic_reproduction', False)
        self.aging = self.cfg.get('agent_aging', False)

    def step(self, state: SystemState) -> SystemState:
        """
        Advance the ABM by one macro-step (self.steps ABM ticks).

        Reads initial populations in priority order:
          1. state.final_populations  (output of a previous step - enables chaining)
          2. state.prey_t / state.predator_t  (explicit initial conditions)
          3. defaults: (100, 40)

        Always writes results back to prey_t and predator_t as well as
        final_populations so the returned state can be passed directly
        to the next step() call without any manual field promotion.
        """
        if 'final_populations' in state.values:
            pops = np.asarray(state.values['final_populations']).ravel()
            prey = float(pops[0])
            pred = float(pops[1]) if len(pops) > 1 else 40.0
        else:
            prey = float(np.asarray(state.values.get('prey_t', [100.0])).ravel()[0])
            pred = float(np.asarray(state.values.get('predator_t', [40.0])).ravel()[0])

        result = self.forward_with_interventions(
            SystemState(),
            {'prey_t': np.array([prey]), 'predator_t': np.array([pred])},
        )
        pops = np.asarray(result.values['final_populations']).ravel()
        result.values['prey_t'] = np.array([float(pops[0])])
        result.values['predator_t'] = np.array([float(pops[1]) if len(pops) > 1 else 0.0])
        return result

    def forward_with_interventions(self, input_state: SystemState, interventions: dict) -> SystemState:
        # Resolve inputs (handling defaults and interventions)
        def get_input(name, default):
            val = interventions.get(name, input_state.get(name, default))
            return np.atleast_1d(val)

        prey_in = get_input('prey_t', 100.0)
        pred_in = get_input('predator_t', 40.0)

        # Broadcast to batch size
        batch_size = max(len(prey_in), len(pred_in))
        if len(prey_in) < batch_size:
            prey_in = np.resize(prey_in, batch_size)
        if len(pred_in) < batch_size:
            pred_in = np.resize(pred_in, batch_size)

        results = np.zeros((batch_size, 2))

        # Run batch
        for i in range(batch_size):
            p = prey_in[i].item()
            pr = pred_in[i].item()

            if self.spatial:
                f_prey, f_pred = _run_spatial_simulation(
                    self.steps, int(p), int(pr), self.grid_size, self.params_arr, self.aging
                )
            else:  # Non-spatial
                curr_p, curr_pr = p, pr
                for _ in range(self.steps):
                    curr_p, curr_pr = _run_nonspatial_step(
                        curr_p, curr_pr, self.params_arr, self.stochastic, self.aging
                    )
                f_prey, f_pred = curr_p, curr_pr

            results[i, 0] = f_prey
            results[i, 1] = f_pred

        return input_state.merge(SystemState(values={
            'prey_t': prey_in,
            'predator_t': pred_in,
            'final_populations': results
        }))


# High-level model (Lotka-Volterra)

def solve_lotka_volterra(prey_t, predator_t, high_level_model_params):
    # This function receives scalar inputs (called by the engine for each batch item)
    prey, pred = float(prey_t), float(predator_t)
    dt_steps = high_level_model_params['dt']

    # Euler integration to match discrete steps of low-level model
    for _ in range(dt_steps):
        if prey < 1e-3 or pred < 1e-3:
            break
        d_prey = (high_level_model_params['alpha'] * prey) - (high_level_model_params['beta'] * prey * pred)
        d_pred = (high_level_model_params['delta'] * prey * pred) - (high_level_model_params['gamma'] * pred)
        prey += d_prey
        pred += d_pred

    return np.array([max(0.0, prey), max(0.0, pred)])

def calibrate_high_level_model(low_level_model_ideal, search_space):
    """
    Fits LV parameters to the "ideal" low-level model (non-spatial, deterministic).
    """
    print("Calibrating high-level model on ideal low-level model")

    # Target: Run the Ideal low-level model from a standard starting point
    init_state = SystemState(values={
        'prey_t': np.array([100.0]),
        'predator_t': np.array([40.0])
    })

    res = low_level_model_ideal.forward(init_state)
    target_vec = res['final_populations'][0]

    print(f"Target outcome: Prey={target_vec[0]:.1f}, Pred={target_vec[1]:.1f}")

    # Grid search
    param_names = list(search_space.keys())
    grid = list(itertools.product(*search_space.values()))

    best_p, min_err = None, float('inf')

    for p_tuple in tqdm(grid, desc="Fitting params"):
        curr = dict(zip(param_names, p_tuple))
        curr['dt'] = low_level_model_ideal.steps

        high_level_model_out = solve_lotka_volterra(100.0, 40.0, curr)
        err = np.sum((high_level_model_out - target_vec)**2)

        if err < min_err:
            min_err = err
            best_p = curr

    print(f"Best Parameters: {best_p}")
    return best_p


def main():
    """Simulate predator-prey ABM for ideal and complex configurations."""
    low_level_model_base_config = {
        "prey_reproduce_prob": 0.1,
        "predation_prob": 0.001,
        "predator_reproduce_prob": 0.5,
        "predator_starvation_prob": 0.05,
        "num_steps": 50,
        "grid_size": 20,
        "prey_age_death_prob": 0.02,
        "predator_age_death_prob": 0.04,
    }

    toggles = [False, True]
    print("Predator-prey ABM simulation (prey_0=100, pred_0=50):")
    print(f"{'Configuration':<35} | {'Final prey':<12} | {'Final predators'}")
    print("-" * 65)

    for spatial, stochastic, aging in itertools.product(toggles, toggles, toggles):
        cfg = low_level_model_base_config.copy()
        cfg.update({'spatial': spatial, 'stochastic_reproduction': stochastic, 'agent_aging': aging})
        low_level_model = AgentBasedModel(cfg)

        features = []
        if spatial:    features.append("Spatial")
        if stochastic: features.append("Stoch")
        if aging:      features.append("Aging")
        name = " + ".join(features) if features else "Ideal (deterministic)"

        state = low_level_model.forward_with_interventions(
            SystemState(values={}),
            {"prey_t": np.array([[100.0]]), "predator_t": np.array([[50.0]])},
        )
        pops = np.asarray(state.values.get("final_populations", [[0, 0]])).ravel()
        print(f"{name:<35} | {pops[0]:<12.1f} | {pops[1]:.1f}")


if __name__ == '__main__':
    main()
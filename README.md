# Causal Abstraction Framework

[![PyPI](https://img.shields.io/pypi/v/causal-abstraction-eval.svg)](https://pypi.org/project/causal-abstraction-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for **validating causal abstractions on simulated complex systems**. It accompanies
the paper *"Validating Causal Abstraction Metrics on Simulated Complex Systems"* and provides a unified
implementation of a benchmark of idealized complex systems, their ground-truth abstractions, and a
large battery of candidate validity metrics including the paper's contribution, the **Causal
Abstraction Error (CAE)**.

> The paper is the source of truth for all concepts, terminology, and notation. This
> README mirrors that notation; where the two differ, trust the paper.

---

### Key concepts

We reason about complex systems (a neural network, a physics simulation, a digital circuit, a CPU)
using simplified high-level models. The central question is: *when is a high-level causal model a
**valid** explanation of a low-level system?*

The framework formalizes this as **constructive causal abstraction**. A low-level structural causal
model `M` is abstracted by a high-level model `E` through a map `τ` (a variable coarse-graining `a`
plus per-variable value maps `τ_X`). Validity is tested on the **commuting diagram**: intervening on
`E` should yield the same result as grounding that intervention into `M`, running `M`, and abstracting
the result back up.

The paper's central metric is the **Causal Abstraction Error (CAE)**, a continuous validity score with
a built-in faithfulness test over unmapped variables `Φ` (variables claimed to be causally inert
for the abstraction). It comes in two directions and two forms, all exposed through `StandardTasks`:

*   **`CAE↑` (`CAE_up`)** (bottom-up): micro-interventions are sampled in `M`'s space and lifted via
    `τ_X` (`BottomUpSampler`).
*   **`CAE↓` (`CAE_down`)** (top-down): macro-interventions are sampled in `E`'s domain and grounded
    into `M` (`TopDownSampler`).
*   Each runs in a faithful form (the canonical CAE, which also perturbs the `Φ` variables to
    detect leakage) and a non-faithful ablation (`CAE_up_nf` / `CAE_down_nf`).

The paper benchmarks 30+ candidate metrics from observational, functional, information-theoretic, and
causal families, and shows that only causal-abstraction metrics *with* faithfulness testing reliably
discriminate valid from invalid abstractions.

### Installation

Install the latest release from [PyPI](https://pypi.org/project/causal-abstraction-eval/):

```bash
pip install causal_abstraction_eval
```

To install from source instead — for example to run the benchmark suites, which are not part of
the distribution (see [Distribution contents](#distribution-contents)):

```bash
git clone https://github.com/MelouxM/CAE.git
cd CAE
pip install .
```

For development (editable install with the test tooling):

```bash
pip install -e ".[dev]"
```

To reproduce the paper's exact numbers, install the pinned environment snapshot
rather than the loose ranges in `pyproject.toml`:

```bash
pip install -r requirements.lock
```

`requirements.lock` is a fully-resolved capture of the environment the benchmark
results were produced in (it excludes this package itself and `tracr`; install
those as above); `pyproject.toml` keeps deliberately loose ranges for downstream
consumers.

Optional dependency groups (extras): `neural` (PyTorch), `physics` (numba), `circuit` (PySpice),
`figures` (matplotlib, for the figure generators), `dev` (pytest), and `all` (every system
dependency installable from PyPI). Each benchmark system pulls in only the extras it needs.

The Tracr system (`systems/08_tracr.py`) additionally requires `tracr`, DeepMind's
RASP→transformer compiler. It is not published on PyPI (and a direct git reference would make this
package unpublishable to PyPI), so install it manually:

```bash
pip install git+https://github.com/google-deepmind/tracr.git
```

> Requires Python ≥ 3.11. The import name is `causal_abstraction`; the distribution name is
> `causal_abstraction_eval`.

### Distribution contents

The published wheel (and sdist) ship only the importable `causal_abstraction` framework, not the
benchmark itself. Reproducing the benchmark requires a Git clone rather than a PyPI install.

### Tutorial

New to the framework? Start with [`tutorial.ipynb`](tutorial.ipynb), a runnable notebook that builds a
minimal abstraction from scratch, evaluates one of the provided benchmark systems, and runs a selection
of baseline metrics, all in the same `M` / `E` / `τ` notation as the paper.

### Benchmark systems and runnable suites

The benchmark comprises ten complex systems (rows 01–10 below) plus a suite of controlled
contrastive experiments (row 11). Each system is defined under `systems/` and evaluated by a matching
suite under `test/`; the `test/NN_*.py` script loads `systems/NN_*.py` by path and runs the full metric
battery across conditions (`valid`, plus invalid contrastive conditions such as `fail`, `inv_internal`,
`noise`), persisting results to `test/results/NN_*_results.json`.

| # | System | Definition | Suite |
|---|---|---|---|
| 01 | Logic circuit (2-bit adder) | `systems/01_logic_circuit.py` | `test/01_logic_circuit.py` |
| 02 | Transistor circuit (SPICE) | `systems/02_transistor_circuit.py` | `test/02_transistor_circuit.py` |
| 03 | Gas simulation (Lennard-Jones → ideal/Van der Waals gas) | `systems/03_gas_simulation.py` | `test/03_gas_simulation.py` |
| 04 | Predator–prey (agent-based → Lotka–Volterra) | `systems/04_predator_prey.py` | `test/04_predator_prey.py` |
| 05 | Heat equation (1D) | `systems/05_heat_equation.py` | `test/05_heat_equation.py` |
| 06 | Heat equation (2D) | `systems/06_heat_equation_2d.py` | `test/06_heat_equation_2d.py` |
| 07 | Ising model | `systems/07_ising_model.py` | `test/07_ising_model.py` |
| 08 | Tracr compiled transformer | `systems/08_tracr.py` | `test/08_tracr.py` |
| 09 | Gene regulatory network | `systems/09_grn/grn.py` | `test/09_grn.py` |
| 10 | MOS 6502 CPU | `systems/10_cpu_6502.py` | `test/10_cpu_6502.py` |
| 11 | Controlled contrastive experiments | `systems/11_controlled_experiments.py` | `test/11_controlled_experiments.py` |

> Note 1: The gene regulatory network does not follow the flat `systems/NN_<name>.py` convention
> and is stored at `systems/09_grn/grn.py` (its suite `test/09_grn.py` loads it specially).

> Note 2: system 10 (MOS 6502) requires native libraries built from three upstream emulators that
> are not vendored in this repo. Before running `test/10_cpu_6502.py`, follow
> [`systems/10_cpu_6502_libs/README.md`](systems/10_cpu_6502_libs/README.md) to fetch the pinned
> upstream sources and run `build_libs.sh`.

Run a suite forever (until `Ctrl-C`) or for a bounded number of runs:

```bash
python test/01_logic_circuit.py            # run until interrupted
python test/01_logic_circuit.py --runs 10  # bounded
```

Companion scripts under `test/` cover statistical power / discrimination analysis
(`power_*.py`, `run_power.py`) and figure generation (`figures/get_figure_*.py`, needs the
`figures` extra). The unit-test suite (`test/test_*.py`) is run with `pytest`.

The `test/` directory is organized as follows:

* `test/test_*.py`, `test/conftest.py`: the pytest unit suite.
* `test/NN_*.py`, `test/power_*.py` and the shared infrastructure (`runner.py`, `utils.py`,
  `power_runner.py`, `run_power.py`, `tracr_size.py`): the runnable evaluation and power suites.
* `test/results/`: where the suites write benchmark results (`*_results.json`, `power_*.json`)
  when run; generated locally and not committed to the repo.
* `test/figures/`: figure-generation utilities (`get_figure_*.py`, `figure_utils.py`) and the
  generated PDFs they produce.
* `test/artifacts/`: generated runtime artifacts (e.g. cached binaries).

### Library architecture

The library (`causal_abstraction/`) is organized as follows; the canonical public API is whatever
`causal_abstraction/__init__.py` re-exports.

*   **`primitives.py`**: core data types (`SystemState`, `AbstractVariable`, probability
    distributions, and the `UNMAPPED` sentinel).
*   **`schema.py`**: the low-level variable layout (`MicroVariableSchema`, `Variable`,
    `MicroSelector`) and the coarse-graining map `CoarseGrainingMap` (the variable map `a`, which also
    defines the unmapped set `Φ` and the internal variables).
*   **`valuemap.py`**: the value map `τ_X` (`ValueMap`, `ContinuousValueMap`): `abstract()` lifts
    micro-states to macro-labels, `ground()` does the reverse.
*   **`models/`**: `high_level.py` defines the high-level SCM `E` (`CausalGraph`); `low_level.py`
    defines the low-level SCM `M` (`LowLevelModel`, `NoisyLowLevelModel`); `neural.py` wraps PyTorch
    models (`NeuralModel`).
*   **`spaces/base.py`**: subspace geometry used to define variable domains (`RectSubspace`,
    `SphereSubspace`, `UnionSubspace`, `ComplementSubspace`, `FullSubspace`, `UniformSubspace`,
    `GaussianSubspace`).
*   **`sampling.py`**: the intervention samplers `TopDownSampler` (`CAE↓`), `BottomUpSampler` (`CAE↑`),
    `CombinedFaithfulnessSampler` (inline `Φ` faithfulness test), plus `PairedSampler` and
    `NoisyMeasurementSampler`.
*   **`paths.py`**: the commuting-diagram traversal (`DiagramBuilder`, `CausalPath`).
*   **`metrics.py`**: the dissimilarity / divergence family `D` (`MSEMetric`, `L2Metric`,
    `JSDivergenceMetric`, `KLDivergenceMetric`, `MMDMetric`, …) plus dynamical metrics
    (`TrajectoryMSEMetric`, `DTWMetric`, `TemporalAutocorrelationMetric`, `SpectralMetric`) and
    `get_metric`.
*   **`analytical_metrics.py`**: analytical baseline metrics (`IIAMetric`, `BCCMetric`,
    `DCCMetric`, `SobolSensitivityMetric`, `IBLagrangianMetric`, …) drawn from the four metric
    families compared in the paper. With the dissimilarity metrics above and the `CAE↑`/`CAE↓`
    tasks, these make up the 30+ candidate metrics benchmarked in the paper.
*   **`tasks.py`**: `EvaluationTask`, `TaskResults`, and `StandardTasks`, the factory exposing the two
    causal-abstraction tasks `CAE↑` / `CAE↓` (faithful and non-faithful) plus observational baselines.
*   **`engine.py`**: the orchestrator `EvaluationEngine` (`run_tasks`, `run_analytical_metrics`).
*   **`config.py`**: `EvaluationConfig`.
*   **`experiment.py`**: `ExperimentResults`.

#### Auxiliary diagnostics

`PrecisionMetric` (in `metrics.py`, enabled via `EvaluationConfig(check_precision=True)`) is an auxiliary,
static diagnostic that measures how specific the value map `τ` is (how small the micro-space region each
abstract label maps to). It is not a validity score but a descriptive property of the mapping.

### Typical wiring

Build a `MicroVariableSchema` → `CoarseGrainingMap` (`a`, defines `Φ`) → `ValueMap` (`τ_X`) → a
low-level model (`LowLevelModel`, `M`) and a high-level model (`CausalGraph`, `E`) → construct an
`EvaluationEngine` from them and an `EvaluationConfig` → run `StandardTasks` (CAE↑/CAE↓) and/or the
analytical metrics through the engine. See any `systems/NN_*.py` paired with its `test/NN_*.py` suite
for a complete worked example.

### License

This project is licensed under the MIT License.

Some benchmark systems build on, link against, or redistribute third-party code
and data under their own licenses: notably, the MOS 6502 system (the `fake6502`,
`break6502`/M6502Core, and `perfect6502` emulator cores, plus the generated
`Decoder6502.bin`) and the gene-regulatory-network system (a GINsim
segment-polarity model). See [`THIRD_PARTY`](THIRD_PARTY) for the per-component
upstream URLs, pinned versions, and licenses.
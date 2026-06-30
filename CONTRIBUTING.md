# Contributing

Thanks for your interest in the `causal_abstraction_eval` framework (import name
`causal_abstraction`). This repository accompanies the paper *"Validating Causal
Abstraction Metrics on Simulated Complex Systems"*.

Contributions of every kind are welcome: bug fixes, new benchmark systems, new or
revised metrics, documentation, and changes to terminology or notation. There are no
off-limits areas; the only expectation is that you explain the rationale for a
change in your pull request so reviewers can evaluate it (see
[Pull requests](#pull-requests)).

## Development setup

```bash
git clone https://github.com/MelouxM/CAE.git
cd CAE
pip install -e ".[dev]"
```

Optional extras pull in per-system dependencies: `neural` (PyTorch), `physics`
(numba), `circuit` (PySpice), `figures` (matplotlib), and `all`. The Tracr system
needs `tracr` installed separately
(`pip install git+https://github.com/google-deepmind/tracr.git`).

## Common tasks

A thin `Makefile` wraps the common commands:

| Command | What it does |
|---|---|
| `make test` | Run the pytest unit suite (`test/test_*.py`). |
| `make coverage` | Run the unit suite with coverage (`pytest --cov=causal_abstraction`). |
| `make lint` | Run `ruff check`. |
| `make format` | Run `ruff format`. |
| `make suite SYS=01_logic_circuit RUNS=10` | Run one evaluation suite for a bounded number of runs. |

Equivalently, by hand:

```bash
pytest                                       # unit tests
pytest --cov=causal_abstraction              # with coverage
ruff check .                                 # lint
python test/01_logic_circuit.py --runs 10    # one evaluation suite, bounded
```

## Tests and runnable suites

`test/` holds two different things:

- **Unit tests**: `test/test_*.py` (+ `conftest.py`), collected by `pytest`. Many of
  them are **characterization tests** that pin the *current* output of a metric, path,
  or baseline. Their job is to catch behavior that moves unexpectedly: if you are only
  refactoring, a failing pin tells you something changed that shouldn't have. If you
  *intentionally* change a metric's behavior, update its pinned values in the same PR
  and explain the change.
- **Evaluation / power suites**: `test/NN_*.py`, `test/power_*.py`, built on
  `runner.py` / `power_runner.py` / `utils.py`. They run the full metric battery and
  write the paper's reported numbers to `test/results/*.json`; they are *not* collected
  by `pytest`. The numeric-prefixed system modules (`01_logic_circuit.py`, …) are not
  importable through normal package mechanics, so the suites load them by file path via
  `utils.load_system`.

The result files under `test/results/` are generated locally and are not committed to the
repo. If a change affects them, regenerate the affected suites (e.g.
`python test/01_logic_circuit.py`) and describe what changed and why in the PR; those
numbers are cited in the paper, so reviewers need to weigh the impact.

## Adding or changing a benchmark system

A benchmark system spans three places. A complete PR touches all of them so they stay
in sync:

- `systems/NN_*.py`: the system definition (the `(M, E, τ)` triple and its invalid
  contrastive conditions);
- `test/NN_*.py`: the matching evaluation suite;
- `paper/`: the systems table (`sections/04_benchmark.tex`) and the appendix entry.

## Pull requests

- Keep changes focused and explain why, referencing the relevant paper section where
  applicable.
- Confirm `make test` and `make lint` pass.
- If your change affects metric implementations, benchmark systems, notation, or any
  reported benchmark number, call it out explicitly and justify it: regenerate
  the affected results and update the paper-facing artifacts (tables, definitions) so
  the code, the numbers, and the paper stay consistent.

By contributing, you agree your contributions are licensed under the project's
MIT License.

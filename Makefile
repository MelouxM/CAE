# Thin task runner for common development commands. See CONTRIBUTING.md.
.PHONY: help test coverage lint format suite

# Override on the command line, e.g. `make suite SYS=01_logic_circuit RUNS=10`.
SYS  ?= 01_logic_circuit
RUNS ?= 10

help:
	@echo "Targets:"
	@echo "  test      Run the pytest unit suite (test/test_*.py)."
	@echo "  coverage  Run the unit suite with coverage over causal_abstraction."
	@echo "  lint      Run ruff check."
	@echo "  format    Run ruff format."
	@echo "  suite     Run one evaluation suite, bounded: make suite SYS=01_logic_circuit RUNS=10"

test:
	pytest

coverage:
	pytest --cov=causal_abstraction --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

suite:
	python test/$(SYS).py --runs $(RUNS)

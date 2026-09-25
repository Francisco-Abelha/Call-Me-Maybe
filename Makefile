# Mandatory rules per subject IV.2: install, run, debug, clean, lint.
# `uv run` resolves the environment from pyproject.toml/uv.lock on every call,
# so no manual venv activation is ever needed.

MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	          --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict help

help:
	@echo "install      Sync the virtual environment from pyproject.toml"
	@echo "run          Run the project"
	@echo "debug        Run the project under pdb"
	@echo "clean        Remove Python and tool caches"
	@echo "lint         flake8 + mypy"
	@echo "lint-strict  flake8 + mypy --strict"

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name '*.pyc' -delete

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

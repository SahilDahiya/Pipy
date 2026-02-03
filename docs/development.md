# Development Workflow

read_when: you need setup, test, or local run instructions

## Runtime

This project uses Python 3.12+ with `uv` for environment management.

## Commands

- Install deps: `uv sync` (creates/uses `.venv`).
- Install dev deps: `uv sync --extra dev`.
- Install test deps: `uv sync --extra test`.
- Run tests: `uv run pytest`.
- Lint and format: `uv run ruff check .` / `uv run ruff format .`.
- Run RPC: `uv run pi-rpc` (or `uv run python -m pi_sdk.rpc`).

If tests are missing for new behavior, add them before expanding APIs.

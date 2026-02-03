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

## API test env

- Tests load `.env` automatically (via `python-dotenv`).
- Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` for integration tests.
- Override models with `PI_OPENAI_TEST_MODEL` / `PI_ANTHROPIC_TEST_MODEL`.
- Default integration models: `gpt-4.1-nano` (OpenAI) and `claude-3-haiku-20240307` (Anthropic).
- Thinking-stream test is opt-in: set `PI_ANTHROPIC_REQUIRE_THINKING=1` and
  `PI_ANTHROPIC_THINKING_MODEL` (default `claude-3-7-sonnet-20250219`).

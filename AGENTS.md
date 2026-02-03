# Repository Guidelines

## Project Structure & Module Organization
- `pi_ai/` holds provider integrations, streaming, auth, models, and validation.
- `pi_agent/` contains the agent loop, events, and core state.
- `pi_tools/` implements tool primitives (read/write/edit/shell, etc.).
- `pi_session/` manages JSONL persistence and session trees.
- `pi_sdk/` is the embedding SDK and RPC entry point (`pi_sdk/rpc.py`).
- `tests/` contains pytest coverage.
- `docs/` contains architecture and workflow docs. Start at `docs/README.md` and add any new docs to its index.
- `main.py` is a small runnable entry for quick smoke checks.

## Build, Test, and Development Commands
- Install dependencies: `uv sync` (creates/uses `.venv`).
- Install dev dependencies (lint/type check): `uv sync --extra dev`.
- Install test dependencies: `uv sync --extra test`.
- Run lint/type check: `uv run ruff check .`.
- Auto-format: `uv run ruff format .`.
- Run tests: `uv run pytest`.
- Run the local entry: `uv run python main.py`.
- Run RPC entry point: `uv run pi-rpc`.

## Coding Style & Naming Conventions
- Python 3.12. Match existing formatting (4-space indentation).
- Use `snake_case` for variables/functions, `PascalCase` for classes/types, and `UPPER_SNAKE_CASE` for constants.
- Use Ruff for linting and type-check style checks.
- Keep provider logic in `pi_ai/`, core agent logic in `pi_agent/`, session handling in `pi_session/`, and tool logic in `pi_tools/`.
- Avoid side effects at import time; keep I/O in entry points or explicit functions.

## Testing Guidelines
- Tests live in `tests/` and use `test_*.py` naming.
- Use `pytest` and `pytest-asyncio` (configured in `pyproject.toml`).
- For bug reports, start by writing a reproducing test.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative messages (e.g., "Add ...", "Fix ..."). Follow that style.
- PRs should include a concise summary, relevant run instructions, and doc updates when behavior changes.

## Agent-Specific Workflow Notes
- Start with `docs/README.md` to find the right doc quickly.
- Keep changes small and reviewable.
- Prefer additive changes over refactors unless explicitly requested.
- If behavior changes, update relevant docs and the index in `docs/README.md`.
- Use ASCII by default; add Unicode only when already used.
- Do not ask the user for non-consequential actions; proceed with best judgment and note it in the response.

## Debugging
- When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug.
- Then, have subagents try to fix the bug and prove it with a passing test.
- If verification requires running the app outside the sandbox (e.g., network access), ask for approval and proceed once granted.

## Workflow
- Plan work in small steps.
- Run tests when available; document if tests are missing.
- Avoid destructive git operations unless explicitly asked.

## Docs Discipline
- Every new doc must be linked in `docs/README.md`.
- Include a `read_when` line at the top of each doc.

## Git
- Safe by default: `git status`/`git diff`/`git log`. Push only when the user asks.
- `git checkout` is ok for PR review or explicit request.
- Branch changes require user consent.
- Destructive ops forbidden unless explicit (`reset --hard`, `clean`, `restore`, `rm`, ...).
- Remotes under `~/repos`: prefer HTTPS; flip SSH->HTTPS before pull/push.
- Don't delete/rename unexpected stuff; stop and ask.
- No repo-wide S/R scripts; keep edits small and reviewable.
- Avoid manual `git stash`; if Git auto-stashes during pull/rebase, that's fine (hint, not hard guardrail).
- If the user types a command ("pull and push"), that's consent for that command.
- No amend unless asked.
- Big review: `git --no-pager diff --color=never`.

## Critical Thinking
- Fix root cause (not band-aid).
- Unsure: read more code; if still stuck, ask with short options.
- Conflicts: call out; pick safer path.
- Unrecognized changes: assume other agent; keep going; focus your changes. If it causes issues, stop and ask the user.
- Multiple agents: prefer additive changes, minimize refactors, and ask before reworking files with unrelated edits.
- Leave breadcrumb notes in thread.
- If multiple options are equally important, do them all in a reasonable sequence instead of asking me to choose.
- For any performance improvement, include a brief tradeoff analysis before implementation and summarize it in the final response.
- For performance-related changes, capture before/after measurements when feasible.

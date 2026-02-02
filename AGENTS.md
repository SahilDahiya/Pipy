# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the TypeScript runtime. Key areas: `src/cli/` (CLI entry at `src/cli/index.ts`), `src/runtime/`, `src/tools/`, `src/adapters/`, `src/gateway/`, `src/pi/`, and `src/contracts/`.
- `docs/` contains architecture and workflow docs. Start at `docs/README.md` and add any new docs to its index.
- `specs/` stores product and system specs. `prompts/` and `skills/` hold templates and skill content.
- `dist/` is for build outputs. `cmd/` and `internal/` are reserved for future entry points and core packages.

## Build, Test, and Development Commands
- Install dependencies: `bun install`
- Run in watch mode: `bun run dev`
- Run once: `bun run start`
- Build the CLI binary: `bun run build` (outputs `dist/wbot`)
- Install locally: `bun run install:local`
- Typecheck: `bun run typecheck`
- Tests are not configured yet; add a runner and a script before enforcing test expectations.

## Coding Style & Naming Conventions
- TypeScript with ES modules. Use 2-space indentation and semicolons (match existing files).
- Use `camelCase` for variables/functions, `PascalCase` for types/classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer `kebab-case` file names (see `src/cli/key-store.ts`).
- Keep side-effecting CLI code in `src/cli/` and reusable logic under `src/runtime/` or `src/tools/`.

## Testing Guidelines
- No framework is present yet. If you add tests, create a top-level `tests/` directory and name files `*.test.ts`.
- Document the chosen runner (for example, Bun test or Vitest) in `package.json` scripts.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative messages (e.g., "Add ...", "Fix ..."). Follow that style.
- PRs should include a concise summary, relevant run instructions, and doc updates when behavior changes.

## Agent-Specific Workflow Notes
- Start with `docs/README.md` to find the right doc quickly.
- Keep changes small and reviewable.
- Prefer additive changes over refactors unless explicitly requested.
- If behavior changes, update relevant docs and the index in `docs/README.md`.
- Use ASCII by default; add Unicode only when already used.
- For bug reports, start by writing a reproducing test when a test runner exists.
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

## Python Setup Hints (Optional)
- If you build a Python companion, mirror the CLI entry with a `pyproject.toml`, a `src/wbot/` package, and a `__main__.py` entry point.
- Use a local virtual environment (e.g., `.venv`) and document install/run steps alongside the Bun workflow.

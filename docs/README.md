# pi-python docs

read_when: start here, onboarding, or you need to find the right doc fast

## Quick map by task

- Understand system boundaries and responsibilities: `docs/architecture.md`
- Track porting scope and phases: `docs/porting-plan.md`
- Confirm project goal and scope: `docs/goal.md`
- Follow the agent loop and event flow: `docs/agent-loop.md`
- Provider behavior and env vars: `docs/providers.md`
- Default tools and contracts: `docs/tools.md`
- Session persistence and branching: `docs/sessions.md`
- SDK embedding surface: `docs/sdk.md`
- OAuth and auth storage: `docs/auth.md`
- RPC integration: `docs/rpc.md`
- Development workflow: `docs/development.md`

## Document index

- `docs/architecture.md` - Core layers, data flow, and invariants.
- `docs/porting-plan.md` - Source mapping to pi-mono and phase plan.
- `docs/goal.md` - Goal statement, in-scope packages, and out-of-scope items.
- `docs/agent-loop.md` - Turn lifecycle, steering/follow-up, and event hooks.
- `docs/providers.md` - OpenAI/Anthropic settings and compatibility notes.
- `docs/tools.md` - Default tool behavior and extension points.
- `docs/sessions.md` - JSONL format, session trees, and branching.
- `docs/sdk.md` - Embedding entry points and usage patterns.
- `docs/auth.md` - OAuth flows and auth.json storage.
- `docs/rpc.md` - JSON-over-stdin/stdout protocol summary.
- `docs/development.md` - Setup, tests, and local run commands.

## Keeping docs fresh

- Update docs when public APIs, events, or tool behavior changes.
- Add new docs to both the "Quick map by task" and "Document index".

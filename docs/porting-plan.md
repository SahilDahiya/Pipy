# Porting Plan

read_when: you are planning the Python port or sequencing implementation work

## Scope summary

In scope:
- `pi_ai` (LLM abstraction and providers)
- `pi_agent` (agent loop and events)
- `pi_tools` (read/write/edit/bash)
- `pi_session` (JSONL persistence and session trees)
- `pi_sdk` (embedding surface and RPC bridge)
- OAuth for Anthropic and OpenAI subscriptions

Out of scope:
- Any UI (CLI, TUI, web UI)
- Bots or channel integrations
- Prompt templates, skills, or extension systems

## Source mapping (pi-mono)

- `packages/ai` -> `pi_ai`
- `packages/agent` -> `pi_agent`
- `packages/coding-agent/src/tools` -> `pi_tools`
- `packages/coding-agent/src/core/session*` -> `pi_session`
- `packages/coding-agent/src/core/sdk.ts` -> `pi_sdk`
- `packages/coding-agent/src/core/auth-storage.ts` -> `pi_ai/auth`

## Phase 1: Working agent loop

1. Define core types and events in `pi_ai/types.py` and `pi_agent/events.py`.
2. Implement OpenAI Completions streaming provider (first).
3. Implement Anthropic Messages streaming provider.
4. Add cross-provider context serialization.
5. Implement the agent loop with tool execution and streaming events.
6. Ship the four tools with minimal validation.
7. Add tests against real APIs once providers are stable.

## Phase 2: Embedding and persistence

8. Implement JSONL session persistence and session trees.
9. Add `create_agent` in `pi_sdk/sdk.py`.
10. Implement RPC mode in `pi_sdk/rpc.py`.

## Phase 3: Subscription auth

11. Add OpenAI Codex OAuth (official flow first).
12. Add Anthropic Claude Pro/Max OAuth.
13. Persist and refresh tokens via `pi_ai/auth/storage.py`.

## Primary references to read first

1. `packages/ai/README.md`
2. `packages/ai/src/providers/openai-completions.ts`
3. `packages/ai/src/providers/anthropic.ts`
4. `packages/ai/src/agent/agent-loop.ts`
5. `packages/agent/src/agent.ts`
6. `packages/agent/src/types.ts`
7. `packages/coding-agent/src/core/sdk.ts`
8. `packages/coding-agent/src/tools/`
9. `packages/coding-agent/src/core/auth-storage.ts`

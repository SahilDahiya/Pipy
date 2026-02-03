# Serialization & Casing

read_when: you are touching RPC, session JSONL, or provider payload serialization

## Policy

- Internal Python APIs use snake_case.
- Wire formats follow pi-mono camelCase for compatibility (RPC and session JSONL).
- Value enums are normalized at the boundary (for example `tool_use` ↔ `toolUse`,
  `tool_call` ↔ `toolCall`, `tool_result` ↔ `toolResult`).

## Where conversions happen

- `pi_ai.utils.serialization` contains casing helpers and message/event converters.
- RPC outputs camelCase; RPC inputs accept camelCase or snake_case.
- Session JSONL is persisted in camelCase, then normalized back to snake_case when loaded.

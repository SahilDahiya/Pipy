# Project Goal

read_when: you need the authoritative scope and intent for pi-python

## Goal

Port the core of pi-mono into Python so developers can `pip install pi-python` and embed a Pi-style agent engine in any Python app. This is an engine (runtime + SDK), not an application or UI.

## Source of truth

- Continuously refer back to `/home/dahiy/repos/pi-mono` while porting.
- When in doubt, re-check the corresponding pi-mono module before implementing or adjusting behavior.

## In scope

- `pi_ai`: LLM abstraction with OpenAI Completions + Anthropic Messages
- `pi_agent`: Agent loop, tool execution, streaming events
- `pi_tools`: read/write/edit/bash tools
- `pi_session`: JSONL persistence + session trees
- `pi_sdk`: Embedding surface + RPC (stdin/stdout JSON)
- OAuth for OpenAI Codex subscriptions and Anthropic Pro/Max

## Out of scope

- Any UI (CLI, TUI, web)
- Bots or channel integrations (Slack, Telegram, etc.)
- Prompt templates, skills, extension systems, compaction
- Providers beyond OpenAI Completions and Anthropic Messages

## Provider order

1. OpenAI Completions API (covers OpenAI + compatible endpoints)
2. Anthropic Messages API

## Implementation phases

1. Working agent loop (types, OpenAI + Anthropic providers, tools)
2. Embedding + persistence (SDK + sessions)
3. Subscription OAuth (OpenAI Codex + Anthropic)

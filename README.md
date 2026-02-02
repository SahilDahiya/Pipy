# pi-python

pi-python is a Python port of the pi-mono agent engine. It provides the runtime and SDK for embedding Pi-style agents into any Python app (FastAPI, Slack, Telegram, etc). This repo builds the engine, not the UI.

## What this repo is for
- A minimal agent loop with tool execution and streaming events.
- A provider-agnostic LLM layer with OpenAI Completions and Anthropic Messages.
- Session persistence and portable session trees.
- A small SDK surface for embedding and an RPC bridge for non-Python clients.

## What is out of scope
- No CLI or TUI UI.
- No web UI or proxy.
- No prompt templates, skills, or extension system.

## Project layout (planned)
- `pi_ai/` - LLM providers, streaming, auth, and model registry.
- `pi_agent/` - Agent loop, events, and core state.
- `pi_tools/` - read/write/edit/bash tools.
- `pi_session/` - JSONL persistence and session trees.
- `pi_sdk/` - embedding SDK and RPC mode.
- `tests/` - coverage for providers, tools, and the loop.

## Docs
- `docs/README.md` - start here.
- `docs/architecture.md` - system boundaries and event flow.
- `docs/porting-plan.md` - phases, scope, and source mapping.

## Status
Early scaffold. See `docs/porting-plan.md` for sequencing.

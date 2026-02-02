# Architecture Overview

read_when: you need the big picture or want to understand system boundaries

## System overview

pi-python is a minimal agent engine with a small SDK surface. It exposes a streaming event loop, tool execution, and session persistence so consumers can build any transport or UI on top.

## Core layers

- `pi_ai` - Providers, auth, tool calling, token/cost tracking, and unified streaming.
- `pi_agent` - Agent loop, message queueing, and event emission.
- `pi_tools` - Four default tools: read, write, edit, bash.
- `pi_session` - JSONL persistence and session trees.
- `pi_sdk` - Embedding SDK and RPC mode for non-Python consumers.

## Data flow

1. App sends a user message to the agent.
2. Agent loop calls the provider stream and emits events.
3. Tool calls are executed and fed back into the loop.
4. Events are streamed to the consumer and persisted in the session.

## Boundaries and invariants

- The engine is policy-free; consumers decide UX and guardrails.
- Streaming is first-class and supports cancellation at every layer.
- Tools are side-effecting and observable; every call emits start/end events.
- Sessions are portable between providers via serialized context.

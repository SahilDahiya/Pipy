# Agent Loop

read_when: you need the runtime flow, event sequence, or steering/follow-up behavior

## Loop lifecycle

1. A user prompt (or injected steering message) starts a turn.
2. The agent streams an assistant response and emits message events.
3. If tool calls are present, tools run and tool result messages are added.
4. The loop repeats until no tool calls remain.

## Steering and follow-up

- **Steering messages** are checked before the next assistant response and after each tool execution. If present, remaining tool calls are skipped with an error tool result, and the steering message is injected immediately.
- **Follow-up messages** are checked after the agent would otherwise stop. If present, they start another turn.

## Context hooks

- `transform_context(messages, signal)` can prune, inject, or rewrite messages before conversion to LLM format.
- `convert_to_llm(messages)` maps `AgentMessage` entries into `Message` objects the provider can accept.
- `get_api_key(provider)` resolves short-lived tokens (for OAuth) before each LLM call.

## Events

- Turn lifecycle: `agent_start`, `turn_start`, `turn_end`, `agent_end`.
- Message lifecycle: `message_start`, `message_update`, `message_end`.
- Tool lifecycle: `tool_execution_start`, `tool_execution_update`, `tool_execution_end`.

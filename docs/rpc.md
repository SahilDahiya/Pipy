# RPC Mode

read_when: you are integrating via JSON over stdin/stdout

## Start RPC

```bash
python -m pi_sdk.rpc
# or
pi-rpc
```

## Commands

- Commands accept pi-mono camelCase fields (snake_case also accepted).

- `{"type": "prompt", "message": "..."}` (alias: `send`): send a user message.
- `{"type": "steer", "message": "..."}`: queue a steering message.
- `{"type": "follow_up", "message": "..."}`: queue a follow-up message.
- `{"type": "abort"}`: cancel the current run.
- `{"type": "reset"}` / `{"type": "new_session"}`: clear messages and queues.
- `{"type": "get_state"}`: return the current agent state.
- `{"type": "set_model", "provider": "...", "modelId": "..."}`: switch models.
- `{"type": "set_thinking_level", "level": "low"}`: adjust reasoning level.
- `{"type": "set_steering_mode", "mode": "all"}` / `set_follow_up_mode`.
- `{"type": "get_messages"}`: return the current message list.

Each command returns a `response` object. `prompt`/`send` also stream agent events
on stdout using pi-mono-compatible camelCase fields (for example `contentIndex`,
`toolCall`, `stopReason`).

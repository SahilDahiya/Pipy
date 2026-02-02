# RPC Mode

read_when: you are integrating via JSON over stdin/stdout

## Start RPC

```bash
python -m pi_sdk.rpc
# or
pi-rpc
```

## Commands

- `{"type": "send", "text": "..."}`: send a user message.
- `{"type": "reset"}`: clear the current message history.

Each command yields JSON events on stdout, matching the agent event stream.

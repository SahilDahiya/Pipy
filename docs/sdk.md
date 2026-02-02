# SDK Usage

read_when: you are embedding pi-python or wiring it to a transport

## Create an agent

```python
from pi_ai.models import get_model
from pi_sdk import create_agent

agent = create_agent(
    model=get_model("anthropic", "claude-sonnet-4-5"),
    system_prompt="You are a helpful assistant.",
)
```

## Stream events

```python
async for event in agent.send("Summarize README.md"):
    if event["type"] == "text_delta":
        print(event["delta"], end="")
```

## Sessions and tools

- Pass `session_path` to resume a JSONL session.
- When a session includes model or thinking-level entries, they are restored unless explicitly overridden.
- Override tools via `tools=[...]` or use `create_default_tools(cwd)`.
- Provide `auth_path` or `auth_storage` to reuse OAuth tokens.

## RPC mode

Use `python -m pi_sdk.rpc` (or `pi-rpc`) for JSON-over-stdin/stdout integration.

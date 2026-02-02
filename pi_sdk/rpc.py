"""JSON-over-stdin/stdout RPC bridge for pi-python."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

from pi_agent.agent import Agent
from pi_ai.models import get_model
from pi_sdk.sdk import create_agent


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[call-arg]
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


async def _handle_send(agent: Agent, payload: Dict[str, Any]) -> None:
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("send requires a 'text' string")

    stream = agent.send(text)
    async for event in stream:
        sys.stdout.write(json.dumps(_to_jsonable(event)) + "\n")
        sys.stdout.flush()


async def _read_lines() -> None:
    provider = os.getenv("PI_PROVIDER", "openai")
    model_id = os.getenv("PI_MODEL", "gpt-4o-mini")
    agent = create_agent(provider=provider, model_id=model_id)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        msg_type = data.get("type")

        if msg_type == "send":
            await _handle_send(agent, data)
        elif msg_type == "reset":
            agent.clear_messages()
            sys.stdout.write(json.dumps({"type": "reset_ok"}) + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write(json.dumps({"type": "error", "message": "Unknown message type"}) + "\n")
            sys.stdout.flush()


def main() -> None:
    asyncio.run(_read_lines())


if __name__ == "__main__":
    main()

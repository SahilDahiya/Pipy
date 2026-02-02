import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from pi_sdk import rpc
from pi_ai.streaming import AssistantMessageEventStream
from pi_ai.types import TextContent
from tests.helpers import create_assistant_message


@pytest.mark.asyncio
async def test_rpc_to_jsonable_handles_models():
    message = create_assistant_message([TextContent(text="ok")])
    payload = rpc._to_jsonable(message)
    assert payload["role"] == "assistant"
    assert payload["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_rpc_handle_prompt_emits_response_and_events(monkeypatch):
    stream = AssistantMessageEventStream()

    async def run_stream():
        await asyncio.sleep(0)
        msg = create_assistant_message([TextContent(text="ok")])
        stream.push({"type": "done", "reason": "stop", "message": msg})
        stream.end(msg)

    asyncio.create_task(run_stream())

    class DummyAgent:
        def __init__(self):
            self.state = SimpleNamespace(
                is_streaming=False,
                messages=[],
                pending_tool_calls=set(),
                error=None,
                thinking_level="off",
                model=None,
            )

        def send(self, _text, images=None):
            return stream

    buffer = io.StringIO()
    monkeypatch.setattr(rpc.sys, "stdout", buffer)

    await rpc._handle_prompt(DummyAgent(), {"message": "hello"})
    output = buffer.getvalue().strip().splitlines()
    assert output
    response = json.loads(output[0])
    assert response["type"] == "response"
    assert response["command"] == "prompt"
    event = json.loads(output[-1])
    assert event["type"] in {"done", "error"}

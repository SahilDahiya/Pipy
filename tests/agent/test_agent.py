import asyncio

import pytest

from pi_agent.agent import Agent
from pi_ai.models import create_openai_model
from pi_ai.streaming import AssistantMessageEventStream
from pi_ai.types import TextContent

from tests.helpers import create_assistant_message, create_user_message


def _model():
    return create_openai_model("gpt-4o-mini", provider="openai")


@pytest.mark.asyncio
async def test_agent_state_and_mutators():
    agent = Agent(model=_model())
    assert agent.state.system_prompt == ""
    assert agent.state.messages == []
    assert agent.state.is_streaming is False

    agent.set_system_prompt("Test prompt")
    assert agent.state.system_prompt == "Test prompt"

    new_model = create_openai_model("gpt-4o", provider="openai")
    agent.set_model(new_model)
    assert agent.state.model == new_model

    agent.set_thinking_level("high")
    assert agent.state.thinking_level == "high"

    agent.append_message(create_user_message("Hello"))
    assert len(agent.state.messages) == 1
    agent.clear_messages()
    assert agent.state.messages == []


@pytest.mark.asyncio
async def test_agent_steering_and_follow_up_queues():
    agent = Agent(model=_model())
    steering = create_user_message("Steer")
    follow_up = create_user_message("Follow")
    agent.steer(steering)
    agent.follow_up(follow_up)

    assert steering not in agent.state.messages
    assert follow_up not in agent.state.messages

    agent.clear_all_queues()
    agent.reset()
    assert agent.state.messages == []


@pytest.mark.asyncio
async def test_agent_prevents_parallel_send():
    def stream_fn(_model, _context, options):
        stream = AssistantMessageEventStream()

        async def run():
            stream.push({"type": "start", "partial": create_assistant_message([TextContent(text="")])})
            while not options.signal.is_set():
                await asyncio.sleep(0.01)
            msg = create_assistant_message([TextContent(text="aborted")], stop_reason="aborted")
            stream.push({"type": "error", "reason": "aborted", "error": msg})
            stream.end(msg)

        asyncio.create_task(run())
        return stream

    agent = Agent(model=_model(), stream_fn=stream_fn)
    first = agent.send("first")

    await asyncio.sleep(0.05)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError, match="Agent is already processing"):
        agent.send("second")

    agent.abort()
    await first.result()


@pytest.mark.asyncio
async def test_session_id_forwarded():
    received = {}

    def stream_fn(_model, _context, options):
        received["session_id"] = options.session_id
        stream = AssistantMessageEventStream()

        async def run():
            await asyncio.sleep(0)
            msg = create_assistant_message([TextContent(text="ok")])
            stream.push({"type": "done", "reason": "stop", "message": msg})
            stream.end(msg)

        asyncio.create_task(run())
        return stream

    agent = Agent(model=_model(), session_id="session-abc", stream_fn=stream_fn)
    stream = agent.send("hello")
    async for _ in stream:
        pass
    assert received["session_id"] == "session-abc"

    agent.session_id = "session-def"
    stream = agent.send("hello again")
    async for _ in stream:
        pass
    assert received["session_id"] == "session-def"


@pytest.mark.asyncio
async def test_max_retry_delay_forwarded():
    received = {}

    def stream_fn(_model, _context, options):
        received["max_retry_delay_ms"] = options.max_retry_delay_ms
        stream = AssistantMessageEventStream()

        async def run():
            await asyncio.sleep(0)
            msg = create_assistant_message([TextContent(text="ok")])
            stream.push({"type": "done", "reason": "stop", "message": msg})
            stream.end(msg)

        asyncio.create_task(run())
        return stream

    agent = Agent(model=_model(), max_retry_delay_ms=1234, stream_fn=stream_fn)
    stream = agent.send("hello")
    async for _ in stream:
        pass
    assert received["max_retry_delay_ms"] == 1234


def test_continue_session_requires_valid_last_message():
    agent = Agent(model=_model())
    with pytest.raises(RuntimeError, match="No messages to continue"):
        agent.continue_session()

    agent.append_message(create_assistant_message([TextContent(text="hi")]))
    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        agent.continue_session()

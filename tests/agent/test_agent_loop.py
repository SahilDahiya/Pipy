import asyncio

import pytest

from pi_agent.loop import agent_loop, agent_loop_continue
from pi_agent.types import AgentContext, AgentLoopConfig, AgentMessage
from pi_ai.streaming import AssistantMessageEventStream
from pi_ai.types import Message, TextContent, ToolCall
from pi_tools.base import ToolDefinition, ToolResult

from tests.helpers import create_assistant_message, create_model, create_user_message


def _mock_stream(message):
    stream = AssistantMessageEventStream()

    async def _run():
        await asyncio.sleep(0)
        stream.push({"type": "done", "reason": message.stop_reason, "message": message})
        stream.end(message)

    asyncio.create_task(_run())
    return stream


def _identity_converter(messages: list[AgentMessage]) -> list[Message]:
    return [m for m in messages if m.role in {"user", "assistant", "toolResult"}]


@pytest.mark.asyncio
async def test_agent_loop_emits_events():
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
    user_prompt: AgentMessage = create_user_message("Hello")
    config = AgentLoopConfig(model=create_model(), convert_to_llm=_identity_converter)

    events = []
    stream = agent_loop(
        [user_prompt],
        context,
        config,
        stream_fn=lambda *_args, **_kwargs: _mock_stream(
            create_assistant_message([TextContent(text="Hi")])
        ),
    )

    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = {event["type"] for event in events}
    assert "agent_start" in event_types
    assert "turn_start" in event_types
    assert "message_start" in event_types
    assert "message_end" in event_types
    assert "turn_end" in event_types
    assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_custom_messages_filtered():
    class CustomNotification:
        def __init__(self, text: str) -> None:
            self.role = "notification"
            self.text = text

    notification = CustomNotification("note")
    context = AgentContext(system_prompt="", messages=[notification], tools=[])
    user_prompt: AgentMessage = create_user_message("Hello")

    converted: list[Message] = []

    def convert(messages: list[AgentMessage]) -> list[Message]:
        nonlocal converted
        converted = [
            m for m in messages if getattr(m, "role", None) in {"user", "assistant", "toolResult"}
        ]
        return converted

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert)
    stream = agent_loop(
        [user_prompt],
        context,
        config,
        stream_fn=lambda *_args, **_kwargs: _mock_stream(
            create_assistant_message([TextContent(text="ok")])
        ),
    )

    async for _ in stream:
        pass

    assert len(converted) == 1
    assert converted[0].role == "user"


@pytest.mark.asyncio
async def test_transform_context_before_convert():
    context = AgentContext(
        system_prompt="",
        messages=[
            create_user_message("old1"),
            create_assistant_message([TextContent(text="r1")]),
            create_user_message("old2"),
            create_assistant_message([TextContent(text="r2")]),
        ],
        tools=[],
    )

    transformed: list[AgentMessage] = []
    converted: list[Message] = []

    async def transform(messages, _signal=None):
        nonlocal transformed
        transformed = messages[-2:]
        return transformed

    def convert(messages):
        nonlocal converted
        converted = [m for m in messages if m.role in {"user", "assistant", "toolResult"}]
        return converted

    config = AgentLoopConfig(
        model=create_model(),
        transform_context=transform,
        convert_to_llm=convert,
    )

    stream = agent_loop(
        [create_user_message("new")],
        context,
        config,
        stream_fn=lambda *_args, **_kwargs: _mock_stream(
            create_assistant_message([TextContent(text="ok")])
        ),
    )

    async for _ in stream:
        pass

    assert len(transformed) == 2
    assert len(converted) == 2


@pytest.mark.asyncio
async def test_tool_calls_execute_and_continue():
    executed = []

    async def execute(_tool_call_id, params, *_args, **_kwargs):
        executed.append(params["value"])
        return ToolResult(content=[TextContent(text=f"ok:{params['value']}")], details={"value": params["value"]})

    tool = ToolDefinition(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=_identity_converter)

    call_index = 0

    def stream_fn(_model, _ctx, _options):
        nonlocal call_index
        stream = AssistantMessageEventStream()

        async def run():
            nonlocal call_index
            await asyncio.sleep(0)
            if call_index == 0:
                msg = create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})],
                    stop_reason="toolUse",
                )
                stream.push({"type": "done", "reason": "toolUse", "message": msg})
                stream.end(msg)
            else:
                msg = create_assistant_message([TextContent(text="done")])
                stream.push({"type": "done", "reason": "stop", "message": msg})
                stream.end(msg)
            call_index += 1

        asyncio.create_task(run())
        return stream

    stream = agent_loop([create_user_message("go")], context, config, stream_fn=stream_fn)
    async for _ in stream:
        pass

    assert executed == ["hello"]


@pytest.mark.asyncio
async def test_steering_skips_remaining_tool_calls():
    executed = []

    async def execute(_tool_call_id, params, *_args, **_kwargs):
        executed.append(params["value"])
        return ToolResult(content=[TextContent(text=f"ok:{params['value']}")], details={"value": params["value"]})

    tool = ToolDefinition(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    queued = create_user_message("interrupt")
    queued_delivered = False
    call_index = 0
    saw_interrupt = False

    async def get_steering():
        nonlocal queued_delivered
        if executed and not queued_delivered:
            queued_delivered = True
            return [queued]
        return []

    config = AgentLoopConfig(model=create_model(), convert_to_llm=_identity_converter, get_steering_messages=get_steering)

    def stream_fn(_model, ctx, _options):
        nonlocal call_index, saw_interrupt
        if call_index == 1:
            saw_interrupt = any(
                m.role == "user" and isinstance(m.content, str) and m.content == "interrupt"
                for m in ctx.messages
            )

        stream = AssistantMessageEventStream()

        async def run():
            nonlocal call_index
            await asyncio.sleep(0)
            if call_index == 0:
                msg = create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    stop_reason="toolUse",
                )
                stream.push({"type": "done", "reason": "toolUse", "message": msg})
                stream.end(msg)
            else:
                msg = create_assistant_message([TextContent(text="done")])
                stream.push({"type": "done", "reason": "stop", "message": msg})
                stream.end(msg)
            call_index += 1

        asyncio.create_task(run())
        return stream

    events = []
    stream = agent_loop([create_user_message("start")], context, config, stream_fn=stream_fn)
    async for event in stream:
        events.append(event)

    tool_ends = [e for e in events if e["type"] == "tool_execution_end"]
    assert executed == ["first"]
    assert len(tool_ends) == 2
    assert tool_ends[0]["isError"] is False
    assert tool_ends[1]["isError"] is True
    assert saw_interrupt is True


def test_agent_loop_continue_requires_messages():
    context = AgentContext(system_prompt="", messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=_identity_converter)
    with pytest.raises(ValueError, match="Cannot continue: no messages in context"):
        agent_loop_continue(context, config)


@pytest.mark.asyncio
async def test_agent_loop_continue_no_user_event():
    user_message = create_user_message("Hello")
    context = AgentContext(system_prompt="", messages=[user_message], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=_identity_converter)

    stream = agent_loop_continue(
        context,
        config,
        stream_fn=lambda *_args, **_kwargs: _mock_stream(
            create_assistant_message([TextContent(text="Response")])
        ),
    )

    events = []
    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert len(messages) == 1
    assert messages[0].role == "assistant"

    message_end = [e for e in events if e["type"] == "message_end"]
    assert len(message_end) == 1
    assert message_end[0]["message"].role == "assistant"

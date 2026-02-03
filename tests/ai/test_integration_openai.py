import os

import pytest

from pi_ai.stream import complete, stream
from pi_ai.models import create_openai_model
from pi_ai.providers.openai import OpenAICompletionsOptions
from pi_ai.types import Context, Tool, ToolCall, UserMessage


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("PI_OPENAI_TEST_MODEL", "gpt-5-nano")
OPENAI_THINKING_MODEL_ID = os.getenv("PI_OPENAI_THINKING_MODEL", "gpt-5-mini")

pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY is required for OpenAI integration tests."
)


def _calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Operation to perform",
                },
            },
            "required": ["a", "b", "operation"],
        },
    )


@pytest.mark.asyncio
async def test_openai_basic_text_generation():
    model = create_openai_model(OPENAI_MODEL_ID, provider="openai")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Reply with exactly: hello test")],
    )
    response = await complete(
        model,
        context,
        OpenAICompletionsOptions(api_key=OPENAI_API_KEY, temperature=0),
    )
    assert response.role == "assistant"
    assert response.content
    text = "".join(block.text for block in response.content if block.type == "text")
    assert "hello" in text.lower()


@pytest.mark.asyncio
async def test_openai_tool_call_streaming():
    model = create_openai_model(OPENAI_MODEL_ID, provider="openai")
    context = Context(
        system_prompt="You are a helpful assistant that must use the calculator tool.",
        messages=[
            UserMessage(content="Calculate 15 + 27 using the calculator tool."),
        ],
        tools=[_calculator_tool()],
    )

    options = OpenAICompletionsOptions(
        api_key=OPENAI_API_KEY,
        tool_choice="required",
        temperature=0,
    )

    stream_response = stream(model, context, options)
    saw_start = False
    saw_delta = False
    saw_end = False

    async for event in stream_response:
        if event["type"] == "toolcall_start":
            saw_start = True
        if event["type"] == "toolcall_delta":
            saw_delta = True
        if event["type"] == "toolcall_end":
            saw_end = True

    final_message = await stream_response.result()
    tool_calls = [block for block in final_message.content if isinstance(block, ToolCall)]

    assert saw_start
    assert saw_delta
    assert saw_end
    assert tool_calls
    assert tool_calls[0].name == "calculator"


@pytest.mark.asyncio
async def test_openai_reasoning_effort():
    model = create_openai_model(OPENAI_THINKING_MODEL_ID, provider="openai", reasoning=True)
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Explain 2 + 2 in one sentence.")],
    )
    response = await complete(
        model,
        context,
        OpenAICompletionsOptions(
            api_key=OPENAI_API_KEY,
            reasoning_effort="low",
            max_tokens=256,
        ),
    )
    assert response.role == "assistant"
    has_text = any(block.type == "text" and block.text.strip() for block in response.content)
    has_thinking = any(block.type == "thinking" and block.thinking.strip() for block in response.content)
    assert has_text or has_thinking

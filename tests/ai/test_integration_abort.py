import asyncio
import os

import pytest

from pi_ai.stream import complete, stream
from pi_ai.models import create_anthropic_model, create_openai_model
from pi_ai.providers.anthropic import AnthropicOptions
from pi_ai.providers.openai import OpenAICompletionsOptions
from pi_ai.types import Context, UserMessage


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


async def _abort_stream(model, context, options, abort_after_chars: int = 80):
    signal = asyncio.Event()
    options.signal = signal
    response = stream(model, context, options)

    text = ""
    start_time = asyncio.get_event_loop().time()

    async for event in response:
        if event["type"] in {"text_delta", "thinking_delta"}:
            text += event["delta"]
        if len(text) >= abort_after_chars:
            signal.set()
        if asyncio.get_event_loop().time() - start_time > 5:
            signal.set()

    return await response.result()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY is required for OpenAI abort integration tests."
)
async def test_openai_abort_and_continue():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                content="Write a short paragraph about the ocean, then list 10 fish species."
            )
        ],
    )

    abort_message = await _abort_stream(
        model,
        context,
        OpenAICompletionsOptions(api_key=OPENAI_API_KEY, temperature=0),
    )
    assert abort_message.stop_reason == "aborted"

    context.messages.append(abort_message)
    context.messages.append(UserMessage(content="Continue with only 2 fish species."))

    follow_up = await complete(
        model,
        context,
        OpenAICompletionsOptions(api_key=OPENAI_API_KEY, temperature=0),
    )
    assert follow_up.stop_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY is required for Anthropic abort tests."
)
async def test_anthropic_abort_and_continue():
    model = create_anthropic_model("claude-sonnet-4-5", provider="anthropic")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                content="Explain quantum tunneling in simple terms, then give three examples."
            )
        ],
    )

    abort_message = await _abort_stream(
        model,
        context,
        AnthropicOptions(api_key=ANTHROPIC_API_KEY, temperature=0),
    )
    assert abort_message.stop_reason == "aborted"

    context.messages.append(abort_message)
    context.messages.append(UserMessage(content="Continue with only one example."))

    follow_up = await complete(
        model,
        context,
        AnthropicOptions(api_key=ANTHROPIC_API_KEY, temperature=0),
    )
    assert follow_up.stop_reason == "stop"

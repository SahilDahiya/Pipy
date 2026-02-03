import asyncio
import os

import pytest

from pi_ai.stream import stream
from pi_ai.models import create_anthropic_model, create_openai_model
from pi_ai.providers.anthropic import AnthropicOptions
from pi_ai.providers.openai import OpenAICompletionsOptions
from pi_ai.types import Context, UserMessage


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_MODEL_ID = os.getenv("PI_OPENAI_TEST_MODEL", "gpt-5-nano")
ANTHROPIC_MODEL_ID = os.getenv("PI_ANTHROPIC_TEST_MODEL", "claude-3-haiku-20240307")


async def _abort_with_signal(model, context, options, abort_after_chars: int = 80):
    signal = asyncio.Event()
    options.signal = signal
    response = stream(model, context, options)
    text = ""

    async for event in response:
        if event["type"] in {"text_delta", "thinking_delta"}:
            text += event["delta"]
        if len(text) >= abort_after_chars:
            signal.set()

    return await response.result()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY is required for OpenAI token tests."
)
async def test_openai_abort_has_no_usage_tokens():
    model = create_openai_model(OPENAI_MODEL_ID, provider="openai")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Write a long list of 50 short words.")],
    )
    message = await _abort_with_signal(
        model,
        context,
        OpenAICompletionsOptions(api_key=OPENAI_API_KEY, temperature=0),
    )
    assert message.stop_reason == "aborted"
    assert message.usage.input == 0
    assert message.usage.output == 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY is required for Anthropic token tests."
)
async def test_anthropic_abort_reports_input_tokens():
    model = create_anthropic_model(ANTHROPIC_MODEL_ID, provider="anthropic")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Write a long list of 50 short words.")],
    )
    message = await _abort_with_signal(
        model,
        context,
        AnthropicOptions(api_key=ANTHROPIC_API_KEY, temperature=0),
    )
    assert message.stop_reason == "aborted"
    assert message.usage.input > 0

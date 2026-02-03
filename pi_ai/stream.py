"""Unified streaming helpers for provider responses."""

from __future__ import annotations

from typing import Optional

from .streaming import AssistantMessageEventStream
from .types import AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions
from .providers import anthropic as anthropic_provider
from .providers import openai as openai_provider


def stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    if model.api == "openai-completions":
        return openai_provider.stream_openai_completions(model, context, options)
    if model.api == "anthropic-messages":
        return anthropic_provider.stream_anthropic(model, context, options)
    raise NotImplementedError(f"Streaming not implemented for API: {model.api}")


async def complete(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessage:
    response = stream(model, context, options)
    return await response.result()


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    if model.api == "openai-completions":
        return openai_provider.stream_simple_openai_completions(model, context, options)
    if model.api == "anthropic-messages":
        return anthropic_provider.stream_simple_anthropic(model, context, options)
    raise NotImplementedError(f"Streaming not implemented for API: {model.api}")


async def complete_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessage:
    response = stream_simple(model, context, options)
    return await response.result()

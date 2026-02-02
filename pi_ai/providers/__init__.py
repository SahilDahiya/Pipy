"""Provider implementations and interfaces."""

from __future__ import annotations

from typing import Optional

from ..types import Context, Model, SimpleStreamOptions
from .anthropic import stream_simple_anthropic
from .openai import stream_simple_openai_completions

__all__ = [
    "anthropic",
    "base",
    "openai",
    "stream_simple",
]


def stream_simple(model: Model, context: Context, options: Optional[SimpleStreamOptions] = None):
    if model.api == "openai-completions":
        return stream_simple_openai_completions(model, context, options)
    if model.api == "anthropic-messages":
        return stream_simple_anthropic(model, context, options)
    raise NotImplementedError(f"Streaming not implemented for API: {model.api}")

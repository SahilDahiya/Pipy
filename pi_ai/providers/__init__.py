"""Provider implementations and interfaces."""

from __future__ import annotations

from typing import Optional

from ..types import Context, Model, SimpleStreamOptions
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
    raise NotImplementedError(f"Streaming not implemented for API: {model.api}")

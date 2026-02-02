"""Cross-provider context serialization."""

from __future__ import annotations

from typing import Any, Dict

from .types import Context


def serialize_context(context: Context) -> Dict[str, Any]:
    return context.model_dump()


def deserialize_context(payload: Dict[str, Any]) -> Context:
    return Context.model_validate(payload)

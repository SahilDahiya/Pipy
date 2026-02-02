"""Helpers for translating simple stream options into provider options."""

from __future__ import annotations

from typing import Dict, Optional

from ..types import Model, SimpleStreamOptions, StreamOptions

_DEFAULT_THINKING_BUDGETS: Dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


def build_base_options(
    model: Model,
    options: Optional[SimpleStreamOptions],
    api_key: Optional[str] = None,
) -> StreamOptions:
    max_tokens = options.max_tokens if options else None
    if max_tokens is None and model.max_tokens is not None:
        max_tokens = min(model.max_tokens, 32000)

    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=max_tokens,
        signal=options.signal if options else None,
        api_key=api_key or (options.api_key if options else None),
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
    )


def clamp_reasoning(effort: Optional[str]) -> Optional[str]:
    return "high" if effort == "xhigh" else effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: str,
    custom_budgets: Optional[Dict[str, int]] = None,
) -> tuple[int, int]:
    budgets = {**_DEFAULT_THINKING_BUDGETS, **(custom_budgets or {})}
    level = clamp_reasoning(reasoning_level)
    if not level:
        return base_max_tokens, 0

    thinking_budget = budgets.get(level, 0)
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    min_output_tokens = 1024
    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget

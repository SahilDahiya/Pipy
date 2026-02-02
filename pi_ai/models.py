"""Model registry and capability metadata."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Model, ModelCost, OpenAICompletionsCompat, Usage

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_MODEL_REGISTRY: Dict[Tuple[str, str], Model] = {}


def register_model(model: Model) -> None:
    _MODEL_REGISTRY[(model.provider, model.id)] = model


def create_openai_model(
    model_id: str,
    *,
    provider: str = "openai",
    base_url: str | None = None,
    reasoning: bool = False,
    input_modalities: List[str] | None = None,
    context_window: int | None = None,
    max_tokens: int | None = None,
    cost: ModelCost | None = None,
    headers: Dict[str, str] | None = None,
    compat: OpenAICompletionsCompat | None = None,
    supports_xhigh: bool = False,
) -> Model:
    return Model(
        id=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url or DEFAULT_OPENAI_BASE_URL,
        reasoning=reasoning,
        input=input_modalities or ["text"],
        context_window=context_window,
        max_tokens=max_tokens,
        cost=cost or ModelCost(),
        headers=headers or {},
        compat=compat,
        supports_xhigh=supports_xhigh,
    )


def get_model(provider: str, model_id: str) -> Model:
    key = (provider, model_id)
    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key]
    if provider == "openai":
        model = create_openai_model(model_id)
        register_model(model)
        return model
    raise KeyError(f"Model not found: {provider}/{model_id}. Register it first.")


def list_models(provider: str | None = None) -> List[Model]:
    if provider is None:
        return list(_MODEL_REGISTRY.values())
    return [model for (prov, _), model in _MODEL_REGISTRY.items() if prov == provider]


def supports_xhigh(model: Model) -> bool:
    return model.supports_xhigh


def calculate_cost(model: Model, usage: Usage) -> None:
    rates = model.cost
    if not rates:
        usage.cost.total = 0.0
        return

    usage.cost.input = usage.input * rates.input / 1_000_000
    usage.cost.output = usage.output * rates.output / 1_000_000
    usage.cost.cache_read = usage.cache_read * rates.cache_read / 1_000_000
    usage.cost.cache_write = usage.cache_write * rates.cache_write / 1_000_000
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )

"""Model registry and capability metadata."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Model, ModelCost, OpenAICompletionsCompat, Usage

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

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


def create_anthropic_model(
    model_id: str,
    *,
    provider: str = "anthropic",
    base_url: str | None = None,
    reasoning: bool = True,
    input_modalities: List[str] | None = None,
    context_window: int | None = None,
    max_tokens: int | None = None,
    cost: ModelCost | None = None,
    headers: Dict[str, str] | None = None,
) -> Model:
    return Model(
        id=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url=base_url or DEFAULT_ANTHROPIC_BASE_URL,
        reasoning=reasoning,
        input=input_modalities or ["text"],
        context_window=context_window,
        max_tokens=max_tokens,
        cost=cost or ModelCost(),
        headers=headers or {},
        compat=None,
        supports_xhigh=False,
    )


def get_model(provider: str, model_id: str) -> Model:
    key = (provider, model_id)
    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key]
    if provider == "openai":
        model = create_openai_model(model_id)
        register_model(model)
        return model
    if provider == "anthropic":
        model = create_anthropic_model(model_id)
        register_model(model)
        return model
    raise KeyError(f"Model not found: {provider}/{model_id}. Register it first.")


def list_models(provider: str | None = None) -> List[Model]:
    if provider is None:
        return list(_MODEL_REGISTRY.values())
    return [model for (prov, _), model in _MODEL_REGISTRY.items() if prov == provider]


def supports_xhigh(model: Model) -> bool:
    xhigh_models = {"gpt-5.1-codex-max", "gpt-5.2", "gpt-5.2-codex"}
    return model.supports_xhigh or model.id in xhigh_models


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


def _register_if_missing(model: Model) -> None:
    key = (model.provider, model.id)
    if key not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[key] = model


def _register_builtin_models() -> None:
    # OpenAI (completions) defaults
    _register_if_missing(
        create_openai_model(
            "gpt-4o-mini",
            provider="openai",
            reasoning=False,
            input_modalities=["text", "image"],
            context_window=128000,
            max_tokens=16384,
            cost=ModelCost(input=0.15, output=0.6, cache_read=0.08, cache_write=0),
        )
    )
    _register_if_missing(
        create_openai_model(
            "gpt-4o",
            provider="openai",
            reasoning=False,
            input_modalities=["text", "image"],
            context_window=128000,
            max_tokens=16384,
            cost=ModelCost(input=2.5, output=10, cache_read=1.25, cache_write=0),
        )
    )

    # Anthropic defaults
    _register_if_missing(
        create_anthropic_model(
            "claude-sonnet-4-5",
            provider="anthropic",
            reasoning=True,
            input_modalities=["text", "image"],
            context_window=200000,
            max_tokens=64000,
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        )
    )
    _register_if_missing(
        create_anthropic_model(
            "claude-3-5-sonnet-20241022",
            provider="anthropic",
            reasoning=False,
            input_modalities=["text", "image"],
            context_window=200000,
            max_tokens=8192,
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        )
    )
    _register_if_missing(
        create_anthropic_model(
            "claude-3-5-haiku-20241022",
            provider="anthropic",
            reasoning=False,
            input_modalities=["text", "image"],
            context_window=200000,
            max_tokens=8192,
            cost=ModelCost(input=0.8, output=4, cache_read=0.08, cache_write=1),
        )
    )


_register_builtin_models()

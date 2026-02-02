"""Model registry and capability metadata."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Model, Usage

_MODEL_REGISTRY: Dict[Tuple[str, str], Model] = {}


def register_model(model: Model) -> None:
    _MODEL_REGISTRY[(model.provider, model.id)] = model


def get_model(provider: str, model_id: str) -> Model:
    key = (provider, model_id)
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Model not found: {provider}/{model_id}. Register it first.")
    return _MODEL_REGISTRY[key]


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

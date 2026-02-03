import os

from pi_ai.providers import anthropic as anthropic_provider


def _set_env(value: str | None) -> None:
    if value is None:
        os.environ.pop("PI_CACHE_RETENTION", None)
    else:
        os.environ["PI_CACHE_RETENTION"] = value


def test_cache_retention_defaults_short():
    original = os.environ.get("PI_CACHE_RETENTION")
    _set_env(None)
    try:
        retention, cache_control = anthropic_provider._get_cache_control(
            "https://api.anthropic.com/v1", None
        )
    finally:
        _set_env(original)

    assert retention == "short"
    assert cache_control == {"type": "ephemeral"}


def test_cache_retention_env_long_sets_ttl():
    original = os.environ.get("PI_CACHE_RETENTION")
    _set_env("long")
    try:
        retention, cache_control = anthropic_provider._get_cache_control(
            "https://api.anthropic.com/v1", None
        )
    finally:
        _set_env(original)

    assert retention == "long"
    assert cache_control == {"type": "ephemeral", "ttl": "1h"}


def test_cache_retention_env_long_no_ttl_for_proxy():
    original = os.environ.get("PI_CACHE_RETENTION")
    _set_env("long")
    try:
        retention, cache_control = anthropic_provider._get_cache_control(
            "https://proxy.example.com/v1", None
        )
    finally:
        _set_env(original)

    assert retention == "long"
    assert cache_control == {"type": "ephemeral"}


def test_cache_retention_none_disables_cache_control():
    retention, cache_control = anthropic_provider._get_cache_control(
        "https://api.anthropic.com/v1", "none"
    )
    assert retention == "none"
    assert cache_control is None


def test_cache_retention_explicit_short_overrides_env():
    original = os.environ.get("PI_CACHE_RETENTION")
    _set_env("long")
    try:
        retention, cache_control = anthropic_provider._get_cache_control(
            "https://api.anthropic.com/v1", "short"
        )
    finally:
        _set_env(original)

    assert retention == "short"
    assert cache_control == {"type": "ephemeral"}

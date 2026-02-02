from pi_ai.models import create_openai_model
from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking, build_base_options
from pi_ai.types import SimpleStreamOptions


def test_build_base_options_defaults_max_tokens():
    model = create_openai_model("gpt-4o-mini", provider="openai", max_tokens=50000)
    base = build_base_options(model, SimpleStreamOptions(), api_key="key")
    assert base.max_tokens == 32000

    model = create_openai_model("gpt-4o-mini", provider="openai", max_tokens=1000)
    base = build_base_options(model, SimpleStreamOptions(), api_key="key")
    assert base.max_tokens == 1000

    options = SimpleStreamOptions(max_tokens=1234)
    base = build_base_options(model, options, api_key="key")
    assert base.max_tokens == 1234


def test_adjust_max_tokens_for_thinking_caps_to_model():
    max_tokens, thinking_budget = adjust_max_tokens_for_thinking(8000, 16000, "high")
    assert max_tokens == 16000
    assert thinking_budget == 14976

from pi_ai.providers import anthropic as anthropic_provider
from pi_ai.providers import openai as openai_provider


def test_openai_streaming_json_empty_returns_dict():
    assert openai_provider._parse_streaming_json("") == {}


def test_openai_streaming_json_parses_prefix():
    raw = '{"a": 1}{"b": 2'
    assert openai_provider._parse_streaming_json(raw) == {"a": 1}


def test_anthropic_streaming_json_empty_returns_dict():
    assert anthropic_provider._parse_streaming_json("") == {}


def test_anthropic_streaming_json_parses_prefix():
    raw = '{"ok": true}{"next": 3'
    assert anthropic_provider._parse_streaming_json(raw) == {"ok": True}

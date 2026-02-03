import pytest

from pi_ai.stream import complete, complete_simple, stream, stream_simple
from pi_ai.models import create_anthropic_model, create_openai_model
from pi_ai.streaming import AssistantMessageEventStream
from pi_ai.types import Context, TextContent
from pi_ai.providers import anthropic as anthropic_provider
from pi_ai.providers import openai as openai_provider

from tests.helpers import create_assistant_message


def test_stream_dispatches_openai(monkeypatch):
    model = create_openai_model("gpt-4o-mini", provider="openai")
    ctx = Context(messages=[])
    sentinel = object()

    def fake_stream(_model, _context, _options=None):
        return sentinel

    monkeypatch.setattr(openai_provider, "stream_openai_completions", fake_stream)
    assert stream(model, ctx) is sentinel


def test_stream_dispatches_anthropic(monkeypatch):
    model = create_anthropic_model("claude-sonnet-4-5", provider="anthropic")
    ctx = Context(messages=[])
    sentinel = object()

    def fake_stream(_model, _context, _options=None):
        return sentinel

    monkeypatch.setattr(anthropic_provider, "stream_anthropic", fake_stream)
    assert stream(model, ctx) is sentinel


def test_stream_simple_dispatches_openai(monkeypatch):
    model = create_openai_model("gpt-4o-mini", provider="openai")
    ctx = Context(messages=[])
    sentinel = object()

    def fake_stream(_model, _context, _options=None):
        return sentinel

    monkeypatch.setattr(openai_provider, "stream_simple_openai_completions", fake_stream)
    assert stream_simple(model, ctx) is sentinel


def test_stream_simple_dispatches_anthropic(monkeypatch):
    model = create_anthropic_model("claude-sonnet-4-5", provider="anthropic")
    ctx = Context(messages=[])
    sentinel = object()

    def fake_stream(_model, _context, _options=None):
        return sentinel

    monkeypatch.setattr(anthropic_provider, "stream_simple_anthropic", fake_stream)
    assert stream_simple(model, ctx) is sentinel


@pytest.mark.asyncio
async def test_complete_returns_stream_result(monkeypatch):
    model = create_openai_model("gpt-4o-mini", provider="openai")
    ctx = Context(messages=[])
    stream = AssistantMessageEventStream()
    message = create_assistant_message([TextContent(text="ok")])

    stream.push({"type": "done", "reason": "stop", "message": message})
    stream.end(message)

    def fake_stream(_model, _context, _options=None):
        return stream

    monkeypatch.setattr(openai_provider, "stream_openai_completions", fake_stream)
    result = await complete(model, ctx)
    assert result == message


@pytest.mark.asyncio
async def test_complete_simple_returns_stream_result(monkeypatch):
    model = create_anthropic_model("claude-sonnet-4-5", provider="anthropic")
    ctx = Context(messages=[])
    stream = AssistantMessageEventStream()
    message = create_assistant_message([TextContent(text="ok")])

    stream.push({"type": "done", "reason": "stop", "message": message})
    stream.end(message)

    def fake_stream(_model, _context, _options=None):
        return stream

    monkeypatch.setattr(anthropic_provider, "stream_simple_anthropic", fake_stream)
    result = await complete_simple(model, ctx)
    assert result == message

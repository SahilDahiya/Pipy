"""Test helpers for pi-python."""

from __future__ import annotations

from typing import List

from pi_ai.models import create_openai_model
from pi_ai.types import AssistantMessage, Message, TextContent, ToolCall, UserMessage


def create_model():
    return create_openai_model("mock", provider="openai")


def create_assistant_message(
    content: List[TextContent | ToolCall],
    stop_reason: str = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-completions",
        provider="openai",
        model="mock",
        stop_reason=stop_reason,
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text)


def identity_converter(messages: List[Message]) -> List[Message]:
    return [m for m in messages if m.role in {"user", "assistant", "tool_result"}]

"""Core types for messages, tools, models, and stream events."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Api = Literal["openai-completions", "anthropic-messages"]
StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]
CacheRetention = Literal["none", "short", "long"]


class ModelCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class UsageCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


class OpenAICompletionsCompat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supports_store: Optional[bool] = None
    supports_developer_role: Optional[bool] = None
    supports_reasoning_effort: Optional[bool] = None
    supports_usage_in_streaming: Optional[bool] = None
    supports_strict_mode: Optional[bool] = None
    max_tokens_field: Optional[Literal["max_completion_tokens", "max_tokens"]] = None
    requires_tool_result_name: Optional[bool] = None
    requires_assistant_after_tool_result: Optional[bool] = None
    requires_thinking_as_text: Optional[bool] = None
    requires_mistral_tool_ids: Optional[bool] = None
    thinking_format: Optional[Literal["openai", "zai", "qwen"]] = None
    openrouter_routing: Optional[Dict[str, List[str]]] = None
    vercel_gateway_routing: Optional[Dict[str, List[str]]] = None


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    api: Api
    provider: str
    name: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    reasoning: bool = False
    input: List[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    compat: Optional[OpenAICompletionsCompat] = None
    supports_xhigh: bool = False


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = None


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ThinkingContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    thought_signature: Optional[str] = None


UserContentBlock = Union[TextContent, ImageContent]
AssistantContentBlock = Union[TextContent, ThinkingContent, ToolCall]


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, Any]


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: Union[str, List[UserContentBlock]]
    timestamp: Optional[int] = None


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: List[AssistantContentBlock] = Field(default_factory=list)
    api: Api
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolResultMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: List[UserContentBlock] = Field(default_factory=list)
    details: Optional[Dict[str, Any]] = None
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    tools: Optional[List[Tool]] = None


@dataclass
class StreamOptions:
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    signal: Optional[asyncio.Event] = None
    session_id: Optional[str] = None
    on_payload: Optional[Callable[[Dict[str, Any]], None]] = None
    cache_retention: Optional[CacheRetention] = None
    max_retry_delay_ms: Optional[int] = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    reasoning: Optional[Literal["minimal", "low", "medium", "high", "xhigh"]] = None
    thinking_budgets: Optional[Dict[str, int]] = None
    tool_choice: Optional[str | Dict[str, Dict[str, str]]] = None

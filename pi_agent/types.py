"""Core agent types for pi-python."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence

from pi_ai.types import Message, Model, SimpleStreamOptions
from pi_tools.base import ToolDefinition, ToolResult

ThinkingLevel = Optional[Literal["off", "minimal", "low", "medium", "high", "xhigh"]]

AgentMessage = Message


@dataclass
class AgentContext:
    system_prompt: str
    messages: List[AgentMessage]
    tools: Optional[List[ToolDefinition]] = None


@dataclass
class AgentLoopConfig:
    model: Model
    convert_to_llm: Callable[[List[AgentMessage]], List[Message] | Awaitable[List[Message]]]
    transform_context: Optional[
        Callable[[List[AgentMessage], Optional[asyncio.Event]], List[AgentMessage] | Awaitable[List[AgentMessage]]]
    ] = None
    stream_fn: Optional[Callable[[Model, Any, SimpleStreamOptions], Any | Awaitable[Any]]] = None
    get_api_key: Optional[Callable[[str], Awaitable[Optional[str]] | Optional[str]]] = None
    get_steering_messages: Optional[Callable[[], Sequence[AgentMessage] | Awaitable[Sequence[AgentMessage]]]] = None
    get_follow_up_messages: Optional[Callable[[], Sequence[AgentMessage] | Awaitable[Sequence[AgentMessage]]]] = None
    reasoning: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    thinking_budgets: Optional[Dict[str, int]] = None
    on_payload: Optional[Callable[[Dict[str, Any]], None]] = None
    signal: Optional[asyncio.Event] = None


@dataclass
class AgentToolResult:
    content: List[Any]
    details: Optional[Dict[str, Any]] = None


AgentEvent = Dict[str, Any]

StreamFn = Callable[[Model, Any, SimpleStreamOptions], Any]

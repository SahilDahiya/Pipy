"""Core agent types for pi-python."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pi_ai.types import Message, Model, SimpleStreamOptions
from pi_tools.base import ToolDefinition, ToolResult

ThinkingLevel = Optional[str]

AgentMessage = Message


@dataclass
class AgentContext:
    system_prompt: str
    messages: List[AgentMessage]
    tools: Optional[List[ToolDefinition]] = None


@dataclass
class AgentLoopConfig:
    model: Model
    convert_to_llm: Callable[[List[AgentMessage]], List[Message]]
    stream_fn: Optional[Callable[[Model, Any, SimpleStreamOptions], Any]] = None
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

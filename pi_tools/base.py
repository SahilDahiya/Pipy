"""Tool definition primitives for pi-python."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pi_ai.types import ImageContent, TextContent, Tool

ToolContent = TextContent | ImageContent
ToolUpdateCallback = Callable[["ToolResult"], None]


@dataclass
class ToolResult:
    content: List[ToolContent]
    details: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    name: str
    label: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[[str, Dict[str, Any], Optional[asyncio.Event], Optional[ToolUpdateCallback]], Awaitable[ToolResult]]

    def to_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, parameters=self.parameters)

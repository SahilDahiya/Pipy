"""Write tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from pi_ai.types import TextContent

from .base import ToolDefinition, ToolResult
from .path_utils import resolve_to_cwd

WRITE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def create_write_tool(cwd: str) -> ToolDefinition:
    async def execute(
        _tool_call_id: str,
        params: Dict[str, object],
        signal: Optional[asyncio.Event] = None,
        _on_update=None,
    ) -> ToolResult:
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        path = str(params.get("path"))
        content = str(params.get("content", ""))
        absolute_path = Path(resolve_to_cwd(path, cwd))
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(content, encoding="utf-8")
        return ToolResult(
            content=[TextContent(text=f"Successfully wrote {len(content)} bytes to {path}")],
            details=None,
        )

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=WRITE_SCHEMA,
        execute=execute,
    )


def write_tool() -> ToolDefinition:
    return create_write_tool(str(Path.cwd()))

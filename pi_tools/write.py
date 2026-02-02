"""Write tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

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


@dataclass
class WriteOperations:
    write_file: Callable[[str, str], Awaitable[None]]
    mkdir: Callable[[str], Awaitable[None]]


async def _default_write_file(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


async def _default_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


DEFAULT_WRITE_OPERATIONS = WriteOperations(write_file=_default_write_file, mkdir=_default_mkdir)


@dataclass
class WriteToolOptions:
    operations: Optional[WriteOperations] = None


def create_write_tool(cwd: str, options: Optional[WriteToolOptions] = None) -> ToolDefinition:
    ops = (options.operations if options else None) or DEFAULT_WRITE_OPERATIONS

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
        absolute_path = resolve_to_cwd(path, cwd)
        directory = str(Path(absolute_path).parent)

        await ops.mkdir(directory)
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        await ops.write_file(absolute_path, content)
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")
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

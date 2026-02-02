"""Read tool."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pi_ai.types import ImageContent, TextContent

from .base import ToolDefinition, ToolResult
from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_head

READ_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


@dataclass
class ReadToolDetails:
    truncation: Optional[TruncationResult] = None


def _detect_image_mime(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".apng"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    return None


def create_read_tool(cwd: str) -> ToolDefinition:
    async def execute(
        _tool_call_id: str,
        params: Dict[str, object],
        signal: Optional[asyncio.Event] = None,
        _on_update=None,
    ) -> ToolResult:
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        path = str(params.get("path"))
        offset = params.get("offset")
        limit = params.get("limit")

        absolute_path = Path(resolve_read_path(path, cwd))
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime_type = _detect_image_mime(absolute_path)
        if mime_type:
            data = absolute_path.read_bytes()
            encoded = base64.b64encode(data).decode("ascii")
            content = [
                TextContent(text=f"Read image file [{mime_type}]"),
                ImageContent(data=encoded, mime_type=mime_type),
            ]
            return ToolResult(content=content, details=None)

        text = absolute_path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        total_lines = len(lines)

        start_line = 0
        if isinstance(offset, (int, float)):
            start_line = max(0, int(offset) - 1)
        if start_line >= total_lines:
            raise ValueError(f"Offset {offset} is beyond end of file ({total_lines} lines total)")

        selected = lines[start_line:]
        if isinstance(limit, (int, float)):
            selected = selected[: int(limit)]

        selected_text = "\n".join(selected)
        truncation = truncate_head(selected_text)

        if truncation.first_line_exceeds_limit:
            line_size = format_size(len(lines[start_line].encode("utf-8")))
            message = (
                f"[Line {start_line + 1} is {line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line + 1}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            return ToolResult(
                content=[TextContent(text=message)],
                details=ReadToolDetails(truncation=truncation).__dict__,
            )

        output_text = truncation.content
        if truncation.truncated:
            end_line = start_line + truncation.output_lines
            next_offset = end_line + 1
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line + 1}-{end_line} of {total_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line + 1}-{end_line} of {total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
        elif limit is not None and start_line + len(selected) < total_lines:
            remaining = total_lines - (start_line + len(selected))
            next_offset = start_line + len(selected) + 1
            output_text += (
                f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        return ToolResult(
            content=[TextContent(text=output_text)],
            details=ReadToolDetails(truncation=truncation).__dict__,
        )

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
            f"Text output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
            "Use offset/limit for large files."
        ),
        parameters=READ_SCHEMA,
        execute=execute,
    )


def read_tool() -> ToolDefinition:
    return create_read_tool(str(Path.cwd()))

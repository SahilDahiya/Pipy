"""Read tool."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

from pi_ai.types import ImageContent, TextContent

from .base import ToolDefinition, ToolResult
from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head

READ_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


def _detect_image_mime_from_bytes(data: bytes) -> Optional[str]:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _detect_image_mime_from_file(path: str) -> Optional[str]:
    with open(path, "rb") as handle:
        header = handle.read(32)
    return _detect_image_mime_from_bytes(header)


def _resize_image_if_needed(data: bytes, mime_type: str, max_dim: int) -> tuple[bytes, Optional[str], str]:
    try:
        from PIL import Image
    except Exception:
        return data, None, mime_type

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width <= max_dim and height <= max_dim:
            return data, None, mime_type
        image.thumbnail((max_dim, max_dim))
        output = io.BytesIO()
        format_name = (image.format or "").upper() or "PNG"
        image.save(output, format=format_name)
        new_data = output.getvalue()
        format_to_mime = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "GIF": "image/gif",
            "WEBP": "image/webp",
        }
        resolved_mime = format_to_mime.get(format_name, mime_type)
        return new_data, f"Resized image to {image.width}x{image.height}", resolved_mime
    except Exception:
        return data, None, mime_type


@dataclass
class ReadOperations:
    read_file: Callable[[str], Awaitable[bytes]]
    access: Callable[[str], Awaitable[None]]
    detect_image_mime_type: Optional[Callable[[str], Awaitable[Optional[str]]]] = None


async def _default_read_file(path: str) -> bytes:
    return Path(path).read_bytes()


async def _default_access(path: str) -> None:
    if not os.access(path, os.R_OK):
        raise FileNotFoundError(path)


DEFAULT_READ_OPERATIONS = ReadOperations(
    read_file=_default_read_file,
    access=_default_access,
    detect_image_mime_type=_detect_image_mime_from_file,
)


@dataclass
class ReadToolOptions:
    auto_resize_images: bool = True
    operations: Optional[ReadOperations] = None


def create_read_tool(cwd: str, options: Optional[ReadToolOptions] = None) -> ToolDefinition:
    opts = options or ReadToolOptions()
    ops = opts.operations or DEFAULT_READ_OPERATIONS
    auto_resize_images = opts.auto_resize_images

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

        absolute_path = resolve_read_path(path, cwd)
        await ops.access(absolute_path)

        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        mime_type = None
        if ops.detect_image_mime_type:
            mime_type = await ops.detect_image_mime_type(absolute_path)

        if mime_type:
            data = await ops.read_file(absolute_path)
            if auto_resize_images:
                data, resize_note, mime_type = _resize_image_if_needed(data, mime_type, 2000)
            else:
                resize_note = None
            encoded = base64.b64encode(data).decode("ascii")
            text_note = f"Read image file [{mime_type}]"
            if resize_note:
                text_note += f"\n{resize_note}"
            content = [
                TextContent(text=text_note),
                ImageContent(data=encoded, mime_type=mime_type),
            ]
            return ToolResult(content=content, details=None)

        text = (await ops.read_file(absolute_path)).decode("utf-8", errors="replace")
        lines = text.split("\n")
        total_lines = len(lines)

        start_line = 0
        if isinstance(offset, (int, float)):
            start_line = max(0, int(offset) - 1)
        if start_line >= total_lines:
            raise ValueError(f"Offset {offset} is beyond end of file ({total_lines} lines total)")

        selected = lines[start_line:]
        user_limited_lines: Optional[int] = None
        if isinstance(limit, (int, float)):
            end_line = min(start_line + int(limit), total_lines)
            selected = lines[start_line:end_line]
            user_limited_lines = end_line - start_line

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
                details={"truncation": truncation.__dict__},
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
        elif user_limited_lines is not None and start_line + user_limited_lines < total_lines:
            remaining = total_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text += (
                f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        return ToolResult(
            content=[TextContent(text=output_text)],
            details={"truncation": truncation.__dict__} if truncation.truncated else None,
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

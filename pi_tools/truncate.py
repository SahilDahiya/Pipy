"""Output truncation utilities for tools."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 1000
DEFAULT_MAX_BYTES = 30 * 1024


@dataclass
class TruncationResult:
    truncated: bool
    truncated_by: str | None
    output_lines: int
    total_lines: int
    output_bytes: int
    total_bytes: int
    first_line_exceeds_limit: bool = False
    content: str = ""


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(text: str) -> TruncationResult:
    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)
    truncated = False
    truncated_by = None

    output_lines = lines
    if total_lines > DEFAULT_MAX_LINES:
        output_lines = lines[:DEFAULT_MAX_LINES]
        truncated = True
        truncated_by = "lines"

    output_text = "\n".join(output_lines)
    output_bytes = len(output_text.encode("utf-8"))

    if output_bytes > DEFAULT_MAX_BYTES:
        truncated = True
        truncated_by = "bytes"
        encoded = output_text.encode("utf-8")
        output_text = encoded[:DEFAULT_MAX_BYTES].decode("utf-8", errors="ignore")
        output_bytes = len(output_text.encode("utf-8"))

    first_line_exceeds = False
    if lines:
        first_line_bytes = len(lines[0].encode("utf-8"))
        if first_line_bytes > DEFAULT_MAX_BYTES:
            first_line_exceeds = True

    return TruncationResult(
        truncated=truncated,
        truncated_by=truncated_by,
        output_lines=len(output_text.split("\n")) if output_text else 0,
        total_lines=total_lines,
        output_bytes=output_bytes,
        total_bytes=total_bytes,
        first_line_exceeds_limit=first_line_exceeds,
        content=output_text,
    )


def truncate_tail(text: str) -> TruncationResult:
    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)
    truncated = False
    truncated_by = None

    output_lines = lines
    if total_lines > DEFAULT_MAX_LINES:
        output_lines = lines[-DEFAULT_MAX_LINES:]
        truncated = True
        truncated_by = "lines"

    output_text = "\n".join(output_lines)
    output_bytes = len(output_text.encode("utf-8"))

    if output_bytes > DEFAULT_MAX_BYTES:
        truncated = True
        truncated_by = "bytes"
        encoded = output_text.encode("utf-8")
        output_text = encoded[-DEFAULT_MAX_BYTES:].decode("utf-8", errors="ignore")
        output_bytes = len(output_text.encode("utf-8"))

    return TruncationResult(
        truncated=truncated,
        truncated_by=truncated_by,
        output_lines=len(output_text.split("\n")) if output_text else 0,
        total_lines=total_lines,
        output_bytes=output_bytes,
        total_bytes=total_bytes,
        content=output_text,
    )

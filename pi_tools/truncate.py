"""Output truncation utilities for tools."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(text: str, *, max_lines: int | None = None, max_bytes: int | None = None) -> TruncationResult:
    max_lines = DEFAULT_MAX_LINES if max_lines is None else max_lines
    max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes

    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if lines:
        first_line_bytes = len(lines[0].encode("utf-8"))
        if first_line_bytes > max_bytes:
            return TruncationResult(
                content="",
                truncated=True,
                truncated_by="bytes",
                total_lines=total_lines,
                total_bytes=total_bytes,
                output_lines=0,
                output_bytes=0,
                last_line_partial=False,
                first_line_exceeds_limit=True,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )

    output_lines_arr: list[str] = []
    output_bytes = 0
    truncated_by: str = "lines"

    for idx, line in enumerate(lines):
        if idx >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + (1 if idx > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes += line_bytes

    output_text = "\n".join(output_lines_arr)
    final_output_bytes = len(output_text.encode("utf-8"))

    return TruncationResult(
        content=output_text,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    truncated = encoded[-max_bytes:]
    return truncated.decode("utf-8", errors="ignore")


def truncate_tail(text: str, *, max_lines: int | None = None, max_bytes: int | None = None) -> TruncationResult:
    max_lines = DEFAULT_MAX_LINES if max_lines is None else max_lines
    max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes

    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: list[str] = []
    output_bytes = 0
    truncated_by: str = "lines"
    last_line_partial = False

    for idx in range(len(lines) - 1, -1, -1):
        if len(output_lines_arr) >= max_lines:
            truncated_by = "lines"
            break
        line = lines[idx]
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines_arr else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes = len(truncated_line.encode("utf-8"))
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes += line_bytes

    output_text = "\n".join(output_lines_arr)
    final_output_bytes = len(output_text.encode("utf-8"))

    return TruncationResult(
        content=output_text,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )

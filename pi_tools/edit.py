"""Edit tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

from pi_ai.types import TextContent

from .base import ToolDefinition, ToolResult
from .edit_diff import (
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .path_utils import resolve_to_cwd

EDIT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "oldText": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
        "newText": {"type": "string", "description": "New text to replace the old text with"},
    },
    "required": ["path", "oldText", "newText"],
}


@dataclass
class EditToolDetails:
    diff: str
    first_changed_line: Optional[int] = None


@dataclass
class EditOperations:
    read_file: Callable[[str], Awaitable[bytes]]
    write_file: Callable[[str, str], Awaitable[None]]
    access: Callable[[str], Awaitable[None]]


async def _default_read_file(path: str) -> bytes:
    return Path(path).read_bytes()


async def _default_write_file(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


async def _default_access(path: str) -> None:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(path)
    if not resolved.is_file():
        raise FileNotFoundError(path)


DEFAULT_EDIT_OPERATIONS = EditOperations(
    read_file=_default_read_file,
    write_file=_default_write_file,
    access=_default_access,
)


@dataclass
class EditToolOptions:
    operations: Optional[EditOperations] = None


def create_edit_tool(cwd: str, options: Optional[EditToolOptions] = None) -> ToolDefinition:
    ops = (options.operations if options else None) or DEFAULT_EDIT_OPERATIONS

    async def execute(
        _tool_call_id: str,
        params: Dict[str, object],
        signal: Optional[asyncio.Event] = None,
        _on_update=None,
    ) -> ToolResult:
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        path = str(params.get("path"))
        old_text = str(params.get("oldText", ""))
        new_text = str(params.get("newText", ""))

        absolute_path = resolve_to_cwd(path, cwd)
        try:
            await ops.access(absolute_path)
        except Exception as exc:
            raise FileNotFoundError(f"File not found: {path}") from exc

        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        raw_content = (await ops.read_file(absolute_path)).decode("utf-8", errors="replace")
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        bom, content = strip_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        normalized_old_text = normalize_to_lf(old_text)
        normalized_new_text = normalize_to_lf(new_text)

        match_result = fuzzy_find_text(normalized_content, normalized_old_text)
        if not match_result.found:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
            )

        fuzzy_content = normalize_for_fuzzy_match(normalized_content)
        fuzzy_old_text = normalize_for_fuzzy_match(normalized_old_text)
        occurrences = fuzzy_content.count(fuzzy_old_text)
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
            )

        base_content = match_result.content_for_replacement
        updated = (
            base_content[: match_result.index]
            + normalized_new_text
            + base_content[match_result.index + match_result.match_length :]
        )
        if updated == base_content:
            raise ValueError(
                f"No changes made to {path}. The replacement produced identical content. "
                "This might indicate an issue with special characters or the text not existing as expected."
            )

        final_content = bom + restore_line_endings(updated, original_ending)
        await ops.write_file(absolute_path, final_content)
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        diff, first_changed_line = generate_diff_string(base_content, updated)
        details = EditToolDetails(diff=diff, first_changed_line=first_changed_line)

        return ToolResult(
            content=[TextContent(text=f"Successfully replaced text in {path}.")],
            details=details.__dict__,
        )

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a file by replacing exact text. The oldText must match exactly (including whitespace). "
            "Use this for precise, surgical edits."
        ),
        parameters=EDIT_SCHEMA,
        execute=execute,
    )


def edit_tool() -> ToolDefinition:
    return create_edit_tool(str(Path.cwd()))

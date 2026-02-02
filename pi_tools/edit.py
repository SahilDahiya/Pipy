"""Edit tool."""

from __future__ import annotations

import asyncio
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pi_ai.types import TextContent

from .base import ToolDefinition, ToolResult
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


def _first_changed_line(old: str, new: str) -> Optional[int]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    for idx, (old_line, new_line) in enumerate(zip(old_lines, new_lines), start=1):
        if old_line != new_line:
            return idx
    if len(old_lines) != len(new_lines):
        return min(len(old_lines), len(new_lines)) + 1
    return None


def create_edit_tool(cwd: str) -> ToolDefinition:
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

        absolute_path = Path(resolve_to_cwd(path, cwd))
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = absolute_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            raise ValueError(
                f"Could not find the exact text in {path}. The oldText must match exactly including whitespace."
            )
        if content.count(old_text) > 1:
            raise ValueError(
                f"Found multiple occurrences of the text in {path}. Provide more context to make it unique."
            )

        updated = content.replace(old_text, new_text, 1)
        if updated == content:
            raise ValueError(
                f"No changes made to {path}. The replacement produced identical content."
            )

        absolute_path.write_text(updated, encoding="utf-8")

        diff_lines = difflib.unified_diff(
            content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
        )
        diff = "".join(diff_lines)
        details = EditToolDetails(diff=diff, first_changed_line=_first_changed_line(content, updated))

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

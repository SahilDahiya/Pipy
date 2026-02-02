"""Bash tool."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pi_ai.types import TextContent

from .base import ToolDefinition, ToolResult, ToolUpdateCallback
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_tail

BASH_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


@dataclass
class BashToolDetails:
    truncation: Optional[TruncationResult] = None
    full_output_path: Optional[str] = None


async def _read_stream(stream: asyncio.StreamReader, on_chunk) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = await stream.read(1024)
        if not chunk:
            break
        chunks.append(chunk)
        on_chunk(chunk)
    return b"".join(chunks)


def create_bash_tool(cwd: str) -> ToolDefinition:
    async def execute(
        _tool_call_id: str,
        params: Dict[str, object],
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[ToolUpdateCallback] = None,
    ) -> ToolResult:
        if signal and signal.is_set():
            raise RuntimeError("Operation aborted")

        command = str(params.get("command"))
        timeout = params.get("timeout")
        timeout_secs = float(timeout) if isinstance(timeout, (int, float)) else None

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        output_chunks: list[bytes] = []

        def on_chunk(chunk: bytes) -> None:
            output_chunks.append(chunk)
            if on_update:
                text = b"".join(output_chunks).decode("utf-8", errors="replace")
                truncation = truncate_tail(text)
                on_update(
                    ToolResult(
                        content=[TextContent(text=truncation.content or "")],
                        details={"truncation": truncation.__dict__ if truncation.truncated else None},
                    )
                )

        stdout_task = asyncio.create_task(_read_stream(process.stdout, on_chunk))
        stderr_task = asyncio.create_task(_read_stream(process.stderr, on_chunk))

        try:
            if timeout_secs:
                await asyncio.wait_for(process.wait(), timeout=timeout_secs)
            else:
                await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"Command timed out after {timeout_secs} seconds")

        if signal and signal.is_set():
            process.kill()
            await process.wait()
            raise RuntimeError("Operation aborted")

        stdout_data, stderr_data = await asyncio.gather(stdout_task, stderr_task)
        combined = stdout_data + stderr_data
        text_output = combined.decode("utf-8", errors="replace")
        truncation = truncate_tail(text_output)
        details = BashToolDetails()

        if truncation.truncated:
            temp = tempfile.NamedTemporaryFile(prefix="pi-bash-", suffix=".log", delete=False)
            temp.write(combined)
            temp.flush()
            temp.close()
            details.truncation = truncation
            details.full_output_path = temp.name

            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.truncated_by == "lines":
                truncation.content += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp.name}]"
                )
            else:
                truncation.content += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp.name}]"
                )

        if process.returncode and process.returncode != 0:
            message = truncation.content or text_output or "(no output)"
            message += f"\n\nCommand exited with code {process.returncode}"
            raise RuntimeError(message)

        output_text = truncation.content or text_output or "(no output)"
        return ToolResult(
            content=[TextContent(text=output_text)],
            details=details.__dict__ if (details.truncation or details.full_output_path) else None,
        )

    return ToolDefinition(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB."
        ),
        parameters=BASH_SCHEMA,
        execute=execute,
    )


def bash_tool() -> ToolDefinition:
    return create_bash_tool(str(Path.cwd()))

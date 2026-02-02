"""Bash tool."""

from __future__ import annotations

import asyncio
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

from pi_ai.types import TextContent

from .base import ToolDefinition, ToolResult, ToolUpdateCallback
from .shell import get_shell_config, get_shell_env, kill_process_tree
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


@dataclass
class BashOperations:
    exec: Callable[
        [
            str,
            str,
            Dict[str, object],
        ],
        Awaitable[Dict[str, Optional[int]]],
    ]


@dataclass
class BashSpawnContext:
    command: str
    cwd: str
    env: Dict[str, str]


BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


async def _default_exec(command: str, cwd: str, options: Dict[str, object]) -> Dict[str, Optional[int]]:
    shell_config = get_shell_config()
    shell = shell_config["shell"]
    args = shell_config["args"]

    if not Path(cwd).exists():
        raise RuntimeError(f"Working directory does not exist: {cwd}\nCannot execute bash commands.")

    env = options.get("env") or get_shell_env()
    signal = options.get("signal")
    timeout = options.get("timeout")
    on_data = options.get("on_data")

    process = await asyncio.create_subprocess_exec(
        shell,
        *args,
        command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def _pump(stream: asyncio.StreamReader) -> None:
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break
            if on_data:
                on_data(chunk)

    stdout_task = asyncio.create_task(_pump(process.stdout))
    stderr_task = asyncio.create_task(_pump(process.stderr))

    start_time = asyncio.get_event_loop().time()
    try:
        while True:
            if signal is not None and isinstance(signal, asyncio.Event) and signal.is_set():
                if process.pid:
                    kill_process_tree(process.pid)
                await process.wait()
                raise RuntimeError("aborted")

            if timeout is not None and timeout > 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    if process.pid:
                        kill_process_tree(process.pid)
                    await process.wait()
                    raise RuntimeError(f"timeout:{timeout}")

            try:
                await asyncio.wait_for(process.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                continue
    finally:
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    return {"exit_code": process.returncode}


DEFAULT_BASH_OPERATIONS = BashOperations(exec=_default_exec)


@dataclass
class BashToolOptions:
    operations: Optional[BashOperations] = None
    command_prefix: Optional[str] = None
    spawn_hook: Optional[BashSpawnHook] = None


def _get_temp_file_path() -> str:
    token = secrets.token_hex(8)
    return str(Path(tempfile.gettempdir()) / f"pi-bash-{token}.log")


def _resolve_spawn_context(
    command: str, cwd: str, spawn_hook: Optional[BashSpawnHook]
) -> BashSpawnContext:
    context = BashSpawnContext(command=command, cwd=cwd, env=dict(get_shell_env()))
    return spawn_hook(context) if spawn_hook else context


def create_bash_tool(cwd: str, options: Optional[BashToolOptions] = None) -> ToolDefinition:
    opts = options or BashToolOptions()
    ops = opts.operations or DEFAULT_BASH_OPERATIONS
    command_prefix = opts.command_prefix
    spawn_hook = opts.spawn_hook

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

        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command
        spawn_context = _resolve_spawn_context(resolved_command, cwd, spawn_hook)

        temp_file_path: Optional[str] = None
        temp_file_handle: Optional[object] = None
        total_bytes = 0

        chunks: list[bytes] = []
        chunks_bytes = 0
        max_chunks_bytes = DEFAULT_MAX_BYTES * 2

        def handle_data(data: bytes) -> None:
            nonlocal temp_file_path, temp_file_handle, total_bytes, chunks_bytes
            total_bytes += len(data)

            if total_bytes > DEFAULT_MAX_BYTES and temp_file_path is None:
                temp_file_path = _get_temp_file_path()
                temp_file_handle = open(temp_file_path, "wb")
                for chunk in chunks:
                    temp_file_handle.write(chunk)

            if temp_file_handle is not None:
                temp_file_handle.write(data)

            chunks.append(data)
            chunks_bytes += len(data)

            while chunks_bytes > max_chunks_bytes and len(chunks) > 1:
                removed = chunks.pop(0)
                chunks_bytes -= len(removed)

            if on_update:
                full_text = b"".join(chunks).decode("utf-8", errors="replace")
                truncation = truncate_tail(full_text)
                on_update(
                    ToolResult(
                        content=[TextContent(text=truncation.content or "")],
                        details={
                            "truncation": truncation.__dict__ if truncation.truncated else None,
                            "full_output_path": temp_file_path,
                        },
                    )
                )

        try:
            result = await ops.exec(
                spawn_context.command,
                spawn_context.cwd,
                {
                    "on_data": handle_data,
                    "signal": signal,
                    "timeout": timeout_secs,
                    "env": spawn_context.env,
                },
            )
        except Exception as exc:
            if temp_file_handle is not None:
                temp_file_handle.close()
            full_buffer = b"".join(chunks)
            output = full_buffer.decode("utf-8", errors="replace")
            message = str(exc)
            if message == "aborted":
                if output:
                    output += "\n\n"
                output += "Command aborted"
                raise RuntimeError(output) from exc
            if message.startswith("timeout:"):
                timeout_value = message.split(":", 1)[1]
                if output:
                    output += "\n\n"
                output += f"Command timed out after {timeout_value} seconds"
                raise RuntimeError(output) from exc
            raise

        if temp_file_handle is not None:
            temp_file_handle.close()

        full_buffer = b"".join(chunks)
        full_output = full_buffer.decode("utf-8", errors="replace")

        truncation = truncate_tail(full_output)
        output_text = truncation.content or "(no output)"

        details: Optional[BashToolDetails] = None

        if truncation.truncated:
            details = BashToolDetails(truncation=truncation, full_output_path=temp_file_path)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines

            if truncation.last_line_partial:
                last_line = full_output.split("\n")[-1] if full_output else ""
                last_line_size = format_size(len(last_line.encode("utf-8")))
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {temp_file_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp_file_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file_path}]"
                )

        exit_code = result.get("exit_code")
        if exit_code not in (0, None):
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise RuntimeError(output_text)

        return ToolResult(
            content=[TextContent(text=output_text)],
            details={
                "truncation": details.truncation.__dict__ if details and details.truncation else None,
                "full_output_path": details.full_output_path if details else None,
            }
            if details
            else None,
        )

    return ToolDefinition(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters=BASH_SCHEMA,
        execute=execute,
    )


def bash_tool() -> ToolDefinition:
    return create_bash_tool(str(Path.cwd()))

"""SDK entry points for embedding pi-python."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from pi_agent.agent import Agent
from pi_ai.models import get_model
from pi_ai.types import Model
from pi_session.manager import SessionManager
from pi_tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool
from pi_tools.base import ToolDefinition


def create_default_tools(cwd: str) -> list[ToolDefinition]:
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_agent(
    *,
    model: Optional[Model] = None,
    provider: str = "openai",
    model_id: str = "gpt-4o-mini",
    system_prompt: str = "",
    cwd: Optional[str] = None,
    tools: Optional[Sequence[ToolDefinition]] = None,
    session_path: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Agent:
    resolved_cwd = cwd or str(Path.cwd())
    resolved_model = model or get_model(provider, model_id)
    resolved_tools = list(tools) if tools is not None else create_default_tools(resolved_cwd)
    session_manager = SessionManager(session_path) if session_path else None

    return Agent(
        model=resolved_model,
        system_prompt=system_prompt,
        tools=resolved_tools,
        session_id=session_id,
        api_key=api_key,
        session_manager=session_manager,
    )

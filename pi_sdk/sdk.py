"""SDK entry points for embedding pi-python."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Literal, Optional, Sequence

from pi_agent.agent import Agent
from pi_ai.auth import AuthStorage
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
    thinking_level: Optional[str] = None,
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
    cwd: Optional[str] = None,
    tools: Optional[Sequence[ToolDefinition]] = None,
    session_path: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    auth_path: Optional[str] = None,
    auth_storage: Optional[AuthStorage] = None,
    convert_to_llm: Optional[Callable[[list], list | Awaitable[list]]] = None,
    transform_context: Optional[Callable[[list, Optional[object]], list | Awaitable[list]]] = None,
    get_steering_messages: Optional[Callable[[], Sequence | Awaitable[Sequence]]] = None,
    get_follow_up_messages: Optional[Callable[[], Sequence | Awaitable[Sequence]]] = None,
) -> Agent:
    resolved_cwd = cwd or str(Path.cwd())
    resolved_model = model or get_model(provider, model_id)
    resolved_tools = list(tools) if tools is not None else create_default_tools(resolved_cwd)
    session_manager = SessionManager(session_path) if session_path else None
    resolved_auth = auth_storage or (AuthStorage(auth_path) if auth_path else None)

    async def resolve_api_key(provider_name: str) -> Optional[str]:
        if api_key:
            return api_key
        if resolved_auth:
            return await resolved_auth.get_api_key(provider_name)
        return None

    agent = Agent(
        model=resolved_model,
        system_prompt=system_prompt,
        tools=resolved_tools,
        thinking_level=thinking_level,
        steering_mode=steering_mode,
        follow_up_mode=follow_up_mode,
        session_id=session_id,
        api_key=api_key,
        get_api_key=resolve_api_key,
        session_manager=session_manager,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )
    if session_manager and Path(session_manager.path).exists():
        agent.replace_messages(session_manager.load_messages())
    return agent

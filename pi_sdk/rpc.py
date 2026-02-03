"""JSON-over-stdin/stdout RPC bridge for pi-python."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional

from pi_agent.agent import Agent
from pi_ai.models import get_model
from pi_ai.types import ImageContent, UserMessage
from pi_sdk.sdk import create_agent


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[call-arg]
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _emit(obj: Any) -> None:
    sys.stdout.write(json.dumps(_to_jsonable(obj)) + "\n")
    sys.stdout.flush()


def _success(command: str, request_id: Optional[str] = None, data: Optional[dict] = None) -> dict:
    payload = {"type": "response", "command": command, "success": True}
    if request_id:
        payload["id"] = request_id
    if data is not None:
        payload["data"] = data
    return payload


def _error(command: str, message: str, request_id: Optional[str] = None) -> dict:
    payload = {"type": "response", "command": command, "success": False, "error": message}
    if request_id:
        payload["id"] = request_id
    return payload


def _parse_images(raw: Any) -> Optional[list[ImageContent]]:
    if not isinstance(raw, list):
        return None
    images: list[ImageContent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        mime_type = item.get("mime_type")
        if isinstance(data, str) and isinstance(mime_type, str):
            images.append(ImageContent(data=data, mime_type=mime_type))
    return images or None


async def _stream_events(stream: Iterable[dict]) -> None:
    async for event in stream:
        _emit(event)


def _build_state(agent: Agent) -> dict:
    state = agent.state
    return {
        "model": _to_jsonable(state.model),
        "thinking_level": state.thinking_level or "off",
        "is_streaming": state.is_streaming,
        "steering_mode": agent.get_steering_mode(),
        "follow_up_mode": agent.get_follow_up_mode(),
        "session_id": agent.session_id or "",
        "message_count": len(state.messages),
        "pending_tool_calls": len(state.pending_tool_calls),
        "error": state.error,
    }


async def _handle_prompt(agent: Agent, payload: Dict[str, Any]) -> None:
    request_id = payload.get("id")
    message = payload.get("message") or payload.get("text")
    if not isinstance(message, str):
        _emit(_error("prompt", "prompt requires a 'message' string", request_id))
        return

    images = _parse_images(payload.get("images"))
    streaming_behavior = payload.get("streaming_behavior")
    if agent.state.is_streaming and streaming_behavior in {"steer", "follow_up"}:
        queued_message = UserMessage(content=message)
        if streaming_behavior == "steer":
            agent.steer(queued_message)
        else:
            agent.follow_up(queued_message)
        _emit(_success("prompt", request_id, {"queued": True}))
        return

    try:
        stream = agent.send(message, images=images)
    except Exception as exc:
        _emit(_error("prompt", str(exc), request_id))
        return

    _emit(_success("prompt", request_id))
    await _stream_events(stream)


async def _handle_steer(agent: Agent, payload: Dict[str, Any]) -> None:
    request_id = payload.get("id")
    message = payload.get("message")
    if not isinstance(message, str):
        _emit(_error("steer", "steer requires a 'message' string", request_id))
        return
    agent.steer(UserMessage(content=message))
    _emit(_success("steer", request_id))


async def _handle_follow_up(agent: Agent, payload: Dict[str, Any]) -> None:
    request_id = payload.get("id")
    message = payload.get("message")
    if not isinstance(message, str):
        _emit(_error("follow_up", "follow_up requires a 'message' string", request_id))
        return
    agent.follow_up(UserMessage(content=message))
    _emit(_success("follow_up", request_id))


async def _handle_set_model(agent: Agent, payload: Dict[str, Any]) -> None:
    request_id = payload.get("id")
    provider = payload.get("provider")
    model_id = payload.get("model_id")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        _emit(_error("set_model", "set_model requires provider and model_id", request_id))
        return
    try:
        model = get_model(provider, model_id)
    except Exception as exc:
        _emit(_error("set_model", str(exc), request_id))
        return
    agent.set_model(model)
    _emit(_success("set_model", request_id, _to_jsonable(model)))


async def _read_lines() -> None:
    provider = os.getenv("PI_PROVIDER", "openai")
    model_id = os.getenv("PI_MODEL", "gpt-4o-mini")
    agent = create_agent(provider=provider, model_id=model_id)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        msg_type = data.get("type")

        if msg_type in {"send", "prompt"}:
            await _handle_prompt(agent, data)
        elif msg_type == "steer":
            await _handle_steer(agent, data)
        elif msg_type == "follow_up":
            await _handle_follow_up(agent, data)
        elif msg_type == "abort":
            agent.abort()
            _emit(_success("abort", data.get("id")))
        elif msg_type in {"reset", "new_session"}:
            agent.reset()
            _emit(_success(msg_type, data.get("id")))
        elif msg_type == "get_state":
            _emit(_success("get_state", data.get("id"), _build_state(agent)))
        elif msg_type == "set_model":
            await _handle_set_model(agent, data)
        elif msg_type == "set_thinking_level":
            level = data.get("level")
            if not isinstance(level, str):
                _emit(_error("set_thinking_level", "level must be a string", data.get("id")))
                continue
            agent.set_thinking_level(level)
            _emit(_success("set_thinking_level", data.get("id")))
        elif msg_type == "set_steering_mode":
            mode = data.get("mode")
            if mode not in {"all", "one-at-a-time"}:
                _emit(_error("set_steering_mode", "mode must be 'all' or 'one-at-a-time'", data.get("id")))
                continue
            agent.set_steering_mode(mode)
            _emit(_success("set_steering_mode", data.get("id")))
        elif msg_type == "set_follow_up_mode":
            mode = data.get("mode")
            if mode not in {"all", "one-at-a-time"}:
                _emit(_error("set_follow_up_mode", "mode must be 'all' or 'one-at-a-time'", data.get("id")))
                continue
            agent.set_follow_up_mode(mode)
            _emit(_success("set_follow_up_mode", data.get("id")))
        elif msg_type == "get_messages":
            _emit(
                _success(
                    "get_messages",
                    data.get("id"),
                    {"messages": _to_jsonable(agent.state.messages)},
                )
            )
        else:
            _emit(_error(msg_type or "unknown", "Unknown message type", data.get("id")))


def main() -> None:
    asyncio.run(_read_lines())


if __name__ == "__main__":
    main()

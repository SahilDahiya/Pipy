"""Serialization helpers for wire compatibility with pi-mono."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from pydantic import BaseModel


_ROLE_TO_CAMEL = {"tool_result": "toolResult"}
_ROLE_TO_SNAKE = {"toolResult": "tool_result"}
_STOP_REASON_TO_CAMEL = {"tool_use": "toolUse"}
_STOP_REASON_TO_SNAKE = {"toolUse": "tool_use"}
_CONTENT_TYPE_TO_CAMEL = {"tool_call": "toolCall"}
_CONTENT_TYPE_TO_SNAKE = {"toolCall": "tool_call"}


def to_camel_key(key: str) -> str:
    if "_" not in key:
        return key
    parts = [part for part in key.split("_") if part]
    if not parts:
        return key
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def to_snake_key(key: str) -> str:
    if "_" in key:
        return key
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def convert_keys(value: Any, key_fn) -> Any:
    if isinstance(value, dict):
        converted: Dict[Any, Any] = {}
        for key, val in value.items():
            new_key = key_fn(key) if isinstance(key, str) else key
            converted[new_key] = convert_keys(val, key_fn)
        return converted
    if isinstance(value, list):
        return [convert_keys(item, key_fn) for item in value]
    return value


def to_camel_dict(value: Any) -> Any:
    return convert_keys(_to_plain(value), to_camel_key)


def to_snake_dict(value: Any) -> Any:
    return convert_keys(_to_plain(value), to_snake_key)


def _normalize_message_values_to_camel(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message.get("role")
    if role in _ROLE_TO_CAMEL:
        message["role"] = _ROLE_TO_CAMEL[role]

    stop_reason = message.get("stopReason")
    if stop_reason in _STOP_REASON_TO_CAMEL:
        message["stopReason"] = _STOP_REASON_TO_CAMEL[stop_reason]

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in _CONTENT_TYPE_TO_CAMEL:
                    block["type"] = _CONTENT_TYPE_TO_CAMEL[block_type]
    return message


def _normalize_message_values_to_snake(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message.get("role")
    if role in _ROLE_TO_SNAKE:
        message["role"] = _ROLE_TO_SNAKE[role]

    stop_reason = message.get("stop_reason")
    if stop_reason in _STOP_REASON_TO_SNAKE:
        message["stop_reason"] = _STOP_REASON_TO_SNAKE[stop_reason]

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in _CONTENT_TYPE_TO_SNAKE:
                    block["type"] = _CONTENT_TYPE_TO_SNAKE[block_type]
    return message


def to_wire_message(message: Any) -> Dict[str, Any]:
    data = to_camel_dict(message)
    if isinstance(data, dict):
        return _normalize_message_values_to_camel(data)
    return data


def from_wire_message(message: Any) -> Any:
    if not isinstance(message, (dict, BaseModel)):
        return message
    data = to_snake_dict(message)
    if isinstance(data, dict):
        return _normalize_message_values_to_snake(data)
    return data


def to_wire_content_block(block: Any) -> Any:
    data = to_camel_dict(block)
    if isinstance(data, dict):
        block_type = data.get("type")
        if block_type in _CONTENT_TYPE_TO_CAMEL:
            data["type"] = _CONTENT_TYPE_TO_CAMEL[block_type]
    return data


def to_wire_event(event: Any) -> Any:
    payload = _to_plain(event)
    if isinstance(payload, dict):
        if "partial" in payload:
            payload["partial"] = to_wire_message(payload["partial"])
        if "message" in payload:
            payload["message"] = to_wire_message(payload["message"])
        if "error" in payload:
            payload["error"] = to_wire_message(payload["error"])
        if "tool_call" in payload:
            payload["tool_call"] = to_wire_content_block(payload["tool_call"])
    payload = to_camel_dict(payload)
    if isinstance(payload, dict):
        reason = payload.get("reason")
        if reason in _STOP_REASON_TO_CAMEL:
            payload["reason"] = _STOP_REASON_TO_CAMEL[reason]
    return payload

"""Tool argument validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError, create_model

from .types import Tool, ToolCall


def _build_model_from_schema(tool: Tool) -> type[BaseModel]:
    properties = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))
    fields: Dict[str, tuple[type, Any]] = {}

    for name, schema in properties.items():
        field_type = _schema_to_type(schema)
        default = ... if name in required else None
        fields[name] = (field_type, default)

    return create_model(f"ToolArgs_{tool.name}", **fields)  # type: ignore[arg-type]


def _schema_to_type(schema: Dict[str, Any]) -> type:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "number":
        return float
    if schema_type == "integer":
        return int
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _normalize_arguments(schema: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(args)
    properties = schema.get("properties", {})
    for prop in properties.keys():
        if prop in normalized:
            continue
        if "_" in prop:
            camel = _snake_to_camel(prop)
            if camel in normalized:
                normalized[prop] = normalized[camel]
    return normalized


def validate_tool_arguments(tools: List[Tool], tool_call: ToolCall) -> Dict[str, Any]:
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f"Tool {tool_call.name} not found")

    model = _build_model_from_schema(tool)
    try:
        normalized_args = _normalize_arguments(tool.parameters, tool_call.arguments)
        validated = model(**normalized_args)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return validated.model_dump()

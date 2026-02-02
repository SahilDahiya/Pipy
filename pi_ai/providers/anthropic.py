"""Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..auth import get_env_api_key
from ..models import calculate_cost
from ..streaming import AssistantMessageEventStream
from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)

ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AnthropicOptions(StreamOptions):
    thinking_enabled: Optional[bool] = None
    thinking_budget_tokens: Optional[int] = None
    tool_choice: Optional[str | Dict[str, str]] = None


async def _maybe_abort(signal: Optional[asyncio.Event]) -> None:
    if signal and signal.is_set():
        raise RuntimeError("Request was aborted")


def stream_anthropic(
    model: Model,
    context: Context,
    options: Optional[AnthropicOptions] = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()

    async def run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
        )

        try:
            api_key = options.api_key if options else None
            if not api_key:
                api_key = get_env_api_key(model.provider)
            if not api_key:
                raise RuntimeError(
                    f"No API key for provider: {model.provider}. Set an env var or pass api_key."
                )

            params = _build_params(model, context, options)
            if options and options.on_payload:
                options.on_payload(params)

            headers = _build_headers(api_key, options.headers if options else None)
            url = _build_url(model.base_url)

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=params, headers=headers) as response:
                    response.raise_for_status()
                    stream.push({"type": "start", "partial": output})

                    current_blocks: Dict[int, TextContent | ThinkingContent | ToolCall] = {}
                    partial_json: Dict[int, str] = {}

                    async for event_name, data in _iter_sse(response):
                        await _maybe_abort(options.signal if options else None)
                        if not data:
                            continue

                        event_type = event_name or data.get("type")
                        if event_type == "message_start":
                            usage = (data.get("message") or {}).get("usage") or {}
                            output.usage.input = usage.get("input_tokens", 0) or 0
                            output.usage.output = usage.get("output_tokens", 0) or 0
                            output.usage.cache_read = usage.get("cache_read_input_tokens", 0) or 0
                            output.usage.cache_write = usage.get("cache_creation_input_tokens", 0) or 0
                            output.usage.total_tokens = (
                                output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write
                            )
                            calculate_cost(model, output.usage)
                        elif event_type == "content_block_start":
                            block = data.get("content_block") or {}
                            index = int(data.get("index", len(output.content)))
                            block_type = block.get("type")
                            if block_type == "text":
                                content = TextContent(text="")
                                output.content.append(content)
                                current_blocks[index] = content
                                stream.push({"type": "text_start", "content_index": index, "partial": output})
                            elif block_type == "thinking":
                                content = ThinkingContent(thinking="", thinking_signature="")
                                output.content.append(content)
                                current_blocks[index] = content
                                stream.push(
                                    {"type": "thinking_start", "content_index": index, "partial": output}
                                )
                            elif block_type == "tool_use":
                                content = ToolCall(
                                    id=block.get("id", ""),
                                    name=block.get("name", ""),
                                    arguments=block.get("input") or {},
                                )
                                output.content.append(content)
                                current_blocks[index] = content
                                partial_json[index] = ""
                                stream.push(
                                    {"type": "toolcall_start", "content_index": index, "partial": output}
                                )
                        elif event_type == "content_block_delta":
                            index = int(data.get("index", 0))
                            delta = data.get("delta") or {}
                            delta_type = delta.get("type")
                            block = current_blocks.get(index)
                            if delta_type == "text_delta" and isinstance(block, TextContent):
                                text_delta = delta.get("text", "")
                                block.text += text_delta
                                stream.push(
                                    {
                                        "type": "text_delta",
                                        "content_index": index,
                                        "delta": text_delta,
                                        "partial": output,
                                    }
                                )
                            elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
                                thinking_delta = delta.get("thinking", "")
                                block.thinking += thinking_delta
                                stream.push(
                                    {
                                        "type": "thinking_delta",
                                        "content_index": index,
                                        "delta": thinking_delta,
                                        "partial": output,
                                    }
                                )
                            elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
                                partial_json[index] = partial_json.get(index, "") + delta.get(
                                    "partial_json", ""
                                )
                                block.arguments = _parse_streaming_json(partial_json[index])
                                stream.push(
                                    {
                                        "type": "toolcall_delta",
                                        "content_index": index,
                                        "delta": delta.get("partial_json", ""),
                                        "partial": output,
                                    }
                                )
                            elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
                                block.thinking_signature = (block.thinking_signature or "") + delta.get(
                                    "signature", ""
                                )
                        elif event_type == "content_block_stop":
                            index = int(data.get("index", 0))
                            block = current_blocks.get(index)
                            if isinstance(block, TextContent):
                                stream.push(
                                    {
                                        "type": "text_end",
                                        "content_index": index,
                                        "content": block.text,
                                        "partial": output,
                                    }
                                )
                            elif isinstance(block, ThinkingContent):
                                stream.push(
                                    {
                                        "type": "thinking_end",
                                        "content_index": index,
                                        "content": block.thinking,
                                        "partial": output,
                                    }
                                )
                            elif isinstance(block, ToolCall):
                                stream.push(
                                    {
                                        "type": "toolcall_end",
                                        "content_index": index,
                                        "tool_call": block,
                                        "partial": output,
                                    }
                                )
                        elif event_type == "message_delta":
                            delta = data.get("delta") or {}
                            stop_reason = delta.get("stop_reason")
                            if stop_reason:
                                output.stop_reason = _map_stop_reason(stop_reason)
                            usage = data.get("usage") or {}
                            if usage:
                                if usage.get("input_tokens") is not None:
                                    output.usage.input = usage.get("input_tokens")
                                if usage.get("output_tokens") is not None:
                                    output.usage.output = usage.get("output_tokens")
                                if usage.get("cache_read_input_tokens") is not None:
                                    output.usage.cache_read = usage.get("cache_read_input_tokens")
                                if usage.get("cache_creation_input_tokens") is not None:
                                    output.usage.cache_write = usage.get("cache_creation_input_tokens")
                                output.usage.total_tokens = (
                                    output.usage.input
                                    + output.usage.output
                                    + output.usage.cache_read
                                    + output.usage.cache_write
                                )
                                calculate_cost(model, output.usage)

            await _maybe_abort(options.signal if options else None)
            stream.push({"type": "done", "reason": output.stop_reason, "message": output})
            stream.end(output)
        except Exception as error:
            output.stop_reason = "aborted" if options and options.signal and options.signal.is_set() else "error"
            output.error_message = str(error)
            stream.push({"type": "error", "reason": output.stop_reason, "error": output})
            stream.end(output)

    asyncio.create_task(run())
    return stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    api_key = options.api_key if options else None
    if not api_key:
        api_key = get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    return stream_anthropic(
        model,
        context,
        AnthropicOptions(
            api_key=api_key,
            headers=options.headers if options else None,
            max_tokens=options.max_tokens if options else None,
            temperature=options.temperature if options else None,
            signal=options.signal if options else None,
            session_id=options.session_id if options else None,
            on_payload=options.on_payload if options else None,
            thinking_enabled=bool(options.reasoning) if options else False,
        ),
    )


def _build_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _build_headers(api_key: str, extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _build_params(model: Model, context: Context, options: Optional[AnthropicOptions]) -> Dict[str, Any]:
    messages = _convert_messages(context.messages, model)
    params: Dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "max_tokens": options.max_tokens if options and options.max_tokens else (model.max_tokens or 1024),
    }

    if context.system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": context.system_prompt,
            }
        ]

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools)

    if options and options.thinking_enabled and model.reasoning:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": options.thinking_budget_tokens or 1024,
        }

    if options and options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


def _convert_messages(messages: List[Message], model: Model) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": msg.content})
            else:
                blocks: List[Dict[str, Any]] = []
                for block in msg.content:
                    if block.type == "text":
                        if block.text.strip():
                            blocks.append({"type": "text", "text": block.text})
                    elif block.type == "image" and "image" in model.input:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type,
                                    "data": block.data,
                                },
                            }
                        )
                if blocks:
                    params.append({"role": "user", "content": blocks})
        elif msg.role == "assistant":
            blocks: List[Dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text" and block.text.strip():
                    blocks.append({"type": "text", "text": block.text})
                elif block.type == "thinking" and block.thinking.strip():
                    if block.thinking_signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.thinking_signature,
                            }
                        )
                    else:
                        blocks.append({"type": "text", "text": block.thinking})
                elif block.type == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})
        elif msg.role == "toolResult":
            blocks: List[Dict[str, Any]] = []
            text_parts = [b.text for b in msg.content if b.type == "text"]
            if text_parts:
                blocks.append({"type": "text", "text": "\n".join(text_parts)})
            if any(b.type == "image" for b in msg.content) and "image" in model.input:
                for block in msg.content:
                    if block.type == "image":
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type,
                                    "data": block.data,
                                },
                            }
                        )
            if not blocks:
                blocks = [{"type": "text", "text": "(see attached image)"}]
            params.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": blocks,
                            "is_error": msg.is_error,
                        }
                    ],
                }
            )

    return params


def _convert_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        schema = tool.parameters
        converted.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            }
        )
    return converted


def _map_stop_reason(reason: str) -> StopReason:
    if reason == "end_turn":
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "toolUse"
    if reason in {"refusal", "sensitive"}:
        return "error"
    return "stop"


def _parse_streaming_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    for idx in range(len(raw), 0, -1):
        try:
            parsed, _ = decoder.raw_decode(raw[:idx])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


async def _iter_sse(response: httpx.Response):
    event_name: Optional[str] = None
    data_lines: List[str] = []

    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines = []
                if data_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    payload = {}
                yield event_name, payload
                event_name = None
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    if data_lines:
        data_str = "\n".join(data_lines)
        if data_str != "[DONE]":
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                payload = {}
            yield event_name, payload

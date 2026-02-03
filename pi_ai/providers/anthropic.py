"""Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..auth import get_env_api_key
from ..models import calculate_cost
from ..streaming import AssistantMessageEventStream
from ..transform import transform_messages
from ..types import (
    AssistantMessage,
    CacheRetention,
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
from ..utils.sanitize_unicode import sanitize_surrogates
from .simple_options import adjust_max_tokens_for_thinking, build_base_options

ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_VERSION = "2.1.2"
CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]
_CLAUDE_TOOL_LOOKUP = {name.lower(): name for name in CLAUDE_CODE_TOOLS}


def _to_claude_code_name(name: str) -> str:
    return _CLAUDE_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: Optional[List[Tool]] = None) -> str:
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


@dataclass
class AnthropicOptions(StreamOptions):
    thinking_enabled: Optional[bool] = None
    thinking_budget_tokens: Optional[int] = None
    interleaved_thinking: Optional[bool] = None
    tool_choice: Optional[str | Dict[str, str]] = None


async def _maybe_abort(signal: Optional[asyncio.Event]) -> None:
    if signal and signal.is_set():
        raise RuntimeError("Request was aborted")


def _is_oauth_token(api_key: str) -> bool:
    return "sk-ant-oat" in api_key


def _resolve_cache_retention(cache_retention: Optional[CacheRetention]) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if os.getenv("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_cache_control(
    base_url: str, cache_retention: Optional[CacheRetention]
) -> tuple[CacheRetention, Optional[Dict[str, str]]]:
    retention = _resolve_cache_retention(cache_retention)
    if retention == "none":
        return retention, None
    ttl = "1h" if retention == "long" and "api.anthropic.com" in base_url else None
    cache_control: Dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return retention, cache_control


def _merge_headers(*sources: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for source in sources:
        if source:
            merged.update(source)
    return merged


def _convert_content_blocks(
    content: List[ImageContent | TextContent],
) -> str | List[Dict[str, Any]]:
    has_images = any(block.type == "image" for block in content)
    if not has_images:
        return sanitize_surrogates(
            "\n".join(block.text for block in content if block.type == "text")
        )

    blocks: List[Dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
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

    if not any(block.get("type") == "text" for block in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


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

            is_oauth = _is_oauth_token(api_key)
            _, cache_control = _get_cache_control(
                model.base_url, options.cache_retention if options else None
            )
            params = _build_params(model, context, is_oauth, cache_control, options)
            if options and options.on_payload:
                options.on_payload(params)

            headers = _build_headers(
                api_key,
                model.headers,
                options.headers if options else None,
                is_oauth,
                options.interleaved_thinking if options else True,
            )
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
                                tool_name = block.get("name", "")
                                if is_oauth:
                                    tool_name = _from_claude_code_name(tool_name, context.tools)
                                content = ToolCall(
                                    id=block.get("id", ""),
                                    name=tool_name,
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

    base = build_base_options(model, options, api_key)

    if not options or not options.reasoning:
        return stream_anthropic(
            model,
            context,
            AnthropicOptions(
                **base.__dict__,
                thinking_enabled=False,
            ),
        )

    base_max_tokens = base.max_tokens or model.max_tokens or 1024
    model_max_tokens = model.max_tokens or base_max_tokens
    adjusted_max, thinking_budget = adjust_max_tokens_for_thinking(
        base_max_tokens,
        model_max_tokens,
        options.reasoning,
        options.thinking_budgets,
    )

    return stream_anthropic(
        model,
        context,
        AnthropicOptions(
            **base.__dict__,
            max_tokens=adjusted_max,
            thinking_enabled=True,
            thinking_budget_tokens=thinking_budget,
        ),
    )


def _build_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _build_headers(
    api_key: str,
    model_headers: Optional[Dict[str, str]],
    extra: Optional[Dict[str, str]],
    is_oauth: bool,
    interleaved_thinking: Optional[bool],
) -> Dict[str, str]:
    beta_features = ["fine-grained-tool-streaming-2025-05-14"]
    if interleaved_thinking is not False:
        beta_features.append("interleaved-thinking-2025-05-14")

    if is_oauth:
        beta_header = f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}"
        base_headers = {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-beta": beta_header,
            "anthropic-version": ANTHROPIC_VERSION,
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
            "x-app": "cli",
        }
    else:
        base_headers = {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-beta": ",".join(beta_features),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "x-api-key": api_key,
        }

    return _merge_headers(base_headers, model_headers, extra)


def _build_params(
    model: Model,
    context: Context,
    is_oauth: bool,
    cache_control: Optional[Dict[str, str]],
    options: Optional[AnthropicOptions],
) -> Dict[str, Any]:
    messages = _convert_messages(context.messages, model, context.tools, is_oauth, cache_control)
    params: Dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "max_tokens": options.max_tokens if options and options.max_tokens else (model.max_tokens or 1024),
    }

    if is_oauth:
        system_blocks = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
        ]
        if context.system_prompt:
            system_blocks.append(
                {"type": "text", "text": sanitize_surrogates(context.system_prompt)}
            )
        if cache_control:
            for block in system_blocks:
                block["cache_control"] = cache_control
        params["system"] = system_blocks
    elif context.system_prompt:
        system_block: Dict[str, Any] = {
            "type": "text",
            "text": sanitize_surrogates(context.system_prompt),
        }
        if cache_control:
            system_block["cache_control"] = cache_control
        params["system"] = [system_block]

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools, is_oauth)

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


def _normalize_tool_call_id(tool_id: str, _model: Model, _source: AssistantMessage) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tool_id)[:64]


def _convert_messages(
    messages: List[Message],
    model: Model,
    tools: Optional[List[Tool]],
    is_oauth: bool,
    cache_control: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    transformed_messages = transform_messages(messages, model, _normalize_tool_call_id)

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]
        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                blocks: List[Dict[str, Any]] = []
                for block in msg.content:
                    if block.type == "text":
                        if block.text.strip():
                            blocks.append(
                                {"type": "text", "text": sanitize_surrogates(block.text)}
                            )
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
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking" and block.thinking.strip():
                    if block.thinking_signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": block.thinking_signature,
                            }
                        )
                    else:
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                elif block.type == "tool_call":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": _to_claude_code_name(block.name) if is_oauth else block.name,
                            "input": block.arguments or {},
                        }
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})
        elif msg.role == "tool_result":
            tool_results: List[Dict[str, Any]] = []
            while i < len(transformed_messages) and transformed_messages[i].role == "tool_result":
                tool_msg = transformed_messages[i]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_msg.tool_call_id,
                        "content": _convert_content_blocks(tool_msg.content),
                        "is_error": tool_msg.is_error,
                    }
                )
                i += 1
            params.append({"role": "user", "content": tool_results})
            continue

        i += 1

    if cache_control and params:
        last_message = params[-1]
        if last_message.get("role") == "user":
            content = last_message.get("content")
            if isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict) and last_block.get("type") in {
                    "text",
                    "image",
                    "tool_result",
                }:
                    last_block["cache_control"] = cache_control

    return params


def _convert_tools(tools: List[Tool], is_oauth: bool) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        schema = tool.parameters
        converted.append(
            {
                "name": _to_claude_code_name(tool.name) if is_oauth else tool.name,
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
        return "tool_use"
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

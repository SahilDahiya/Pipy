"""OpenAI Completions API provider."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import httpx

from ..auth import get_env_api_key
from ..models import calculate_cost, supports_xhigh
from ..streaming import AssistantMessageEventStream
from ..transform import transform_messages
from ..utils.sanitize_unicode import sanitize_surrogates
from ..types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
)


def _normalize_mistral_tool_id(tool_id: str) -> str:
    normalized = "".join(ch for ch in tool_id if ch.isalnum())
    if len(normalized) < 9:
        padding = "ABCDEFGHI"
        normalized = normalized + padding[: 9 - len(normalized)]
    elif len(normalized) > 9:
        normalized = normalized[:9]
    return normalized


def _has_tool_history(messages: List[Message]) -> bool:
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant":
            if any(block.type == "toolCall" for block in msg.content):
                return True
    return False


@dataclass
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: Optional[str | Dict[str, Dict[str, str]]] = None
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high", "xhigh"]] = None


async def _maybe_abort(signal: Optional[asyncio.Event]) -> None:
    if signal and signal.is_set():
        raise RuntimeError("Request was aborted")


def stream_openai_completions(
    model: Model,
    context: Context,
    options: Optional[OpenAICompletionsOptions] = None,
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

            headers = _build_headers(model, context, api_key, options.headers if options else None)
            url = _build_url(model.base_url)

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=params, headers=headers) as response:
                    response.raise_for_status()
                    stream.push({"type": "start", "partial": output})

                    current_block: TextContent | ThinkingContent | ToolCall | None = None
                    current_tool_args: str = ""

                    def block_index() -> int:
                        return len(output.content) - 1

                    def finish_block(block: TextContent | ThinkingContent | ToolCall | None) -> None:
                        if not block:
                            return
                        if block.type == "text":
                            stream.push(
                                {
                                    "type": "text_end",
                                    "content_index": block_index(),
                                    "content": block.text,
                                    "partial": output,
                                }
                            )
                        elif block.type == "thinking":
                            stream.push(
                                {
                                    "type": "thinking_end",
                                    "content_index": block_index(),
                                    "content": block.thinking,
                                    "partial": output,
                                }
                            )
                        elif block.type == "toolCall":
                            stream.push(
                                {
                                    "type": "toolcall_end",
                                    "content_index": block_index(),
                                    "tool_call": block,
                                    "partial": output,
                                }
                            )

                    async for line in response.aiter_lines():
                        await _maybe_abort(options.signal if options else None)
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break

                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage:
                            cached_tokens = (
                                usage.get("prompt_tokens_details", {}) or {}
                            ).get("cached_tokens", 0)
                            reasoning_tokens = (
                                usage.get("completion_tokens_details", {}) or {}
                            ).get("reasoning_tokens", 0)
                            input_tokens = (usage.get("prompt_tokens") or 0) - cached_tokens
                            output_tokens = (usage.get("completion_tokens") or 0) + reasoning_tokens
                            output.usage.input = max(input_tokens, 0)
                            output.usage.output = max(output_tokens, 0)
                            output.usage.cache_read = max(cached_tokens or 0, 0)
                            output.usage.cache_write = 0
                            output.usage.total_tokens = (
                                output.usage.input + output.usage.output + output.usage.cache_read
                            )
                            calculate_cost(model, output.usage)

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason"):
                            output.stop_reason = _map_stop_reason(choice["finish_reason"])

                        delta = choice.get("delta") or {}
                        content_delta = delta.get("content")
                        if content_delta:
                            if not current_block or current_block.type != "text":
                                finish_block(current_block)
                                current_block = TextContent(text=content_delta)
                                output.content.append(current_block)
                                stream.push(
                                    {
                                        "type": "text_start",
                                        "content_index": block_index(),
                                        "partial": output,
                                    }
                                )
                            else:
                                current_block.text += content_delta
                            stream.push(
                                {
                                    "type": "text_delta",
                                    "content_index": block_index(),
                                    "delta": content_delta,
                                    "partial": output,
                                }
                            )

                        reasoning_field = None
                        reasoning_delta = None
                        for field in ("reasoning_content", "reasoning", "reasoning_text"):
                            value = delta.get(field)
                            if value:
                                reasoning_field = field
                                reasoning_delta = value
                                break
                        if reasoning_field and reasoning_delta is not None:
                            if not current_block or current_block.type != "thinking":
                                finish_block(current_block)
                                current_block = ThinkingContent(
                                    thinking=reasoning_delta,
                                    thinking_signature=reasoning_field,
                                )
                                output.content.append(current_block)
                                stream.push(
                                    {
                                        "type": "thinking_start",
                                        "content_index": block_index(),
                                        "partial": output,
                                    }
                                )
                            else:
                                current_block.thinking += reasoning_delta
                            stream.push(
                                {
                                    "type": "thinking_delta",
                                    "content_index": block_index(),
                                    "delta": reasoning_delta,
                                    "partial": output,
                                }
                            )

                        tool_calls = delta.get("tool_calls") or []
                        for tool_call in tool_calls:
                            tool_id = tool_call.get("id") or ""
                            tool_name = (tool_call.get("function") or {}).get("name") or ""
                            if (
                                not current_block
                                or current_block.type != "toolCall"
                                or (tool_id and current_block.id != tool_id)
                            ):
                                finish_block(current_block)
                                current_block = ToolCall(id=tool_id, name=tool_name, arguments={})
                                current_tool_args = ""
                                output.content.append(current_block)
                                stream.push(
                                    {
                                        "type": "toolcall_start",
                                        "content_index": block_index(),
                                        "partial": output,
                                    }
                                )

                            if current_block.type == "toolCall":
                                if tool_id:
                                    current_block.id = tool_id
                                if tool_name:
                                    current_block.name = tool_name
                                args_delta = (tool_call.get("function") or {}).get("arguments") or ""
                                if args_delta:
                                    current_tool_args += args_delta
                                    current_block.arguments = _parse_streaming_json(current_tool_args)
                                stream.push(
                                    {
                                        "type": "toolcall_delta",
                                        "content_index": block_index(),
                                        "delta": args_delta,
                                        "partial": output,
                                    }
                                )

                        reasoning_details = delta.get("reasoning_details") or []
                        if isinstance(reasoning_details, list):
                            for detail in reasoning_details:
                                if (
                                    isinstance(detail, dict)
                                    and detail.get("type") == "reasoning.encrypted"
                                    and detail.get("id")
                                    and detail.get("data")
                                ):
                                    for block in output.content:
                                        if block.type == "toolCall" and block.id == detail["id"]:
                                            block.thought_signature = json.dumps(detail)

                    finish_block(current_block)

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


def stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    api_key = options.api_key if options else None
    if not api_key:
        api_key = get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    reasoning = options.reasoning if options else None
    if reasoning == "xhigh" and not supports_xhigh(model):
        reasoning = "high"

    return stream_openai_completions(
        model,
        context,
        OpenAICompletionsOptions(
            api_key=api_key,
            headers=options.headers if options else None,
            max_tokens=options.max_tokens if options else None,
            temperature=options.temperature if options else None,
            signal=options.signal if options else None,
            session_id=options.session_id if options else None,
            on_payload=options.on_payload if options else None,
            reasoning_effort=reasoning,
            tool_choice=options.tool_choice if options else None,
        ),
    )


def _build_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _build_headers(
    model: Model,
    context: Context,
    api_key: str,
    options_headers: Optional[Dict[str, str]],
) -> Dict[str, str]:
    headers: Dict[str, str] = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    headers.update(model.headers)

    if model.provider == "github-copilot":
        messages = context.messages or []
        last_message = messages[-1] if messages else None
        is_agent_call = bool(last_message and last_message.role != "user")
        headers["X-Initiator"] = "agent" if is_agent_call else "user"
        headers["Openai-Intent"] = "conversation-edits"

        has_images = False
        for msg in messages:
            if msg.role == "user" and isinstance(msg.content, list):
                has_images = any(block.type == "image" for block in msg.content)
            if msg.role == "toolResult" and isinstance(msg.content, list):
                has_images = any(block.type == "image" for block in msg.content)
            if has_images:
                break
        if has_images:
            headers["Copilot-Vision-Request"] = "true"

    if options_headers:
        headers.update(options_headers)

    return headers


def _build_params(
    model: Model,
    context: Context,
    options: Optional[OpenAICompletionsOptions],
) -> Dict[str, Any]:
    compat = _get_compat(model)
    messages = _convert_messages(model, context, compat)
    reasoning_effort = None
    if options and options.reasoning_effort:
        reasoning_effort = options.reasoning_effort
        if reasoning_effort == "xhigh" and not supports_xhigh(model):
            reasoning_effort = "high"

    params: Dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options and options.max_tokens:
        if compat.max_tokens_field == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
    elif _has_tool_history(context.messages):
        params["tools"] = []

    if options and options.tool_choice:
        params["tool_choice"] = options.tool_choice

    if compat.thinking_format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled" if reasoning_effort else "disabled"}
    elif compat.thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(reasoning_effort)
    elif reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort

    if "openrouter.ai" in model.base_url and compat.openrouter_routing:
        params["provider"] = compat.openrouter_routing

    if "ai-gateway.vercel.sh" in model.base_url and compat.vercel_gateway_routing:
        routing = compat.vercel_gateway_routing
        gateway_options: Dict[str, List[str]] = {}
        if routing.get("only"):
            gateway_options["only"] = routing["only"]
        if routing.get("order"):
            gateway_options["order"] = routing["order"]
        if gateway_options:
            params["providerOptions"] = {"gateway": gateway_options}

    return params


def _convert_messages(
    model: Model,
    context: Context,
    compat: OpenAICompletionsCompat,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    def normalize_tool_call_id(tool_id: str, _model: Model, _source: AssistantMessage) -> str:
        if compat.requires_mistral_tool_ids:
            return _normalize_mistral_tool_id(tool_id)
        if "|" in tool_id:
            tool_id = tool_id.split("|")[0]
            tool_id = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tool_id)[:40]
        if model.provider == "openai" and len(tool_id) > 40:
            return tool_id[:40]
        if model.provider == "github-copilot" and "claude" in model.id.lower():
            return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tool_id)[:64]
        return tool_id

    if context.system_prompt:
        role = "developer" if model.reasoning and compat.supports_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: Optional[str] = None

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})
        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                content: List[Dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                            }
                        )
                if "image" not in model.input:
                    content = [c for c in content if c.get("type") != "image_url"]
                if not content:
                    i += 1
                    continue
                params.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            text_blocks = [b for b in msg.content if b.type == "text" and b.text.strip()]
            if text_blocks:
                if model.provider == "github-copilot":
                    assistant_msg["content"] = sanitize_surrogates("".join(b.text for b in text_blocks))
                else:
                    assistant_msg["content"] = [
                        {"type": "text", "text": sanitize_surrogates(b.text)} for b in text_blocks
                    ]

            thinking_blocks = [b for b in msg.content if b.type == "thinking" and b.thinking.strip()]
            if thinking_blocks:
                if compat.requires_thinking_as_text:
                    thinking_text = sanitize_surrogates("\n\n".join(b.thinking for b in thinking_blocks))
                    if assistant_msg.get("content"):
                        content_list = assistant_msg["content"]
                        if isinstance(content_list, list):
                            content_list.insert(0, {"type": "text", "text": thinking_text})
                        else:
                            assistant_msg["content"] = [{"type": "text", "text": thinking_text}]
                    else:
                        assistant_msg["content"] = [{"type": "text", "text": thinking_text}]
                else:
                    signature = thinking_blocks[0].thinking_signature
                    if signature:
                        assistant_msg[signature] = sanitize_surrogates(
                            "\n".join(b.thinking for b in thinking_blocks)
                        )

            tool_calls = [b for b in msg.content if b.type == "toolCall"]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": normalize_tool_call_id(tc.id),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
                reasoning_details: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    if tc.thought_signature:
                        try:
                            parsed = json.loads(tc.thought_signature)
                        except Exception:
                            parsed = None
                        if parsed:
                            reasoning_details.append(parsed)
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            content = assistant_msg.get("content")
            has_content = content is not None and (
                (isinstance(content, str) and len(content) > 0)
                or (isinstance(content, list) and len(content) > 0)
            )
            if not has_content and not assistant_msg.get("tool_calls"):
                i += 1
                continue

            params.append(assistant_msg)
        elif msg.role == "toolResult":
            image_blocks: List[Dict[str, Any]] = []
            j = i

            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_msg_entry = transformed_messages[j]
                if tool_msg_entry.role != "toolResult":
                    break
                text_result = "\n".join(
                    block.text for block in tool_msg_entry.content if block.type == "text"
                )
                text_result = sanitize_surrogates(text_result)
                has_images = any(block.type == "image" for block in tool_msg_entry.content)
                has_text = len(text_result) > 0

                tool_msg: Dict[str, Any] = {
                    "role": "tool",
                    "content": text_result if has_text else "(see attached image)",
                    "tool_call_id": normalize_tool_call_id(tool_msg_entry.tool_call_id),
                }
                if compat.requires_tool_result_name and tool_msg_entry.tool_name:
                    tool_msg["name"] = tool_msg_entry.tool_name
                params.append(tool_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg_entry.content:
                        if block.type == "image":
                            image_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{block.mime_type};base64,{block.data}",
                                    },
                                }
                            )
                j += 1

            if image_blocks:
                if compat.requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Attached image(s) from tool result:"}] + image_blocks,
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"
            i = j
            continue

        last_role = msg.role
        i += 1

    return params


def _convert_tools(tools: List[Tool], compat: OpenAICompletionsCompat) -> List[Dict[str, Any]]:
    converted = []
    for tool in tools:
        function_spec = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            function_spec["strict"] = False
        converted.append({"type": "function", "function": function_spec})
    return converted


def _map_stop_reason(reason: Optional[str]) -> StopReason:
    if reason is None:
        return "stop"
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "length"
    if reason in {"function_call", "tool_calls"}:
        return "toolUse"
    if reason == "content_filter":
        return "error"
    return "error"


def _detect_compat(model: Model) -> OpenAICompletionsCompat:
    provider = model.provider
    base_url = model.base_url

    is_zai = provider == "zai" or "api.z.ai" in base_url
    is_non_standard = (
        provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or provider == "mistral"
        or "mistral.ai" in base_url
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or provider == "opencode"
        or "opencode.ai" in base_url
    )

    use_max_tokens = provider == "mistral" or "mistral.ai" in base_url or "chutes.ai" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_mistral = provider == "mistral" or "mistral.ai" in base_url

    return OpenAICompletionsCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=not is_grok and not is_zai,
        supports_usage_in_streaming=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=is_mistral,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=is_mistral,
        requires_mistral_tool_ids=is_mistral,
        thinking_format="zai" if is_zai else "openai",
        openrouter_routing={},
        vercel_gateway_routing={},
        supports_strict_mode=True,
    )


def _get_compat(model: Model) -> OpenAICompletionsCompat:
    detected = _detect_compat(model)
    if not model.compat:
        return detected

    return OpenAICompletionsCompat(
        supports_store=model.compat.supports_store if model.compat.supports_store is not None else detected.supports_store,
        supports_developer_role=model.compat.supports_developer_role
        if model.compat.supports_developer_role is not None
        else detected.supports_developer_role,
        supports_reasoning_effort=model.compat.supports_reasoning_effort
        if model.compat.supports_reasoning_effort is not None
        else detected.supports_reasoning_effort,
        supports_usage_in_streaming=model.compat.supports_usage_in_streaming
        if model.compat.supports_usage_in_streaming is not None
        else detected.supports_usage_in_streaming,
        max_tokens_field=model.compat.max_tokens_field if model.compat.max_tokens_field else detected.max_tokens_field,
        requires_tool_result_name=model.compat.requires_tool_result_name
        if model.compat.requires_tool_result_name is not None
        else detected.requires_tool_result_name,
        requires_assistant_after_tool_result=model.compat.requires_assistant_after_tool_result
        if model.compat.requires_assistant_after_tool_result is not None
        else detected.requires_assistant_after_tool_result,
        requires_thinking_as_text=model.compat.requires_thinking_as_text
        if model.compat.requires_thinking_as_text is not None
        else detected.requires_thinking_as_text,
        requires_mistral_tool_ids=model.compat.requires_mistral_tool_ids
        if model.compat.requires_mistral_tool_ids is not None
        else detected.requires_mistral_tool_ids,
        thinking_format=model.compat.thinking_format if model.compat.thinking_format else detected.thinking_format,
        openrouter_routing=model.compat.openrouter_routing or detected.openrouter_routing,
        vercel_gateway_routing=model.compat.vercel_gateway_routing or detected.vercel_gateway_routing,
        supports_strict_mode=model.compat.supports_strict_mode
        if model.compat.supports_strict_mode is not None
        else detected.supports_strict_mode,
    )


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

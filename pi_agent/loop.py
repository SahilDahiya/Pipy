"""Agent loop with tool execution and event emission."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Sequence

from pi_ai.providers import stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    ToolCall,
    ToolResultMessage,
    TextContent,
)
from pi_ai.validation import validate_tool_arguments
from pi_tools.base import ToolDefinition, ToolResult

from .events import AgentEventStream
from .types import AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, StreamFn


def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None,
) -> AgentEventStream:
    stream = AgentEventStream()

    async def run() -> None:
        new_messages: List[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages) + list(prompts),
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        for prompt in prompts:
            stream.push({"type": "message_start", "message": prompt})
            stream.push({"type": "message_end", "message": prompt})

        await _run_loop(current_context, new_messages, config, stream, stream_fn)

    import asyncio

    asyncio.create_task(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None,
) -> AgentEventStream:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = AgentEventStream()

    async def run() -> None:
        new_messages: List[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})

        await _run_loop(current_context, new_messages, config, stream, stream_fn)

    import asyncio

    asyncio.create_task(run())
    return stream


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_messages(messages: Optional[Sequence[AgentMessage]]) -> List[AgentMessage]:
    if not messages:
        return []
    return list(messages)


def _transform_accepts_signal(transform) -> bool:
    try:
        params = inspect.signature(transform).parameters
    except (TypeError, ValueError):
        return False
    for param in params.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return True
    return len(params) >= 2


async def _run_loop(
    current_context: AgentContext,
    new_messages: List[AgentMessage],
    config: AgentLoopConfig,
    stream: AgentEventStream,
    stream_fn: Optional[StreamFn],
) -> None:
    first_turn = True
    pending_messages = (
        _normalize_messages(await _maybe_await(config.get_steering_messages()))
        if config.get_steering_messages
        else []
    )

    while True:
        has_more_tool_calls = True
        steering_after_tools: Optional[List[AgentMessage]] = None

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push({"type": "turn_start"})
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    stream.push({"type": "message_start", "message": message})
                    stream.push({"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            message = await _stream_assistant_response(current_context, config, stream, stream_fn)
            new_messages.append(message)

            if message.stop_reason in {"error", "aborted"}:
                stream.push({"type": "turn_end", "message": message, "toolResults": []})
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)
                return

            tool_calls = [block for block in message.content if block.type == "toolCall"]
            has_more_tool_calls = bool(tool_calls)

            tool_results: List[ToolResultMessage] = []
            if has_more_tool_calls:
                execution = await _execute_tool_calls(
                    current_context.tools or [],
                    message,
                    stream,
                    config.signal,
                    config.get_steering_messages,
                )
                tool_results.extend(execution.tool_results)
                steering_after_tools = execution.steering_messages

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push({"type": "turn_end", "message": message, "toolResults": tool_results})

            if steering_after_tools:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending_messages = _normalize_messages(await _maybe_await(config.get_steering_messages()))

        if config.get_follow_up_messages:
            follow_up = _normalize_messages(await _maybe_await(config.get_follow_up_messages()))
            if follow_up:
                pending_messages = follow_up
                continue

        break

    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    stream: AgentEventStream,
    stream_fn: Optional[StreamFn],
) -> AssistantMessage:
    resolved_api_key = config.api_key
    if config.get_api_key is not None:
        maybe_key = config.get_api_key(config.model.provider)
        if hasattr(maybe_key, "__await__"):
            resolved_api_key = await maybe_key  # type: ignore[assignment]
        else:
            resolved_api_key = maybe_key  # type: ignore[assignment]

    messages = context.messages
    if config.transform_context:
        if _transform_accepts_signal(config.transform_context):
            transformed = config.transform_context(messages, config.signal)
        else:
            transformed = config.transform_context(messages)
        messages = await _maybe_await(transformed)

    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=[tool.to_tool() for tool in (context.tools or [])],
    )

    options = SimpleStreamOptions(
        api_key=resolved_api_key,
        headers=config.headers,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        session_id=config.session_id,
        on_payload=config.on_payload,
        reasoning=config.reasoning,
        thinking_budgets=config.thinking_budgets,
        signal=config.signal,
        max_retry_delay_ms=config.max_retry_delay_ms,
    )

    stream_function = stream_fn or config.stream_fn or stream_simple
    response = stream_function(config.model, llm_context, options)
    response = await _maybe_await(response)

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        event_type = event.get("type")
        if event_type == "start":
            partial_message = event["partial"]
            context.messages.append(partial_message)
            added_partial = True
            stream.push({"type": "message_start", "message": partial_message})
            continue

        if event_type in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial_message is not None:
                partial_message = event["partial"]
                context.messages[-1] = partial_message
                stream.push(
                    {
                        "type": "message_update",
                        "assistantMessageEvent": event,
                        "message": partial_message,
                    }
                )
            continue

        if event_type in {"done", "error"}:
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                stream.push({"type": "message_start", "message": final_message})
            stream.push({"type": "message_end", "message": final_message})
            return final_message

    return await response.result()


@dataclass
class ToolExecutionResult:
    tool_results: List[ToolResultMessage]
    steering_messages: Optional[List[AgentMessage]] = None


async def _execute_tool_calls(
    tools: List[ToolDefinition],
    assistant_message: AssistantMessage,
    stream: AgentEventStream,
    signal: Optional[object],
    get_steering_messages: Optional[
        Callable[[], Sequence[AgentMessage] | Awaitable[Sequence[AgentMessage]]]
    ],
) -> ToolExecutionResult:
    results: List[ToolResultMessage] = []
    tool_calls = [block for block in assistant_message.content if block.type == "toolCall"]
    steering_messages: Optional[List[AgentMessage]] = None

    for index, tool_call in enumerate(tool_calls):
        tool = next((t for t in tools if t.name == tool_call.name), None)

        stream.push(
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            }
        )

        is_error = False
        try:
            if tool is None:
                raise RuntimeError(f"Tool {tool_call.name} not found")

            validated_args = validate_tool_arguments([t.to_tool() for t in tools], tool_call)
            result = await tool.execute(
                tool_call.id,
                validated_args,
                signal,
                lambda partial: stream.push(
                    {
                        "type": "tool_execution_update",
                        "toolCallId": tool_call.id,
                        "toolName": tool_call.name,
                        "args": tool_call.arguments,
                        "partialResult": partial,
                    }
                ),
            )
        except Exception as exc:
            result = ToolResult(content=[TextContent(text=str(exc))], details={})
            is_error = True

        stream.push(
            {
                "type": "tool_execution_end",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "result": result,
                "isError": is_error,
            }
        )

        tool_result_message = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
        )
        results.append(tool_result_message)

        stream.push({"type": "message_start", "message": tool_result_message})
        stream.push({"type": "message_end", "message": tool_result_message})

        if get_steering_messages:
            steering = _normalize_messages(await _maybe_await(get_steering_messages()))
            if steering:
                steering_messages = steering
                for skipped in tool_calls[index + 1 :]:
                    results.append(_skip_tool_call(skipped, stream))
                break
    return ToolExecutionResult(tool_results=results, steering_messages=steering_messages)


def _skip_tool_call(tool_call: ToolCall, stream: AgentEventStream) -> ToolResultMessage:
    result = ToolResult(content=[TextContent(text="Skipped due to queued user message.")], details={})

    stream.push(
        {
            "type": "tool_execution_start",
            "toolCallId": tool_call.id,
            "toolName": tool_call.name,
            "args": tool_call.arguments,
        }
    )
    stream.push(
        {
            "type": "tool_execution_end",
            "toolCallId": tool_call.id,
            "toolName": tool_call.name,
            "result": result,
            "isError": True,
        }
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=True,
    )
    stream.push({"type": "message_start", "message": tool_result_message})
    stream.push({"type": "message_end", "message": tool_result_message})
    return tool_result_message

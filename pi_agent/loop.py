"""Agent loop with tool execution and event emission."""

from __future__ import annotations

from typing import List, Optional

from pi_ai.providers import stream_simple
from pi_ai.types import AssistantMessage, Context, SimpleStreamOptions, ToolResultMessage, TextContent
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


async def _run_loop(
    current_context: AgentContext,
    new_messages: List[AgentMessage],
    config: AgentLoopConfig,
    stream: AgentEventStream,
    stream_fn: Optional[StreamFn],
) -> None:
    while True:
        message = await _stream_assistant_response(current_context, config, stream, stream_fn)
        new_messages.append(message)

        if message.stop_reason in {"error", "aborted"}:
            stream.push({"type": "turn_end", "message": message, "toolResults": []})
            stream.push({"type": "agent_end", "messages": new_messages})
            stream.end(new_messages)
            return

        tool_calls = [block for block in message.content if block.type == "toolCall"]
        if not tool_calls:
            stream.push({"type": "turn_end", "message": message, "toolResults": []})
            break

        tool_results = await _execute_tool_calls(
            current_context.tools or [],
            message,
            stream,
            config.signal,
        )
        for result in tool_results:
            current_context.messages.append(result)
            new_messages.append(result)

        stream.push({"type": "turn_end", "message": message, "toolResults": tool_results})

    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    stream: AgentEventStream,
    stream_fn: Optional[StreamFn],
) -> AssistantMessage:
    messages = config.convert_to_llm(context.messages)
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=messages,
        tools=[tool.to_tool() for tool in (context.tools or [])],
    )

    options = SimpleStreamOptions(
        api_key=config.api_key,
        headers=config.headers,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        session_id=config.session_id,
        on_payload=config.on_payload,
        reasoning=config.reasoning,
        thinking_budgets=config.thinking_budgets,
    )

    stream_function = stream_fn or config.stream_fn or stream_simple
    response = stream_function(config.model, llm_context, options)

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


async def _execute_tool_calls(
    tools: List[ToolDefinition],
    assistant_message: AssistantMessage,
    stream: AgentEventStream,
    signal: Optional[object],
) -> List[ToolResultMessage]:
    results: List[ToolResultMessage] = []
    for tool_call in [block for block in assistant_message.content if block.type == "toolCall"]:
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

    return results

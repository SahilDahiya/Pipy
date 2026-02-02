"""Message normalization for cross-provider handoff."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from .types import AssistantMessage, Message, Model, TextContent, ToolCall, ToolResultMessage


def transform_messages(
    messages: List[Message],
    model: Model,
    normalize_tool_call_id: Optional[Callable[[str, Model, AssistantMessage], str]] = None,
) -> List[Message]:
    tool_call_id_map: Dict[str, str] = {}
    transformed: List[Message] = []

    for msg in messages:
        if msg.role == "user":
            transformed.append(msg)
            continue

        if msg.role == "toolResult":
            tool_call_id = msg.tool_call_id
            normalized = tool_call_id_map.get(tool_call_id)
            if normalized and normalized != tool_call_id:
                transformed.append(
                    ToolResultMessage(
                        tool_call_id=normalized,
                        tool_name=msg.tool_name,
                        content=msg.content,
                        details=msg.details,
                        is_error=msg.is_error,
                        timestamp=msg.timestamp,
                    )
                )
            else:
                transformed.append(msg)
            continue

        if msg.role == "assistant":
            assistant_msg: AssistantMessage = msg
            is_same_model = (
                assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
                and assistant_msg.model == model.id
            )

            normalized_content = []
            for block in assistant_msg.content:
                if block.type == "thinking":
                    if is_same_model and block.thinking_signature:
                        normalized_content.append(block)
                        continue
                    if not block.thinking or not block.thinking.strip():
                        continue
                    if is_same_model:
                        normalized_content.append(block)
                    else:
                        normalized_content.append(TextContent(text=block.thinking))
                    continue

                if block.type == "text":
                    if is_same_model:
                        normalized_content.append(block)
                    else:
                        normalized_content.append(TextContent(text=block.text))
                    continue

                if block.type == "toolCall":
                    tool_call = block
                    if not is_same_model and tool_call.thought_signature:
                        tool_call = ToolCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                    if not is_same_model and normalize_tool_call_id:
                        normalized_id = normalize_tool_call_id(tool_call.id, model, assistant_msg)
                        if normalized_id != tool_call.id:
                            tool_call_id_map[tool_call.id] = normalized_id
                            tool_call = ToolCall(
                                id=normalized_id,
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                            )
                    normalized_content.append(tool_call)
                    continue

            transformed.append(
                AssistantMessage(
                    role="assistant",
                    content=normalized_content,
                    api=assistant_msg.api,
                    provider=assistant_msg.provider,
                    model=assistant_msg.model,
                    usage=assistant_msg.usage,
                    stop_reason=assistant_msg.stop_reason,
                    error_message=assistant_msg.error_message,
                    timestamp=assistant_msg.timestamp,
                )
            )
            continue

        transformed.append(msg)

    # Insert synthetic tool results for orphaned calls
    result: List[Message] = []
    pending_tool_calls: List[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def _flush_pending() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        for tc in pending_tool_calls:
            if tc.id not in existing_tool_result_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=int(time.time() * 1000),
                    )
                )
        pending_tool_calls = []
        existing_tool_result_ids = set()

    for msg in transformed:
        if msg.role == "assistant":
            if pending_tool_calls:
                _flush_pending()

            if msg.stop_reason in {"error", "aborted"}:
                continue

            tool_calls = [block for block in msg.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(msg)
        elif msg.role == "toolResult":
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
        elif msg.role == "user":
            if pending_tool_calls:
                _flush_pending()
            result.append(msg)
        else:
            result.append(msg)

    return result

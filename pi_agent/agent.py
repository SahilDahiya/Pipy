"""Agent class wrapper for the pi-python loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Literal, Optional, Sequence

from pi_ai.types import ImageContent, Message, Model, TextContent, UserMessage
from pi_ai.providers import stream_simple
from pi_session.manager import SessionManager
from pi_tools.base import ToolDefinition

from .events import AgentEventStream
from .loop import agent_loop, agent_loop_continue
from .types import AgentContext, AgentEvent, AgentLoopConfig, AgentMessage


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    tools: List[ToolDefinition]
    messages: List[AgentMessage]
    thinking_level: Optional[str]
    is_streaming: bool
    stream_message: Optional[AgentMessage]
    pending_tool_calls: set[str]
    error: Optional[str] = None


def _get_role(message: AgentMessage) -> Optional[str]:
    if hasattr(message, "role"):
        return message.role  # type: ignore[attr-defined]
    if isinstance(message, dict):
        return message.get("role")
    return None


def _default_convert_to_llm(messages: List[AgentMessage]) -> List[Message]:
    filtered: List[Message] = []
    for message in messages:
        role = _get_role(message)
        if role in {"user", "assistant", "tool_result"}:
            filtered.append(message)  # type: ignore[list-item]
    return filtered


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str = "",
        tools: Optional[Sequence[ToolDefinition]] = None,
        thinking_level: Optional[str] = None,
        steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
        follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
        convert_to_llm: Optional[
            Callable[[List[AgentMessage]], List[Message] | Awaitable[List[Message]]]
        ] = None,
        transform_context: Optional[
            Callable[[List[AgentMessage], Optional[asyncio.Event]], List[AgentMessage] | Awaitable[List[AgentMessage]]]
        ] = None,
        get_steering_messages: Optional[
            Callable[[], Sequence[AgentMessage] | Awaitable[Sequence[AgentMessage]]]
        ] = None,
        get_follow_up_messages: Optional[
            Callable[[], Sequence[AgentMessage] | Awaitable[Sequence[AgentMessage]]]
        ] = None,
        stream_fn: Optional[Callable[[Model, Any, Any], Any | Awaitable[Any]]] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        get_api_key: Optional[Callable[[str], Awaitable[Optional[str]] | Optional[str]]] = None,
        headers: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_budgets: Optional[dict] = None,
        max_retry_delay_ms: Optional[int] = None,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            tools=list(tools or []),
            messages=[],
            thinking_level=thinking_level,
            is_streaming=False,
            stream_message=None,
            pending_tool_calls=set(),
        )
        self._listeners: set[Callable[[AgentEvent], None]] = set()
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        self._transform_context = transform_context
        self._get_steering_messages = get_steering_messages
        self._get_follow_up_messages = get_follow_up_messages
        self._stream_fn = stream_fn or stream_simple
        self._session_id = session_id
        self._session_manager = session_manager
        self._get_api_key = get_api_key
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._steering_queue: List[AgentMessage] = []
        self._follow_up_queue: List[AgentMessage] = []
        self._api_key = api_key
        self._headers = headers
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking_budgets = thinking_budgets
        self._max_retry_delay_ms = max_retry_delay_ms
        self._abort_event: Optional[asyncio.Event] = None
        self._running_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def thinking_budgets(self) -> Optional[dict]:
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: Optional[dict]) -> None:
        self._thinking_budgets = value

    @property
    def max_retry_delay_ms(self) -> Optional[int]:
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: Optional[int]) -> None:
        self._max_retry_delay_ms = value

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.add(fn)

        def unsubscribe() -> None:
            self._listeners.discard(fn)

        return unsubscribe

    def set_system_prompt(self, prompt: str) -> None:
        self._state.system_prompt = prompt

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: Optional[str]) -> None:
        self._state.thinking_level = level

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._steering_mode

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._follow_up_mode

    def set_tools(self, tools: Sequence[ToolDefinition]) -> None:
        self._state.tools = list(tools)

    def replace_messages(self, messages: Sequence[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages.append(message)

    def clear_messages(self) -> None:
        self._state.messages = []

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self.clear_all_queues()

    def abort(self) -> None:
        if self._abort_event:
            self._abort_event.set()

    async def wait_for_idle(self) -> None:
        if self._running_task:
            await self._running_task

    def send(
        self,
        input_value: str | AgentMessage | List[AgentMessage],
        images: Optional[List[ImageContent]] = None,
    ) -> AgentEventStream:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Use steer() or follow_up() to queue messages."
            )

        prompts: List[AgentMessage]
        if isinstance(input_value, list):
            prompts = input_value
        elif isinstance(input_value, str):
            content: List[TextContent | ImageContent] = [TextContent(text=input_value)]
            if images:
                content.extend(images)
            prompts = [UserMessage(content=content)]
        else:
            prompts = [input_value]

        return self._run_loop(prompts)

    def continue_session(self) -> AgentEventStream:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")
        if not self._state.messages:
            raise RuntimeError("No messages to continue from.")
        if self._state.messages[-1].role == "assistant":
            raise RuntimeError("Cannot continue from message role: assistant")

        return self._run_loop(None)

    def _run_loop(self, prompts: Optional[List[AgentMessage]]) -> AgentEventStream:
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning = None
        if self._state.thinking_level and self._state.thinking_level != "off":
            reasoning = self._state.thinking_level

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        config = AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            stream_fn=self._stream_fn,
            get_api_key=self._get_api_key,
            get_steering_messages=self._build_steering_reader(),
            get_follow_up_messages=self._build_follow_up_reader(),
            reasoning=reasoning,
            session_id=self._session_id,
            api_key=self._api_key,
            headers=self._headers,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            thinking_budgets=self._thinking_budgets,
            signal=self._abort_event,
            max_retry_delay_ms=self._max_retry_delay_ms,
        )

        if prompts is None:
            base_stream = agent_loop_continue(context, config, self._stream_fn)
        else:
            base_stream = agent_loop(prompts, context, config, self._stream_fn)

        output_stream = AgentEventStream()

        async def forward() -> None:
            try:
                async for event in base_stream:
                    self._handle_event(event)
                    output_stream.push(event)
            finally:
                self._state.is_streaming = False
                output_stream.end(self._state.messages)

        self._running_task = asyncio.create_task(forward())
        return output_stream

    def _build_steering_reader(self):
        if self._get_steering_messages:
            return self._get_steering_messages

        async def _reader() -> List[AgentMessage]:
            if self._steering_mode == "one-at-a-time":
                if self._steering_queue:
                    return [self._steering_queue.pop(0)]
                return []
            messages = list(self._steering_queue)
            self._steering_queue = []
            return messages

        return _reader

    def _build_follow_up_reader(self):
        if self._get_follow_up_messages:
            return self._get_follow_up_messages

        async def _reader() -> List[AgentMessage]:
            if self._follow_up_mode == "one-at-a-time":
                if self._follow_up_queue:
                    return [self._follow_up_queue.pop(0)]
                return []
            messages = list(self._follow_up_queue)
            self._follow_up_queue = []
            return messages

        return _reader

    def _handle_event(self, event: AgentEvent) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            self._state.stream_message = event["message"]
        elif event_type == "message_update":
            self._state.stream_message = event["message"]
        elif event_type == "message_end":
            self._state.stream_message = None
            self._state.messages.append(event["message"])
            if self._session_manager is not None:
                self._session_manager.append_message(event["message"])
        elif event_type == "tool_execution_start":
            self._state.pending_tool_calls.add(event["tool_call_id"])
        elif event_type == "tool_execution_end":
            self._state.pending_tool_calls.discard(event["tool_call_id"])
        elif event_type == "turn_end":
            message = event.get("message")
            if message is not None and getattr(message, "error_message", None):
                self._state.error = message.error_message
        elif event_type == "agent_end":
            self._state.is_streaming = False
            self._state.stream_message = None

        for listener in list(self._listeners):
            listener(event)

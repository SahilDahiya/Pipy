"""Unified streaming interface for provider responses."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Optional

from .types import AssistantMessage

AssistantMessageEvent = Dict[str, Any]


class AssistantMessageEventStream(AsyncIterator[AssistantMessageEvent]):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AssistantMessageEvent | None] = asyncio.Queue()
        self._done = False
        self._result: Optional[AssistantMessage] = None
        self._result_event = asyncio.Event()

    def push(self, event: AssistantMessageEvent) -> None:
        if self._done:
            return
        event_type = event.get("type")
        if event_type in {"done", "error"}:
            message = event.get("message") or event.get("error")
            if message is not None:
                self._set_result(message)
        self._queue.put_nowait(event)

    def end(self, result: Optional[AssistantMessage] = None) -> None:
        if result is not None:
            self._set_result(result)
        if not self._done:
            self._done = True
            self._queue.put_nowait(None)

    async def result(self) -> AssistantMessage:
        await self._result_event.wait()
        if self._result is None:
            raise RuntimeError("Stream finished without a result message.")
        return self._result

    def _set_result(self, message: AssistantMessage) -> None:
        if self._result is None:
            self._result = message
            self._result_event.set()

    def __aiter__(self) -> "AssistantMessageEventStream":
        return self

    async def __anext__(self) -> AssistantMessageEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

"""Agent event stream utilities."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from .types import AgentEvent


class AgentEventStream(AsyncIterator[AgentEvent]):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._done = False
        self._result: Optional[List[Any]] = None
        self._result_event = asyncio.Event()

    def push(self, event: AgentEvent) -> None:
        if self._done:
            return
        self._queue.put_nowait(event)

    def end(self, messages: Optional[List[Any]] = None) -> None:
        if messages is not None:
            self._result = messages
            self._result_event.set()
        if not self._done:
            self._done = True
            self._queue.put_nowait(None)

    async def result(self) -> List[Any]:
        await self._result_event.wait()
        if self._result is None:
            raise RuntimeError("Stream finished without result messages.")
        return self._result

    def __aiter__(self) -> "AgentEventStream":
        return self

    async def __anext__(self) -> AgentEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

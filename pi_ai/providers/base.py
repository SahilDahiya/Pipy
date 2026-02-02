"""Abstract provider interface."""

from __future__ import annotations

from typing import Callable, Protocol

from ..streaming import AssistantMessageEventStream
from ..types import Context, Model, StreamOptions


class StreamFunction(Protocol):
    def __call__(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

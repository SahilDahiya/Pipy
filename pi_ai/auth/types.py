"""OAuth type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class OAuthCredentials:
    access: str
    refresh: str
    expires: int
    account_id: Optional[str] = None


@dataclass
class OAuthLoginCallbacks:
    on_auth: Callable[[dict], None]
    on_prompt: Callable[[dict], Awaitable[str]]
    on_progress: Optional[Callable[[str], None]] = None
    on_manual_code_input: Optional[Callable[[], Awaitable[str]]] = None

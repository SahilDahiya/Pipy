"""OAuth provider registry and helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Protocol

from .anthropic import login_anthropic, refresh_anthropic_token
from .openai import login_openai_codex, refresh_openai_codex_token
from .types import OAuthCredentials, OAuthLoginCallbacks


class OAuthProviderInterface(Protocol):
    id: str
    name: str

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials: ...

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials: ...

    def get_api_key(self, credentials: OAuthCredentials) -> str: ...


@dataclass
class OAuthProvider:
    id: str
    name: str
    login: Callable[[OAuthLoginCallbacks], Awaitable[OAuthCredentials]]
    refresh_token: Callable[[OAuthCredentials], Awaitable[OAuthCredentials]]
    get_api_key: Callable[[OAuthCredentials], str]


async def _login_anthropic(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    return await login_anthropic(
        lambda url: callbacks.on_auth({"url": url}),
        lambda: callbacks.on_prompt({"message": "Paste the authorization code:"}),
    )


def _anthropic_api_key(credentials: OAuthCredentials) -> str:
    return credentials.access


async def _login_openai(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    return await login_openai_codex(
        on_auth=callbacks.on_auth,
        on_prompt=callbacks.on_prompt,
        on_progress=callbacks.on_progress,
        on_manual_code_input=callbacks.on_manual_code_input,
    )


def _openai_api_key(credentials: OAuthCredentials) -> str:
    return credentials.access


_OAUTH_PROVIDERS: Dict[str, OAuthProvider] = {
    "anthropic": OAuthProvider(
        id="anthropic",
        name="Anthropic (Claude Pro/Max)",
        login=_login_anthropic,
        refresh_token=refresh_anthropic_token,
        get_api_key=_anthropic_api_key,
    ),
    "openai-codex": OAuthProvider(
        id="openai-codex",
        name="OpenAI Codex (ChatGPT Plus/Pro)",
        login=_login_openai,
        refresh_token=refresh_openai_codex_token,
        get_api_key=_openai_api_key,
    ),
}


def get_oauth_provider(provider_id: str) -> Optional[OAuthProvider]:
    return _OAUTH_PROVIDERS.get(provider_id)


def get_oauth_providers() -> list[OAuthProvider]:
    return list(_OAUTH_PROVIDERS.values())


def register_oauth_provider(provider: OAuthProvider) -> None:
    _OAUTH_PROVIDERS[provider.id] = provider


async def get_oauth_api_key(
    provider_id: str,
    credentials: Dict[str, OAuthCredentials],
) -> Optional[dict]:
    provider = get_oauth_provider(provider_id)
    if not provider:
        raise ValueError(f"Unknown OAuth provider: {provider_id}")

    creds = credentials.get(provider_id)
    if creds is None:
        return None

    if creds.expires and creds.expires <= time.time() * 1000:
        creds = await provider.refresh_token(creds)

    return {"new_credentials": creds, "api_key": provider.get_api_key(creds)}

"""Anthropic OAuth flow (Claude Pro/Max)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from .pkce import generate_pkce

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"


@dataclass
class OAuthCredentials:
    access: str
    refresh: str
    expires: int


def _build_authorize_url(state: str, challenge: str) -> str:
    from urllib.parse import urlencode

    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def login_anthropic(
    on_auth_url: Callable[[str], None],
    on_prompt_code: Callable[[], Awaitable[str]],
) -> OAuthCredentials:
    verifier, challenge = generate_pkce()
    auth_url = _build_authorize_url(verifier, challenge)
    on_auth_url(auth_url)

    auth_code = await on_prompt_code()
    code, state = auth_code.split("#", 1) if "#" in auth_code else (auth_code, None)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
        )
    response.raise_for_status()
    data = response.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token response missing fields")

    expires_at = int(time.time() * 1000 + expires_in * 1000 - 5 * 60 * 1000)
    return OAuthCredentials(access=access, refresh=refresh, expires=expires_at)


async def refresh_anthropic_token(refresh_token: str) -> OAuthCredentials:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    response.raise_for_status()
    data = response.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token response missing fields")

    expires_at = int(time.time() * 1000 + expires_in * 1000 - 5 * 60 * 1000)
    return OAuthCredentials(access=access, refresh=refresh, expires=expires_at)

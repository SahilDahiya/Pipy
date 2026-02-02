"""OpenAI Codex OAuth flow."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Awaitable, Callable, Optional

import httpx

from .pkce import generate_pkce

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"

SUCCESS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authentication successful</title>
</head>
<body>
  <p>Authentication successful. Return to your terminal to continue.</p>
</body>
</html>"""


@dataclass
class OAuthCredentials:
    access: str
    refresh: str
    expires: int
    account_id: Optional[str] = None


def _create_state() -> str:
    return secrets.token_hex(16)


def _parse_authorization_input(value: str) -> tuple[Optional[str], Optional[str]]:
    text = value.strip()
    if not text:
        return None, None

    if "#" in text and "code=" not in text:
        parts = text.split("#", 1)
        return parts[0], parts[1] if len(parts) > 1 else None

    if "code=" in text:
        try:
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(text)
            params = parse_qs(parsed.query)
            return params.get("code", [None])[0], params.get("state", [None])[0]
        except Exception:
            return text, None

    return text, None


def _decode_jwt(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return None


def _get_account_id(access_token: str) -> Optional[str]:
    payload = _decode_jwt(access_token) or {}
    auth = payload.get(JWT_CLAIM_PATH, {})
    account_id = auth.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    return None


async def _exchange_authorization_code(code: str, verifier: str) -> OAuthCredentials:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    response.raise_for_status()
    data = response.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token response missing fields")
    account_id = _get_account_id(access)
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")
    return OAuthCredentials(
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000 + expires_in * 1000),
        account_id=account_id,
    )


async def refresh_openai_codex_token(refresh_token: str) -> OAuthCredentials:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    response.raise_for_status()
    data = response.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Token response missing fields")
    account_id = _get_account_id(access)
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")
    return OAuthCredentials(
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000 + expires_in * 1000),
        account_id=account_id,
    )


def _build_authorize_url(state: str, challenge: str, originator: str) -> str:
    from urllib.parse import urlencode

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class _OAuthHandler(BaseHTTPRequestHandler):
    code: Optional[str] = None
    state: Optional[str] = None

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse, parse_qs

        url = urlparse(self.path)
        if url.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(url.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]
        if not state or state != self.state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return
        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        _OAuthHandler.code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(SUCCESS_HTML.encode("utf-8"))


async def login_openai_codex(
    *,
    on_auth: Callable[[dict], None],
    on_prompt: Callable[[dict], Awaitable[str]],
    on_progress: Optional[Callable[[str], None]] = None,
    on_manual_code_input: Optional[Callable[[], Awaitable[str]]] = None,
    originator: str = "pi",
) -> OAuthCredentials:
    verifier, challenge = generate_pkce()
    state = _create_state()
    url = _build_authorize_url(state, challenge, originator)

    _OAuthHandler.state = state
    _OAuthHandler.code = None

    server = None
    server_task = None

    try:
        server = HTTPServer(("127.0.0.1", 1455), _OAuthHandler)

        def serve() -> None:
            server.handle_request()

        loop = asyncio.get_event_loop()
        server_task = loop.run_in_executor(None, serve)
    except OSError:
        if on_progress:
            on_progress("Failed to bind local callback server, falling back to manual paste.")

    on_auth({"url": url, "instructions": "A browser window should open. Complete login to finish."})
    if on_progress and server_task:
        on_progress("Waiting for OAuth callback...")

    code: Optional[str] = None
    manual_task = None
    if on_manual_code_input:
        manual_task = asyncio.create_task(on_manual_code_input())

    wait_set = set()
    if server_task is not None:
        wait_set.add(server_task)
    if manual_task is not None:
        wait_set.add(manual_task)
    if wait_set:
        done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    else:
        done = set()

    if _OAuthHandler.code:
        code = _OAuthHandler.code
    elif manual_task and manual_task in done:
        manual_input = manual_task.result()
        code, state_value = _parse_authorization_input(manual_input)
        if state_value and state_value != state:
            raise RuntimeError("State mismatch")

    if not code:
        prompt_value = await on_prompt({"message": "Paste the authorization code (or full redirect URL):"})
        code, state_value = _parse_authorization_input(prompt_value)
        if state_value and state_value != state:
            raise RuntimeError("State mismatch")

    if server:
        server.server_close()

    if not code:
        raise RuntimeError("Missing authorization code")

    return await _exchange_authorization_code(code, verifier)

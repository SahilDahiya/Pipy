"""Token persistence and refresh logic."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .env import get_env_api_key
from .oauth import get_oauth_api_key, get_oauth_provider
from .types import OAuthCredentials, OAuthLoginCallbacks


@dataclass
class ApiKeyCredential:
    type: str
    key: str


@dataclass
class OAuthCredential:
    type: str
    access: str
    refresh: str
    expires: int
    account_id: Optional[str] = None


AuthCredential = ApiKeyCredential | OAuthCredential


class AuthStorage:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: Dict[str, AuthCredential] = {}
        self._runtime_overrides: Dict[str, str] = {}
        self._fallback_resolver: Optional[Callable[[str], Optional[str]]] = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}
            return
        data: Dict[str, AuthCredential] = {}
        for provider, value in raw.items():
            if value.get("type") == "api_key":
                data[provider] = ApiKeyCredential(type="api_key", key=value.get("key", ""))
            elif value.get("type") == "oauth":
                data[provider] = OAuthCredential(
                    type="oauth",
                    access=value.get("access", ""),
                    refresh=value.get("refresh", ""),
                    expires=value.get("expires", 0),
                    account_id=value.get("account_id"),
                )
        self._data = data

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data: Dict[str, dict] = {}
        for provider, cred in self._data.items():
            data[provider] = asdict(cred)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(self._path, 0o600)

    def reload(self) -> None:
        self._load()

    def set_api_key(self, provider: str, key: str) -> None:
        self._data[provider] = ApiKeyCredential(type="api_key", key=key)
        self._save()

    def set_oauth(self, provider: str, credentials: OAuthCredentials) -> None:
        self._data[provider] = OAuthCredential(
            type="oauth",
            access=credentials.access,
            refresh=credentials.refresh,
            expires=credentials.expires,
            account_id=credentials.account_id,
        )
        self._save()

    def set_runtime_api_key(self, provider: str, key: str) -> None:
        self._runtime_overrides[provider] = key

    def remove_runtime_api_key(self, provider: str) -> None:
        self._runtime_overrides.pop(provider, None)

    def set_fallback_resolver(self, resolver: Callable[[str], Optional[str]]) -> None:
        self._fallback_resolver = resolver

    def get(self, provider: str) -> Optional[AuthCredential]:
        return self._data.get(provider)

    def remove(self, provider: str) -> None:
        self._data.pop(provider, None)
        self._save()

    def list_providers(self) -> list[str]:
        return list(self._data.keys())

    def list(self) -> list[str]:
        return self.list_providers()

    def has(self, provider: str) -> bool:
        return provider in self._data

    def has_auth(self, provider: str) -> bool:
        if provider in self._runtime_overrides:
            return True
        if provider in self._data:
            return True
        if get_env_api_key(provider):
            return True
        if self._fallback_resolver and self._fallback_resolver(provider):
            return True
        return False

    def get_all(self) -> Dict[str, AuthCredential]:
        return dict(self._data)

    async def login(self, provider_id: str, callbacks: OAuthLoginCallbacks) -> None:
        provider = get_oauth_provider(provider_id)
        if not provider:
            raise ValueError(f"Unknown OAuth provider: {provider_id}")
        credentials = await provider.login(callbacks)
        self.set_oauth(provider_id, credentials)

    def logout(self, provider_id: str) -> None:
        self.remove(provider_id)

    async def get_api_key(
        self,
        provider: str,
        refresh_fn: Optional[Callable[[OAuthCredentials], OAuthCredentials]] = None,
    ) -> Optional[str]:
        if provider in self._runtime_overrides:
            return self._runtime_overrides[provider]

        cred = self._data.get(provider)
        if isinstance(cred, ApiKeyCredential):
            return cred.key

        if isinstance(cred, OAuthCredential):
            if cred.expires and cred.expires > int(time.time() * 1000):
                return cred.access
            if refresh_fn:
                new_creds = refresh_fn(
                    OAuthCredentials(
                        access=cred.access,
                        refresh=cred.refresh,
                        expires=cred.expires,
                        account_id=cred.account_id,
                    )
                )
                self.set_oauth(provider, new_creds)
                return new_creds.access
            oauth_map = {
                key: OAuthCredentials(
                    access=value.access,
                    refresh=value.refresh,
                    expires=value.expires,
                    account_id=value.account_id,
                )
                for key, value in self._data.items()
                if isinstance(value, OAuthCredential)
            }
            result = await get_oauth_api_key(provider, oauth_map)
            if result:
                self.set_oauth(provider, result["new_credentials"])
                return result["api_key"]

        env_key = get_env_api_key(provider)
        if env_key:
            return env_key

        if self._fallback_resolver:
            fallback = self._fallback_resolver(provider)
            if fallback:
                return fallback

        return None

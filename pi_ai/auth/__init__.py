"""OAuth and API key storage helpers."""

from __future__ import annotations

import os
from typing import Dict, Optional

from .anthropic import OAuthCredentials as AnthropicOAuthCredentials
from .anthropic import login_anthropic, refresh_anthropic_token
from .openai import OAuthCredentials as OpenAIOAuthCredentials
from .openai import login_openai_codex, refresh_openai_codex_token
from .storage import AuthStorage

__all__ = [
    "anthropic",
    "openai",
    "storage",
    "AuthStorage",
    "AnthropicOAuthCredentials",
    "OpenAIOAuthCredentials",
    "get_env_api_key",
    "login_anthropic",
    "login_openai_codex",
    "refresh_anthropic_token",
    "refresh_openai_codex_token",
]

_ENV_KEY_BY_PROVIDER: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
}


def get_env_api_key(provider: str) -> Optional[str]:
    env_key = _ENV_KEY_BY_PROVIDER.get(provider)
    if not env_key:
        return None
    return os.getenv(env_key)

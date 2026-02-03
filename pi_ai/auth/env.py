"""Environment variable lookup for provider API keys."""

from __future__ import annotations

import os
from typing import Dict, Optional

_ENV_KEY_BY_PROVIDER: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
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

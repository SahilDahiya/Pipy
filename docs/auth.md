# OAuth & Auth Storage

read_when: you need subscription OAuth or want to manage stored credentials

## Auth storage

- `AuthStorage` reads/writes `auth.json` in the agent directory.
- Default location: `~/.pi/agent/auth.json` (override via `PI_CODING_AGENT_DIR`).
- Supports API keys (`type: api_key`) and OAuth tokens (`type: oauth`).

## OpenAI Codex OAuth

```python
from pi_ai.auth import login_openai_codex

creds = await login_openai_codex(
    on_auth=lambda payload: print(payload["url"]),
    on_prompt=lambda payload: input(payload["message"]),
)
```

## Anthropic OAuth

```python
from pi_ai.auth import login_anthropic

creds = await login_anthropic(
    on_auth_url=lambda url: print(url),
    on_prompt_code=lambda: input("Paste the code: "),
)
```

## Using tokens

Store credentials via `AuthStorage.set_oauth()` and pass `auth_storage` or
`auth_path` to `create_agent()` to auto-resolve API keys per request.

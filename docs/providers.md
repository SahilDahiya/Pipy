# Providers

read_when: you are adding a model, debugging provider payloads, or tuning reasoning settings

## OpenAI Completions (and compatibles)

- API: `openai-completions` via `pi_ai.providers.openai`.
- Default base URL: `https://api.openai.com/v1` (override with `Model.base_url`).
- `get_model()` returns built-in metadata for common OpenAI ids (e.g., `gpt-4o`, `gpt-4o-mini`).
- Supports tool calling, streaming usage, and reasoning effort where available.
- Use `OpenAICompletionsCompat` on the model to override URL-based compatibility detection.
- `SimpleStreamOptions.reasoning` maps to `reasoning_effort`. `xhigh` is clamped unless `model.supports_xhigh` is true.
- `SimpleStreamOptions.tool_choice` forwards to `tool_choice`.
- `session_id` is forwarded for provider-side caching when supported.

## Anthropic Messages

- API: `anthropic-messages` via `pi_ai.providers.anthropic`.
- Default base URL: `https://api.anthropic.com/v1`.
- Built-in model metadata includes `claude-sonnet-4-5` and the latest 3.5 models.
- Reasoning enables `thinking` blocks with optional token budgets.
- `PI_CACHE_RETENTION` (or `cache_retention`) controls ephemeral cache hints.
- OAuth tokens use Claude Code tool naming and headers for parity.

## Environment variables

`pi_ai.auth.get_env_api_key()` maps providers to env vars, including:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`
- `AI_GATEWAY_API_KEY` (Vercel AI Gateway)

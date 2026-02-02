from pi_ai.models import create_openai_model
from pi_ai.providers import openai as openai_provider
from pi_ai.types import Context, OpenAICompletionsCompat, Tool


def test_openai_tool_choice_forwarded():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    tools = [
        Tool(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ]
    ctx = Context(system_prompt=None, messages=[], tools=tools)
    options = openai_provider.OpenAICompletionsOptions(tool_choice="required")
    params = openai_provider._build_params(model, ctx, options)
    assert params["tool_choice"] == "required"
    assert params["tools"]


def test_openai_strict_omitted_when_disabled():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    compat = OpenAICompletionsCompat(supports_strict_mode=False)
    tools = [
        Tool(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ]
    converted = openai_provider._convert_tools(tools, compat)
    tool = converted[0]["function"]
    assert "strict" not in tool

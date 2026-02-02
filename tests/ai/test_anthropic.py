from pi_ai.providers import anthropic as anthropic_provider
from pi_ai.types import Context, Model, ModelCost, TextContent, Tool, ToolResultMessage, UserMessage


def test_oauth_tool_name_normalization():
    tools = [
        Tool(
            name="read",
            description="Read file",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ]
    converted = anthropic_provider._convert_tools(tools, is_oauth=True)
    assert converted[0]["name"] == "Read"


def test_group_consecutive_tool_results():
    tool_result_1 = ToolResultMessage(
        tool_call_id="tool-1",
        tool_name="read",
        content=[TextContent(text="one")],
        is_error=False,
    )
    tool_result_2 = ToolResultMessage(
        tool_call_id="tool-2",
        tool_name="read",
        content=[TextContent(text="two")],
        is_error=False,
    )

    model = Model(
        id="claude",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=1000,
    )

    converted = anthropic_provider._convert_messages(
        [tool_result_1, tool_result_2],
        model,
        tools=None,
        is_oauth=False,
        cache_control=None,
    )

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert len(converted[0]["content"]) == 2


def test_cache_retention_long_sets_ttl():
    retention, cache_control = anthropic_provider._get_cache_control(
        "https://api.anthropic.com/v1", "long"
    )
    assert retention == "long"
    assert cache_control is not None
    assert cache_control.get("ttl") == "1h"


def test_anthropic_sanitizes_surrogates():
    model = Model(
        id="claude",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=1000,
    )
    ctx = Context(
        system_prompt="hello\ud83d",
        messages=[UserMessage(content="hi\udc00", timestamp=1)],
        tools=None,
    )
    params = anthropic_provider._build_params(model, ctx, False, None, None)
    system_text = params["system"][0]["text"]
    assert "\ud83d" not in system_text
    assert "\udc00" not in params["messages"][0]["content"]

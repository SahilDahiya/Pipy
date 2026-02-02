from pi_ai.providers import openai as openai_provider
from pi_ai.types import Context, ImageContent, TextContent, ToolCall, ToolResultMessage
from pi_ai.models import create_openai_model

from tests.helpers import create_assistant_message


def test_openai_tool_result_images_create_user_message():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    tool_result = ToolResultMessage(
        tool_call_id="tool-1",
        tool_name="read",
        content=[
            TextContent(text="here"),
            ImageContent(data="abc", mime_type="image/png"),
        ],
        is_error=False,
    )
    ctx = Context(system_prompt=None, messages=[tool_result], tools=None)
    params = openai_provider._build_params(model, ctx, None)

    messages = params["messages"]
    assert any(msg.get("role") == "tool" for msg in messages)
    assert any(msg.get("role") == "user" and isinstance(msg.get("content"), list) for msg in messages)


def test_openai_adds_empty_tools_when_history_present():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    tool_call = ToolCall(id="tool-1", name="echo", arguments={"value": "hi"})
    assistant = create_assistant_message([tool_call], stop_reason="toolUse")
    ctx = Context(system_prompt=None, messages=[assistant], tools=None)
    params = openai_provider._build_params(model, ctx, None)
    assert params["tools"] == []

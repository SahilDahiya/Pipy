from pi_ai.providers import openai as openai_provider
from pi_ai.types import Context, ImageContent, TextContent, ToolCall, ToolResultMessage
from pi_ai.models import create_openai_model

from tests.helpers import create_assistant_message, create_user_message


def test_openai_tool_result_images_create_user_message():
    model = create_openai_model(
        "gpt-4o-mini",
        provider="openai",
        input_modalities=["text", "image"],
    )
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


def test_openai_batches_tool_result_images():
    model = create_openai_model(
        "gpt-4o-mini",
        provider="openai",
        input_modalities=["text", "image"],
    )
    assistant = create_assistant_message(
        [
            ToolCall(id="tool-1", name="read", arguments={"path": "img-1.png"}),
            ToolCall(id="tool-2", name="read", arguments={"path": "img-2.png"}),
        ],
        stop_reason="toolUse",
    )
    tool_result_one = ToolResultMessage(
        tool_call_id="tool-1",
        tool_name="read",
        content=[
            TextContent(text="Read image file [image/png]"),
            ImageContent(data="ZmFrZQ==", mime_type="image/png"),
        ],
        is_error=False,
    )
    tool_result_two = ToolResultMessage(
        tool_call_id="tool-2",
        tool_name="read",
        content=[
            TextContent(text="Read image file [image/png]"),
            ImageContent(data="ZmFrZQ==", mime_type="image/png"),
        ],
        is_error=False,
    )
    ctx = Context(messages=[create_user_message("Read the images"), assistant, tool_result_one, tool_result_two])

    params = openai_provider._build_params(model, ctx, None)
    roles = [msg.get("role") for msg in params["messages"]]
    assert roles == ["user", "assistant", "tool", "tool", "user"]
    image_message = params["messages"][-1]
    assert image_message.get("role") == "user"
    content = image_message.get("content")
    assert isinstance(content, list)
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == 2

def test_openai_adds_empty_tools_when_history_present():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    tool_call = ToolCall(id="tool-1", name="echo", arguments={"value": "hi"})
    assistant = create_assistant_message([tool_call], stop_reason="toolUse")
    ctx = Context(system_prompt=None, messages=[assistant], tools=None)
    params = openai_provider._build_params(model, ctx, None)
    assert params["tools"] == []


def test_openai_sanitizes_surrogates():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    ctx = Context(
        system_prompt="hello\ud83d",
        messages=[create_user_message("hi\udc00")],
        tools=None,
    )
    params = openai_provider._build_params(model, ctx, None)
    assert "\ud83d" not in params["messages"][0]["content"]
    assert "\udc00" not in params["messages"][1]["content"]


def test_openai_normalizes_pipe_tool_call_ids():
    model = create_openai_model("gpt-4o-mini", provider="openai")
    tool_call = ToolCall(id="call_abc|long+id==", name="echo", arguments={"value": "hi"})
    assistant = create_assistant_message([tool_call], stop_reason="toolUse")
    tool_result = ToolResultMessage(
        tool_call_id="call_abc|long+id==",
        tool_name="echo",
        content=[TextContent(text="ok")],
        is_error=False,
    )
    ctx = Context(system_prompt=None, messages=[assistant, tool_result], tools=None)
    params = openai_provider._build_params(model, ctx, None)

    tool_messages = [msg for msg in params["messages"] if msg.get("role") == "tool"]
    assert tool_messages
    assert "|" not in tool_messages[0]["tool_call_id"]


def test_openai_clamps_xhigh_reasoning():
    model = create_openai_model("gpt-4o-mini", provider="openai", reasoning=True)
    ctx = Context(messages=[create_user_message("hi")])
    params = openai_provider._build_params(
        model,
        ctx,
        openai_provider.OpenAICompletionsOptions(reasoning_effort="xhigh"),
    )
    assert params.get("reasoning_effort") == "high"

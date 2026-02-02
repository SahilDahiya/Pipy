from pi_ai.transform import transform_messages
from pi_ai.types import TextContent, ToolCall, ToolResultMessage

from tests.helpers import create_assistant_message, create_model, create_user_message


def test_inserts_tool_result_for_orphan_tool_call():
    assistant = create_assistant_message(
        [ToolCall(id="tool-1", name="echo", arguments={"value": "hi"})],
        stop_reason="toolUse",
    )
    user = create_user_message("follow up")

    model = create_model()
    transformed = transform_messages([assistant, user], model)

    assert len(transformed) == 3
    assert transformed[0].role == "assistant"
    assert transformed[1].role == "toolResult"
    assert transformed[2].role == "user"


def test_tool_call_id_normalization():
    assistant = create_assistant_message(
        [ToolCall(id="orig", name="echo", arguments={"value": "hi"})],
        stop_reason="toolUse",
    )
    tool_result = ToolResultMessage(
        tool_call_id="orig",
        tool_name="echo",
        content=[TextContent(text="ok")],
        is_error=False,
    )

    def normalize(tool_id, _model, _source):
        return "normalized"

    model = create_model()
    transformed = transform_messages([assistant, tool_result], model, normalize)

    assert transformed[1].tool_call_id == "normalized"


def test_cross_provider_thinking_conversion():
    assistant = create_assistant_message(
        [TextContent(text="hi")],
        stop_reason="stop",
    )
    assistant.content.append(
        ToolCall(id="tool-1", name="echo", arguments={"value": "hi"}, thought_signature="sig")
    )
    model = create_model()
    other_model = create_model()
    other_model.provider = "anthropic"
    other_model.api = "anthropic-messages"

    transformed = transform_messages([assistant], other_model)
    tool_call = next(block for block in transformed[0].content if block.type == "toolCall")
    assert tool_call.thought_signature is None

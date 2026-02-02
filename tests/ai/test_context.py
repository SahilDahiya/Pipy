from pi_ai.context import deserialize_context, serialize_context
from pi_ai.types import Context, Tool, UserMessage


def test_context_roundtrip():
    context = Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="hello")],
        tools=[Tool(name="echo", description="Echo", parameters={"type": "object", "properties": {}})],
    )
    payload = serialize_context(context)
    restored = deserialize_context(payload)
    assert restored.system_prompt == context.system_prompt
    assert restored.messages[0].role == "user"
    assert restored.tools[0].name == "echo"

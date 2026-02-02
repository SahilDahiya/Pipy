from pi_sdk import create_agent
from pi_session.manager import SessionManager
from pi_ai.types import UserMessage


def test_create_agent_restores_session_context(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager.open(str(path))
    manager.append_message(UserMessage(content="hello"))
    manager.append_thinking_level_change("high")
    manager.append_model_change("anthropic", "claude-test")

    agent = create_agent(session_path=str(path))
    assert agent.state.thinking_level == "high"
    assert agent.state.model.provider == "anthropic"
    assert agent.state.model.id == "claude-test"
    assert agent.state.messages[0].role == "user"

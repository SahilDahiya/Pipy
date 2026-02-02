import json

from pi_ai.types import UserMessage
from pi_session.manager import SessionManager


def test_session_manager_writes_header(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager.open(str(path))
    manager.append_message(UserMessage(content="hello"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert "id" in header


def test_load_messages(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager.open(str(path))

    manager.append_message(UserMessage(content="hello"))
    manager.append_message(UserMessage(content="world"))

    loaded = manager.load_messages()
    assert [msg.content for msg in loaded] == ["hello", "world"]

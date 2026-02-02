import json

from pi_ai.types import TextContent, UserMessage
from pi_session.manager import SessionManager


def test_session_manager_writes_header(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager(str(path), cwd=str(tmp_path))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["cwd"] == str(tmp_path)
    assert "id" in header

    manager.append_message(UserMessage(content="hello"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_session_tree_and_leaf(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager(str(path), cwd=str(tmp_path))

    first = manager.append_message(UserMessage(content="first"))
    second = manager.append_message(UserMessage(content="second"))

    assert manager.get_leaf_id() == second.entry_id
    tree = manager.get_tree()
    root_id = tree.root_id()
    assert root_id is not None
    assert second.entry_id in tree.children(first.entry_id)


def test_session_branching(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager(str(path), cwd=str(tmp_path))

    first = manager.append_message(UserMessage(content="first"))
    second = manager.append_message(UserMessage(content="second"))
    branch = manager.append_entry(
        "message",
        {"message": UserMessage(content="branch").model_dump()},
        parent_id=first.entry_id,
    )

    tree = manager.get_tree()
    assert branch.entry_id in tree.children(first.entry_id)
    assert second.entry_id in tree.children(first.entry_id)


def test_load_messages(tmp_path):
    path = tmp_path / "session.jsonl"
    manager = SessionManager(str(path), cwd=str(tmp_path))

    manager.append_message(UserMessage(content="hello"))
    manager.append_message(UserMessage(content="world"))

    loaded = manager.load_messages()
    assert [msg.content for msg in loaded] == ["hello", "world"]

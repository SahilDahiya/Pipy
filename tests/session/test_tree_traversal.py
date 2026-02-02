import pytest

from pi_session import SessionManager
from tests.session.helpers import assistant_msg, user_msg


def test_append_chain_parents():
    session = SessionManager.in_memory()
    id1 = session.append_message(user_msg("first"))
    id2 = session.append_message(assistant_msg("second"))
    id3 = session.append_message(user_msg("third"))

    entries = session.get_entries()
    assert len(entries) == 3
    assert entries[0].id == id1
    assert entries[0].parentId is None
    assert entries[1].parentId == id1
    assert entries[2].parentId == id2


def test_append_thinking_model_compaction_custom():
    session = SessionManager.in_memory()
    msg_id = session.append_message(user_msg("hello"))
    thinking_id = session.append_thinking_level_change("high")
    model_id = session.append_model_change("openai", "gpt-4")
    compaction_id = session.append_compaction("summary", msg_id, 1000)
    custom_id = session.append_custom_entry("my_data", {"key": "value"})
    session.append_message(assistant_msg("response"))

    entries = session.get_entries()
    thinking_entry = next(e for e in entries if e.type == "thinking_level_change")
    assert thinking_entry.id == thinking_id
    assert thinking_entry.parentId == msg_id

    model_entry = next(e for e in entries if e.type == "model_change")
    assert model_entry.id == model_id
    assert model_entry.parentId == thinking_id

    compaction_entry = next(e for e in entries if e.type == "compaction")
    assert compaction_entry.id == compaction_id
    assert compaction_entry.parentId == model_id

    custom_entry = next(e for e in entries if e.type == "custom")
    assert custom_entry.id == custom_id
    assert custom_entry.parentId == compaction_id


def test_leaf_pointer_advances():
    session = SessionManager.in_memory()
    assert session.get_leaf_id() is None
    id1 = session.append_message(user_msg("1"))
    assert session.get_leaf_id() == id1
    id2 = session.append_message(assistant_msg("2"))
    assert session.get_leaf_id() == id2


def test_get_branch_paths():
    session = SessionManager.in_memory()
    id1 = session.append_message(user_msg("1"))
    id2 = session.append_message(assistant_msg("2"))
    _id3 = session.append_message(user_msg("3"))
    path = session.get_branch()
    assert [e.id for e in path] == [id1, id2, path[-1].id]

    branch = session.get_branch(id2)
    assert [e.id for e in branch] == [id1, id2]


def test_get_tree_and_branching():
    session = SessionManager.in_memory()
    id1 = session.append_message(user_msg("1"))
    id2 = session.append_message(assistant_msg("2"))
    id3 = session.append_message(user_msg("3"))

    session.branch(id2)
    id4 = session.append_message(user_msg("4-branch"))

    tree = session.get_tree()
    root = next(node for node in tree if node.entry.id == id1)
    child_ids = [child.entry.id for child in root.children]
    assert id2 in child_ids
    node2 = next(child for child in root.children if child.entry.id == id2)
    branch_ids = [child.entry.id for child in node2.children]
    assert id3 in branch_ids
    assert id4 in branch_ids


def test_branch_invalid_raises():
    session = SessionManager.in_memory()
    with pytest.raises(ValueError):
        session.branch("missing")


def test_branch_with_summary_inserts_entry():
    session = SessionManager.in_memory()
    id1 = session.append_message(user_msg("1"))
    session.append_message(assistant_msg("2"))
    session.append_message(user_msg("3"))

    summary_id = session.branch_with_summary(id1, "Summary of abandoned work")
    assert session.get_leaf_id() == summary_id
    summary_entry = next(e for e in session.get_entries() if e.type == "branch_summary")
    assert summary_entry.parentId == id1
    assert summary_entry.summary == "Summary of abandoned work"


def test_branch_with_summary_invalid_raises():
    session = SessionManager.in_memory()
    session.append_message(user_msg("hello"))
    with pytest.raises(ValueError):
        session.branch_with_summary("missing", "summary")


def test_get_leaf_entry():
    session = SessionManager.in_memory()
    assert session.get_leaf_entry() is None
    id1 = session.append_message(user_msg("first"))
    leaf = session.get_leaf_entry()
    assert leaf is not None
    assert leaf.id == id1


def test_get_entry_missing_returns_none():
    session = SessionManager.in_memory()
    assert session.get_entry("missing") is None


def test_get_children_returns_direct_children():
    session = SessionManager.in_memory()
    id1 = session.append_message(user_msg("root"))
    id2 = session.append_message(assistant_msg("child"))
    session.append_message(user_msg("grandchild"))

    children = session.get_children(id1)
    assert [entry.id for entry in children] == [id2]

from pi_session import SessionManager
from tests.session.helpers import assistant_msg, user_msg


def test_labels_set_get_and_clear():
    session = SessionManager.in_memory()
    msg_id = session.append_message(user_msg("hello"))

    assert session.get_label(msg_id) is None
    label_id = session.append_label_change(msg_id, "checkpoint")
    assert session.get_label(msg_id) == "checkpoint"

    entries = session.get_entries()
    label_entry = next(e for e in entries if e.type == "label")
    assert label_entry.id == label_id
    assert label_entry.targetId == msg_id
    assert label_entry.label == "checkpoint"

    session.append_label_change(msg_id, None)
    assert session.get_label(msg_id) is None


def test_label_last_wins():
    session = SessionManager.in_memory()
    msg_id = session.append_message(user_msg("hello"))
    session.append_label_change(msg_id, "first")
    session.append_label_change(msg_id, "second")
    session.append_label_change(msg_id, "third")
    assert session.get_label(msg_id) == "third"


def test_labels_in_tree_nodes():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(user_msg("hello"))
    msg2_id = session.append_message(assistant_msg("hi"))

    session.append_label_change(msg1_id, "start")
    session.append_label_change(msg2_id, "response")

    tree = session.get_tree()
    msg1_node = next(node for node in tree if node.entry.id == msg1_id)
    assert msg1_node.label == "start"
    msg2_node = next(node for node in msg1_node.children if node.entry.id == msg2_id)
    assert msg2_node.label == "response"


def test_labels_preserved_in_branched_session():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(user_msg("hello"))
    msg2_id = session.append_message(assistant_msg("hi"))

    session.append_label_change(msg1_id, "important")
    session.append_label_change(msg2_id, "also-important")

    session.create_branched_session(msg2_id)

    assert session.get_label(msg1_id) == "important"
    assert session.get_label(msg2_id) == "also-important"

    label_entries = [e for e in session.get_entries() if e.type == "label"]
    assert len(label_entries) == 2


def test_labels_not_on_path_removed():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(user_msg("hello"))
    msg2_id = session.append_message(assistant_msg("hi"))
    msg3_id = session.append_message(user_msg("followup"))

    session.append_label_change(msg1_id, "first")
    session.append_label_change(msg2_id, "second")
    session.append_label_change(msg3_id, "third")

    session.create_branched_session(msg2_id)

    assert session.get_label(msg1_id) == "first"
    assert session.get_label(msg2_id) == "second"
    assert session.get_label(msg3_id) is None

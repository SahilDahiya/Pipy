from pi_session import SessionManager
from tests.session.helpers import assistant_msg, user_msg


def test_custom_entries_are_in_tree_and_skipped_in_context():
    session = SessionManager.in_memory()

    msg_id = session.append_message(user_msg("hello"))
    custom_id = session.append_custom_entry("my_data", {"foo": "bar"})
    msg2_id = session.append_message(assistant_msg("hi"))

    entries = session.get_entries()
    custom_entry = next(entry for entry in entries if entry.type == "custom")
    assert custom_entry.custom_type == "my_data"
    assert custom_entry.data == {"foo": "bar"}
    assert custom_entry.id == custom_id
    assert custom_entry.parent_id == msg_id

    path = session.get_branch()
    assert [entry.id for entry in path] == [msg_id, custom_id, msg2_id]

    context = session.build_session_context()
    assert len(context.messages) == 2

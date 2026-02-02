import time
from pathlib import Path

from pi_ai.types import AssistantMessage, TextContent, Usage
from pi_session.manager import SessionManager


def _make_usage() -> Usage:
    return Usage(input=1, output=1, total_tokens=2)


def test_session_modified_uses_message_timestamp(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    manager = SessionManager.open(str(session_path))

    manager.append_message(
        AssistantMessage(
            content=[TextContent(text="hi")],
            api="openai-completions",
            provider="openai",
            model="test",
            usage=_make_usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )
    )

    before_mtime = session_path.stat().st_mtime
    time.sleep(0.01)

    msg_time = int(time.time() * 1000)
    manager.append_message(
        AssistantMessage(
            content=[TextContent(text="later")],
            api="openai-completions",
            provider="openai",
            model="test",
            usage=_make_usage(),
            stop_reason="stop",
            timestamp=msg_time,
        )
    )

    sessions = SessionManager.list(str(tmp_path), str(tmp_path))
    session_info = next((item for item in sessions if item.path == str(session_path)), None)
    assert session_info is not None
    assert int(session_info.modified.timestamp() * 1000) == msg_time
    assert session_info.modified.timestamp() != before_mtime

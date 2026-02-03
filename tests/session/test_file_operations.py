import json
from pathlib import Path

from pi_session import find_most_recent_session, load_entries_from_file, SessionManager


def test_load_entries_from_file(tmp_path):
    missing = tmp_path / "missing.jsonl"
    assert load_entries_from_file(str(missing)) == []

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_entries_from_file(str(empty)) == []

    no_header = tmp_path / "no-header.jsonl"
    no_header.write_text('{"type":"message","id":"1"}\n', encoding="utf-8")
    assert load_entries_from_file(str(no_header)) == []

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not json\n", encoding="utf-8")
    assert load_entries_from_file(str(malformed)) == []

    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
        '{"type":"message","id":"1","parent_id":null,"timestamp":"2025-01-01T00:00:01Z","message":{"role":"user","content":"hi","timestamp":1}}\n',
        encoding="utf-8",
    )
    entries = load_entries_from_file(str(valid))
    assert len(entries) == 2
    assert entries[0]["type"] == "session"
    assert entries[1]["type"] == "message"

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
        "not valid json\n"
        '{"type":"message","id":"1","parent_id":null,"timestamp":"2025-01-01T00:00:01Z","message":{"role":"user","content":"hi","timestamp":1}}\n',
        encoding="utf-8",
    )
    entries = load_entries_from_file(str(mixed))
    assert len(entries) == 2


def test_find_most_recent_session(tmp_path):
    assert find_most_recent_session(str(tmp_path)) is None
    assert find_most_recent_session(str(tmp_path / "missing")) is None

    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "file.json").write_text("{}", encoding="utf-8")
    assert find_most_recent_session(str(tmp_path)) is None

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"type":"message"}\n', encoding="utf-8")
    assert find_most_recent_session(str(tmp_path)) is None

    session = tmp_path / "session.jsonl"
    session.write_text('{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')
    assert find_most_recent_session(str(tmp_path)) == str(session)

    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text('{"type":"session","id":"old","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')
    newer.write_text('{"type":"session","id":"new","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')
    assert find_most_recent_session(str(tmp_path)) == str(newer)


def test_open_recovers_corrupted_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    manager = SessionManager.open(str(empty))
    header = manager.get_header()
    assert header is not None
    assert header.type == "session"
    assert manager.get_session_file() == str(empty.resolve())

    lines = empty.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["type"] == "session"

from pi_session import migrate_session_entries


def test_migrate_v1_adds_ids():
    entries = [
        {"type": "session", "id": "sess-1", "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
        {
            "type": "message",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": {"role": "user", "content": "hi", "timestamp": 1},
        },
        {
            "type": "message",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "api": "test",
                "provider": "test",
                "model": "test",
                "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                "stopReason": "stop",
                "timestamp": 2,
            },
        },
    ]

    migrate_session_entries(entries)
    assert entries[0]["version"] == 3
    assert entries[1]["id"]
    assert entries[1]["parentId"] is None
    assert entries[2]["parentId"] == entries[1]["id"]


def test_migrate_idempotent():
    entries = [
        {"type": "session", "id": "sess-1", "version": 2, "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
        {
            "type": "message",
            "id": "abc12345",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:01Z",
            "message": {"role": "user", "content": "hi", "timestamp": 1},
        },
        {
            "type": "message",
            "id": "def67890",
            "parentId": "abc12345",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "api": "test",
                "provider": "test",
                "model": "test",
                "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                "stopReason": "stop",
                "timestamp": 2,
            },
        },
    ]

    migrate_session_entries(entries)
    assert entries[1]["id"] == "abc12345"
    assert entries[2]["id"] == "def67890"
    assert entries[2]["parentId"] == "abc12345"

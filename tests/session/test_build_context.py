from pi_session import (
    BranchSummaryEntry,
    CompactionEntry,
    ModelChangeEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    build_session_context,
)

from tests.session.helpers import assistant_msg, user_msg


def get_role(message) -> str:
    return message.role if hasattr(message, "role") else message["role"]


def get_content(message):
    return message.content if hasattr(message, "content") else message["content"]


def get_first_text(message) -> str:
    content = get_content(message)
    if isinstance(content, str):
        return content
    first = content[0]
    return first.text if hasattr(first, "text") else first["text"]


def msg(entry_id: str, parent_id: str | None, role: str, text: str) -> SessionMessageEntry:
    message = user_msg(text) if role == "user" else assistant_msg(text)
    return SessionMessageEntry(
        type="message",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        message=message,
    )


def compaction(entry_id: str, parent_id: str | None, summary: str, first_kept: str) -> CompactionEntry:
    return CompactionEntry(
        type="compaction",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        summary=summary,
        first_kept_entry_id=first_kept,
        tokens_before=1000,
    )


def branch_summary(entry_id: str, parent_id: str | None, summary: str, from_id: str) -> BranchSummaryEntry:
    return BranchSummaryEntry(
        type="branch_summary",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        summary=summary,
        from_id=from_id,
    )


def thinking(entry_id: str, parent_id: str | None, level: str) -> ThinkingLevelChangeEntry:
    return ThinkingLevelChangeEntry(
        type="thinking_level_change",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        thinking_level=level,
    )


def model_change(entry_id: str, parent_id: str | None, provider: str, model_id: str) -> ModelChangeEntry:
    return ModelChangeEntry(
        type="model_change",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2025-01-01T00:00:00Z",
        provider=provider,
        model_id=model_id,
    )


def test_build_context_trivial():
    ctx = build_session_context([])
    assert ctx.messages == []
    assert ctx.thinking_level == "off"
    assert ctx.model is None

    entries = [msg("1", None, "user", "hello")]
    ctx = build_session_context(entries)
    assert len(ctx.messages) == 1
    assert get_role(ctx.messages[0]) == "user"


def test_build_context_simple_conversation():
    entries = [
        msg("1", None, "user", "hello"),
        msg("2", "1", "assistant", "hi"),
        msg("3", "2", "user", "how"),
        msg("4", "3", "assistant", "great"),
    ]
    ctx = build_session_context(entries)
    assert [get_role(m) for m in ctx.messages] == ["user", "assistant", "user", "assistant"]


def test_build_context_tracks_thinking_and_model():
    entries = [
        msg("1", None, "user", "hello"),
        thinking("2", "1", "high"),
        msg("3", "2", "assistant", "thinking"),
    ]
    ctx = build_session_context(entries)
    assert ctx.thinking_level == "high"
    assert ctx.model == {"provider": "anthropic", "model_id": "claude-test"}

    entries = [
        msg("1", None, "user", "hello"),
        model_change("2", "1", "openai", "gpt-4"),
        msg("3", "2", "assistant", "hi"),
    ]
    ctx = build_session_context(entries)
    assert ctx.model == {"provider": "anthropic", "model_id": "claude-test"}


def test_build_context_with_compaction():
    entries = [
        msg("1", None, "user", "first"),
        msg("2", "1", "assistant", "response1"),
        msg("3", "2", "user", "second"),
        msg("4", "3", "assistant", "response2"),
        compaction("5", "4", "Summary of first two turns", "3"),
        msg("6", "5", "user", "third"),
        msg("7", "6", "assistant", "response3"),
    ]
    ctx = build_session_context(entries)
    assert get_role(ctx.messages[0]) == "compaction_summary"
    assert "Summary" in ctx.messages[0]["summary"]
    assert get_content(ctx.messages[1]) == "second"
    assert get_first_text(ctx.messages[2]) == "response2"


def test_build_context_multiple_compactions():
    entries = [
        msg("1", None, "user", "a"),
        msg("2", "1", "assistant", "b"),
        compaction("3", "2", "First summary", "1"),
        msg("4", "3", "user", "c"),
        msg("5", "4", "assistant", "d"),
        compaction("6", "5", "Second summary", "4"),
        msg("7", "6", "user", "e"),
    ]
    ctx = build_session_context(entries)
    assert ctx.messages[0]["summary"] == "Second summary"


def test_build_context_branches_and_branch_summary():
    entries = [
        msg("1", None, "user", "start"),
        msg("2", "1", "assistant", "response"),
        msg("3", "2", "user", "branch A"),
        msg("4", "2", "user", "branch B"),
    ]

    ctx_a = build_session_context(entries, "3")
    assert get_content(ctx_a.messages[-1]) == "branch A"

    ctx_b = build_session_context(entries, "4")
    assert get_content(ctx_b.messages[-1]) == "branch B"

    entries = [
        msg("1", None, "user", "start"),
        msg("2", "1", "assistant", "response"),
        msg("3", "2", "user", "abandoned"),
        branch_summary("4", "2", "Summary of abandoned work", "3"),
        msg("5", "4", "user", "new path"),
    ]

    ctx = build_session_context(entries)
    assert get_role(ctx.messages[2]) == "branch_summary"
    assert "Summary" in ctx.messages[2]["summary"]

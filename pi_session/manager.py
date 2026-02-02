"""JSONL session persistence and tree utilities for pi-python."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from pi_ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage

CURRENT_SESSION_VERSION = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id(existing: Iterable[str]) -> str:
    existing_set = set(existing)
    for _ in range(100):
        candidate = uuid4().hex[:8]
        if candidate not in existing_set:
            return candidate
    return uuid4().hex


def _get_agent_dir() -> str:
    env_dir = os.getenv("PI_CODING_AGENT_DIR")
    if env_dir:
        if env_dir == "~":
            return str(Path.home())
        if env_dir.startswith("~/"):
            return str(Path.home() / env_dir[2:])
        return env_dir
    return str(Path.home() / ".pi" / "agent")


def get_sessions_dir() -> str:
    return str(Path(_get_agent_dir()) / "sessions")


def get_default_session_dir(cwd: str) -> str:
    safe_path = f"--{cwd.lstrip('/\\\\').replace('/', '-').replace('\\\\', '-').replace(':', '-')}--"
    session_dir = Path(get_sessions_dir()) / safe_path
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir)


def _parse_iso_timestamp(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _is_message_with_content(message: Any) -> bool:
    return isinstance(message, dict) and isinstance(message.get("role"), str) and "content" in message


def _extract_text_content(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _get_last_activity_time(entries: List[Dict[str, Any]]) -> Optional[int]:
    last_activity: Optional[int] = None
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not _is_message_with_content(message):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        msg_timestamp = message.get("timestamp")
        if isinstance(msg_timestamp, int):
            last_activity = max(last_activity or 0, msg_timestamp)
            continue
        entry_timestamp = entry.get("timestamp")
        if isinstance(entry_timestamp, str):
            parsed = _parse_iso_timestamp(entry_timestamp)
            if parsed:
                last_activity = max(last_activity or 0, int(parsed.timestamp() * 1000))
    return last_activity


def _get_session_modified_date(
    entries: List[Dict[str, Any]],
    header: Dict[str, Any],
    stats_mtime: datetime,
) -> datetime:
    last_activity_time = _get_last_activity_time(entries)
    if isinstance(last_activity_time, int) and last_activity_time > 0:
        return datetime.fromtimestamp(last_activity_time / 1000, tz=timezone.utc)
    header_time = header.get("timestamp")
    if isinstance(header_time, str):
        parsed = _parse_iso_timestamp(header_time)
        if parsed:
            return parsed
    return stats_mtime


def build_session_info(file_path: str) -> Optional[SessionInfo]:
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return None

    entries: List[Dict[str, Any]] = []
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)

    if not entries:
        return None
    header = entries[0]
    if header.get("type") != "session":
        return None

    stats = Path(file_path).stat()
    stats_mtime = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)

    message_count = 0
    first_message = ""
    all_messages: List[str] = []
    name: Optional[str] = None

    for entry in entries:
        if entry.get("type") == "session_info":
            entry_name = entry.get("name")
            if isinstance(entry_name, str) and entry_name.strip():
                name = entry_name.strip()

        if entry.get("type") != "message":
            continue
        message_count += 1
        message = entry.get("message")
        if not _is_message_with_content(message):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text_content = _extract_text_content(message)
        if not text_content:
            continue
        all_messages.append(text_content)
        if not first_message and role == "user":
            first_message = text_content

    cwd = header.get("cwd") if isinstance(header.get("cwd"), str) else ""
    parent_session_path = header.get("parentSession")
    header_time = header.get("timestamp")
    created = _parse_iso_timestamp(header_time) if isinstance(header_time, str) else None
    if created is None:
        created = stats_mtime

    modified = _get_session_modified_date(entries, header, stats_mtime)

    return SessionInfo(
        path=file_path,
        id=str(header.get("id", "")),
        cwd=cwd,
        name=name,
        parent_session_path=parent_session_path,
        created=created,
        modified=modified,
        message_count=message_count,
        first_message=first_message or "(no messages)",
        all_messages_text=" ".join(all_messages),
    )


def list_sessions_from_dir(
    dir_path: str,
    on_progress: Optional[SessionListProgress] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> List[SessionInfo]:
    sessions: List[SessionInfo] = []
    directory = Path(dir_path)
    if not directory.exists():
        return sessions
    try:
        files = [p for p in directory.iterdir() if p.suffix == ".jsonl"]
        total = progress_total if progress_total is not None else len(files)
        loaded = 0
        for file_path in files:
            info = build_session_info(str(file_path))
            loaded += 1
            if on_progress:
                on_progress(progress_offset + loaded, total)
            if info:
                sessions.append(info)
    except Exception:
        return []
    return sessions


@dataclass
class SessionHeader:
    type: str
    id: str
    timestamp: str
    cwd: str
    version: int = CURRENT_SESSION_VERSION
    parentSession: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "cwd": self.cwd,
            "version": self.version,
        }
        if self.parentSession:
            data["parentSession"] = self.parentSession
        return data


@dataclass
class SessionEntry:
    type: str
    id: str
    parentId: Optional[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "parentId": self.parentId,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionMessageEntry(SessionEntry):
    message: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["message"] = self.message
        return data


@dataclass
class ThinkingLevelChangeEntry(SessionEntry):
    thinkingLevel: str

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["thinkingLevel"] = self.thinkingLevel
        return data


@dataclass
class ModelChangeEntry(SessionEntry):
    provider: str
    modelId: str

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["provider"] = self.provider
        data["modelId"] = self.modelId
        return data


@dataclass
class CompactionEntry(SessionEntry):
    summary: str
    firstKeptEntryId: str
    tokensBefore: int
    details: Optional[Dict[str, Any]] = None
    fromHook: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "summary": self.summary,
                "firstKeptEntryId": self.firstKeptEntryId,
                "tokensBefore": self.tokensBefore,
            }
        )
        if self.details is not None:
            data["details"] = self.details
        if self.fromHook is not None:
            data["fromHook"] = self.fromHook
        return data


@dataclass
class BranchSummaryEntry(SessionEntry):
    fromId: str
    summary: str
    details: Optional[Dict[str, Any]] = None
    fromHook: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"fromId": self.fromId, "summary": self.summary})
        if self.details is not None:
            data["details"] = self.details
        if self.fromHook is not None:
            data["fromHook"] = self.fromHook
        return data


@dataclass
class CustomEntry(SessionEntry):
    customType: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["customType"] = self.customType
        if self.data is not None:
            data["data"] = self.data
        return data


@dataclass
class CustomMessageEntry(SessionEntry):
    customType: str
    content: Any
    display: bool
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"customType": self.customType, "content": self.content, "display": self.display})
        if self.details is not None:
            data["details"] = self.details
        return data


@dataclass
class LabelEntry(SessionEntry):
    targetId: str
    label: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["targetId"] = self.targetId
        if self.label is not None:
            data["label"] = self.label
        return data


@dataclass
class SessionInfoEntry(SessionEntry):
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass
class SessionInfo:
    path: str
    id: str
    cwd: str
    name: Optional[str]
    parent_session_path: Optional[str]
    created: datetime
    modified: datetime
    message_count: int
    first_message: str
    all_messages_text: str


SessionListProgress = Callable[[int, int], None]


SessionEntryType = (
    SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
)


@dataclass
class SessionTreeNode:
    entry: SessionEntryType
    children: List["SessionTreeNode"]
    label: Optional[str] = None


@dataclass
class SessionContext:
    messages: List[Any]
    thinkingLevel: str
    model: Optional[Dict[str, str]]


def _entry_from_dict(payload: Dict[str, Any]) -> SessionEntryType:
    entry_type = payload.get("type")
    base = {
        "type": entry_type,
        "id": payload.get("id"),
        "parentId": payload.get("parentId"),
        "timestamp": payload.get("timestamp", _now_iso()),
    }
    if entry_type == "message":
        return SessionMessageEntry(**base, message=payload.get("message", {}))
    if entry_type == "thinking_level_change":
        return ThinkingLevelChangeEntry(**base, thinkingLevel=payload.get("thinkingLevel", "off"))
    if entry_type == "model_change":
        return ModelChangeEntry(**base, provider=payload.get("provider", ""), modelId=payload.get("modelId", ""))
    if entry_type == "compaction":
        return CompactionEntry(
            **base,
            summary=payload.get("summary", ""),
            firstKeptEntryId=payload.get("firstKeptEntryId", ""),
            tokensBefore=payload.get("tokensBefore", 0),
            details=payload.get("details"),
            fromHook=payload.get("fromHook"),
        )
    if entry_type == "branch_summary":
        return BranchSummaryEntry(
            **base,
            fromId=payload.get("fromId", ""),
            summary=payload.get("summary", ""),
            details=payload.get("details"),
            fromHook=payload.get("fromHook"),
        )
    if entry_type == "custom":
        return CustomEntry(**base, customType=payload.get("customType", ""), data=payload.get("data"))
    if entry_type == "custom_message":
        return CustomMessageEntry(
            **base,
            customType=payload.get("customType", ""),
            content=payload.get("content"),
            display=bool(payload.get("display", False)),
            details=payload.get("details"),
        )
    if entry_type == "label":
        return LabelEntry(**base, targetId=payload.get("targetId", ""), label=payload.get("label"))
    if entry_type == "session_info":
        return SessionInfoEntry(**base, name=payload.get("name"))
    return SessionMessageEntry(**base, message=payload.get("message", {}))


def load_entries_from_file(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("type"):
            entries.append(data)
    if not any(entry.get("type") == "session" for entry in entries):
        return []
    return entries


def find_most_recent_session(session_dir: str) -> Optional[str]:
    dir_path = Path(session_dir)
    if not dir_path.exists():
        return None
    candidates: List[Path] = []
    for item in dir_path.iterdir():
        if item.suffix != ".jsonl":
            continue
        entries = load_entries_from_file(str(item))
        if entries:
            candidates.append(item)
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def migrate_session_entries(entries: List[Dict[str, Any]]) -> bool:
    header = next((entry for entry in entries if entry.get("type") == "session"), None)
    version = int(header.get("version", 1)) if header else 1
    changed = False

    if version < 2:
        existing_ids = set()
        previous_id = None
        for entry in entries:
            if entry.get("type") == "session":
                entry["version"] = 2
                continue
            entry_id = _generate_id(existing_ids)
            existing_ids.add(entry_id)
            entry["id"] = entry_id
            entry["parentId"] = previous_id
            previous_id = entry_id
            if entry.get("type") == "compaction" and "firstKeptEntryIndex" in entry:
                index = entry.get("firstKeptEntryIndex")
                if isinstance(index, int) and 0 <= index < len(entries):
                    target = entries[index]
                    if target.get("type") != "session":
                        entry["firstKeptEntryId"] = target.get("id")
                entry.pop("firstKeptEntryIndex", None)
        changed = True
        version = 2

    if version < 3:
        for entry in entries:
            if entry.get("type") == "session":
                entry["version"] = 3
            if entry.get("type") == "message":
                message = entry.get("message") or {}
                if message.get("role") == "hookMessage":
                    message["role"] = "custom"
                    entry["message"] = message
        changed = True

    return changed


_LEAF_UNSET = object()


def build_session_context(
    entries: List[SessionEntryType],
    leaf_id: Optional[str] | object = _LEAF_UNSET,
    by_id: Optional[Dict[str, SessionEntryType]] = None,
) -> SessionContext:
    if by_id is None:
        by_id = {entry.id: entry for entry in entries}

    if leaf_id is _LEAF_UNSET:
        leaf = entries[-1] if entries else None
    elif leaf_id is None:
        leaf = None
    else:
        leaf = by_id.get(leaf_id)

    if leaf is None:
        return SessionContext(messages=[], thinkingLevel="off", model=None)

    path: List[SessionEntryType] = []
    current = leaf
    while current:
        path.insert(0, current)
        current = by_id.get(current.parentId) if current.parentId else None

    thinking_level = "off"
    model: Optional[Dict[str, str]] = None
    compaction: Optional[CompactionEntry] = None

    for entry in path:
        if isinstance(entry, ThinkingLevelChangeEntry):
            thinking_level = entry.thinkingLevel
        elif isinstance(entry, ModelChangeEntry):
            model = {"provider": entry.provider, "modelId": entry.modelId}
        elif isinstance(entry, SessionMessageEntry):
            msg = entry.message
            if msg.get("role") == "assistant":
                model = {"provider": msg.get("provider", ""), "modelId": msg.get("model", "")}
        elif isinstance(entry, CompactionEntry):
            compaction = entry

    messages: List[Any] = []

    def append_message(entry: SessionEntryType) -> None:
        if isinstance(entry, SessionMessageEntry):
            messages.append(entry.message)
        elif isinstance(entry, CustomMessageEntry):
            messages.append(
                {
                    "role": "custom",
                    "customType": entry.customType,
                    "content": entry.content,
                    "display": entry.display,
                    "details": entry.details,
                    "timestamp": int(datetime.fromisoformat(entry.timestamp).timestamp() * 1000),
                }
            )
        elif isinstance(entry, BranchSummaryEntry):
            messages.append(
                {
                    "role": "branchSummary",
                    "summary": entry.summary,
                    "fromId": entry.fromId,
                    "timestamp": int(datetime.fromisoformat(entry.timestamp).timestamp() * 1000),
                }
            )

    if compaction:
        messages.append(
            {
                "role": "compactionSummary",
                "summary": compaction.summary,
                "tokensBefore": compaction.tokensBefore,
                "timestamp": int(datetime.fromisoformat(compaction.timestamp).timestamp() * 1000),
            }
        )
        compaction_idx = next(
            (idx for idx, entry in enumerate(path) if entry.id == compaction.id), -1
        )
        found_first_kept = False
        for entry in path[:compaction_idx]:
            if entry.id == compaction.firstKeptEntryId:
                found_first_kept = True
            if found_first_kept:
                append_message(entry)
        for entry in path[compaction_idx + 1 :]:
            append_message(entry)
    else:
        for entry in path:
            append_message(entry)

    return SessionContext(messages=messages, thinkingLevel=thinking_level, model=model)


class SessionManager:
    def __init__(self, cwd: str, session_dir: str, session_file: Optional[str], persist: bool) -> None:
        self._cwd = cwd
        self._session_dir = session_dir
        self._session_file = session_file
        self._persist = persist
        self._flushed = False
        self._file_entries: List[SessionHeader | SessionEntryType] = []
        self._by_id: Dict[str, SessionEntryType] = {}
        self._labels_by_id: Dict[str, str] = {}
        self._leaf_id: Optional[str] = None
        self._session_id: str = ""

        if self._persist and self._session_dir:
            Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        if session_file:
            self.set_session_file(session_file)
        else:
            self.new_session()

    @classmethod
    def create(cls, cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        dir_path = session_dir or get_default_session_dir(cwd)
        return cls(cwd, dir_path, None, True)

    @classmethod
    def open(cls, path: str, session_dir: Optional[str] = None) -> "SessionManager":
        entries = load_entries_from_file(path)
        header = next((e for e in entries if e.get("type") == "session"), None)
        cwd = header.get("cwd") if header else os.getcwd()
        return cls(cwd, session_dir or str(Path(path).resolve().parent), path, True)

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        dir_path = session_dir or get_default_session_dir(cwd)
        most_recent = find_most_recent_session(dir_path)
        if most_recent:
            return cls(cwd, dir_path, most_recent, True)
        return cls(cwd, dir_path, None, True)

    @staticmethod
    def list(
        cwd: str,
        session_dir: Optional[str] = None,
        on_progress: Optional[SessionListProgress] = None,
    ) -> List[SessionInfo]:
        dir_path = session_dir or get_default_session_dir(cwd)
        sessions = list_sessions_from_dir(dir_path, on_progress)
        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @staticmethod
    def list_all(on_progress: Optional[SessionListProgress] = None) -> List[SessionInfo]:
        sessions_dir = Path(get_sessions_dir())
        if not sessions_dir.exists():
            return []
        try:
            dirs = [entry for entry in sessions_dir.iterdir() if entry.is_dir()]
        except Exception:
            return []

        all_files: List[Path] = []
        for dir_entry in dirs:
            try:
                all_files.extend([p for p in dir_entry.iterdir() if p.suffix == ".jsonl"])
            except Exception:
                continue

        total_files = len(all_files)
        loaded = 0
        sessions: List[SessionInfo] = []
        for file_path in all_files:
            info = build_session_info(str(file_path))
            loaded += 1
            if on_progress:
                on_progress(loaded, total_files)
            if info:
                sessions.append(info)
        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @classmethod
    def in_memory(cls, cwd: Optional[str] = None) -> "SessionManager":
        return cls(cwd or os.getcwd(), "", None, False)

    def set_session_file(self, session_file: str) -> None:
        self._session_file = str(Path(session_file).resolve())
        if Path(self._session_file).exists():
            entries = load_entries_from_file(self._session_file)
            if not entries:
                explicit = self._session_file
                self.new_session()
                self._session_file = explicit
                self._rewrite_file()
                self._flushed = True
                return

            if migrate_session_entries(entries):
                self._rewrite_file(entries)

            header = next((e for e in entries if e.get("type") == "session"), None)
            self._session_id = header.get("id") if header else uuid4().hex
            self._file_entries = [SessionHeader(**header)] if header else []
            for entry in entries:
                if entry.get("type") == "session":
                    continue
                self._file_entries.append(_entry_from_dict(entry))
            self._build_index()
            self._flushed = True
        else:
            explicit = self._session_file
            self.new_session()
            self._session_file = explicit

    def new_session(self, parent_session: Optional[str] = None) -> Optional[str]:
        self._session_id = uuid4().hex
        timestamp = _now_iso()
        header = SessionHeader(
            type="session",
            id=self._session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            version=CURRENT_SESSION_VERSION,
            parentSession=parent_session,
        )
        self._file_entries = [header]
        self._by_id.clear()
        self._labels_by_id.clear()
        self._leaf_id = None
        self._flushed = False

        if self._persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self._session_file = str(Path(self._session_dir) / f"{file_timestamp}_{self._session_id}.jsonl")
        return self._session_file

    def _build_index(self) -> None:
        self._by_id.clear()
        self._labels_by_id.clear()
        self._leaf_id = None
        for entry in self.get_entries():
            self._by_id[entry.id] = entry
            self._leaf_id = entry.id
            if isinstance(entry, LabelEntry):
                if entry.label is not None:
                    self._labels_by_id[entry.targetId] = entry.label
                else:
                    self._labels_by_id.pop(entry.targetId, None)

    def _rewrite_file(self, entries: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self._persist or not self._session_file:
            return
        if entries is None:
            entries = [entry.to_dict() for entry in self._file_entries]
        content = "\n".join(json.dumps(entry) for entry in entries) + "\n"
        Path(self._session_file).write_text(content, encoding="utf-8")

    def _persist_entry(self, entry: SessionEntryType) -> None:
        if not self._persist or not self._session_file:
            return
        if not self._flushed:
            content = "\n".join(json.dumps(entry.to_dict()) for entry in self._file_entries) + "\n"
            Path(self._session_file).write_text(content, encoding="utf-8")
            self._flushed = True
            return
        with Path(self._session_file).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict()) + "\n")

    def _append_entry(self, entry: SessionEntryType) -> None:
        self._file_entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        if isinstance(entry, LabelEntry):
            if entry.label is not None:
                self._labels_by_id[entry.targetId] = entry.label
            else:
                self._labels_by_id.pop(entry.targetId, None)
        self._persist_entry(entry)

    def is_persisted(self) -> bool:
        return self._persist

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_file(self) -> Optional[str]:
        return self._session_file

    def get_header(self) -> Optional[SessionHeader]:
        return next((e for e in self._file_entries if isinstance(e, SessionHeader)), None)

    def get_entries(self) -> List[SessionEntryType]:
        return [e for e in self._file_entries if not isinstance(e, SessionHeader)]

    def get_entry(self, entry_id: str) -> SessionEntryType:
        return self._by_id.get(entry_id)

    def get_label(self, entry_id: str) -> Optional[str]:
        return self._labels_by_id.get(entry_id)

    def get_leaf_id(self) -> Optional[str]:
        return self._leaf_id

    def get_leaf_entry(self) -> Optional[SessionEntryType]:
        if not self._leaf_id:
            return None
        return self._by_id.get(self._leaf_id)

    def get_children(self, parent_id: str) -> List[SessionEntryType]:
        return [entry for entry in self._by_id.values() if entry.parentId == parent_id]

    def get_session_name(self) -> Optional[str]:
        entries = self.get_entries()
        for entry in reversed(entries):
            if isinstance(entry, SessionInfoEntry) and entry.name:
                return entry.name
        return None

    def append_message(self, message: Message | Dict[str, Any]) -> str:
        entry = SessionMessageEntry(
            type="message",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            message=message.model_dump() if hasattr(message, "model_dump") else message,
        )
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, level: str) -> str:
        entry = ThinkingLevelChangeEntry(
            type="thinking_level_change",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            thinkingLevel=level,
        )
        self._append_entry(entry)
        return entry.id

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry = ModelChangeEntry(
            type="model_change",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            provider=provider,
            modelId=model_id,
        )
        self._append_entry(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Optional[Dict[str, Any]] = None,
        from_hook: Optional[bool] = None,
    ) -> str:
        entry = CompactionEntry(
            type="compaction",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            summary=summary,
            firstKeptEntryId=first_kept_entry_id,
            tokensBefore=tokens_before,
            details=details,
            fromHook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        from_hook: Optional[bool] = None,
    ) -> str:
        entry = BranchSummaryEntry(
            type="branch_summary",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            fromId=from_id,
            summary=summary,
            details=details,
            fromHook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_entry(self, custom_type: str, data: Optional[Dict[str, Any]] = None) -> str:
        entry = CustomEntry(
            type="custom",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            customType=custom_type,
            data=data,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_message(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        entry = CustomMessageEntry(
            type="custom_message",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            customType=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._append_entry(entry)
        return entry.id

    def append_label_change(self, target_id: str, label: Optional[str]) -> str:
        entry = LabelEntry(
            type="label",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            targetId=target_id,
            label=label,
        )
        self._append_entry(entry)
        return entry.id

    def append_session_info(self, name: Optional[str]) -> str:
        trimmed = name.strip() if isinstance(name, str) else None
        if trimmed == "":
            trimmed = None
        entry = SessionInfoEntry(
            type="session_info",
            id=_generate_id(self._by_id.keys()),
            parentId=self._leaf_id,
            timestamp=_now_iso(),
            name=trimmed,
        )
        self._append_entry(entry)
        return entry.id

    def get_branch(self, entry_id: Optional[str] = None) -> List[SessionEntryType]:
        if entry_id is None:
            entry_id = self._leaf_id
        if entry_id is None:
            return []
        current = self._by_id.get(entry_id)
        if current is None:
            return []
        path: List[SessionEntryType] = []
        while current:
            path.insert(0, current)
            current = self._by_id.get(current.parentId) if current.parentId else None
        return path

    def get_tree(self) -> List[SessionTreeNode]:
        entries = self.get_entries()
        node_map: Dict[str, SessionTreeNode] = {}
        roots: List[SessionTreeNode] = []

        for entry in entries:
            node_map[entry.id] = SessionTreeNode(
                entry=entry, children=[], label=self._labels_by_id.get(entry.id)
            )

        for entry in entries:
            node = node_map[entry.id]
            if entry.parentId is None or entry.parentId == entry.id:
                roots.append(node)
            else:
                parent = node_map.get(entry.parentId)
                if parent:
                    parent.children.append(node)
                else:
                    roots.append(node)

        stack = list(roots)
        while stack:
            node = stack.pop()
            node.children.sort(key=lambda n: n.entry.timestamp)
            stack.extend(node.children)
        return roots

    def branch(self, branch_from_id: str) -> None:
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: Optional[str],
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        from_hook: Optional[bool] = None,
    ) -> str:
        if branch_from_id is not None and branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        entry = BranchSummaryEntry(
            type="branch_summary",
            id=_generate_id(self._by_id.keys()),
            parentId=branch_from_id,
            timestamp=_now_iso(),
            fromId=branch_from_id or "root",
            summary=summary,
            details=details,
            fromHook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    def create_branched_session(self, leaf_id: str) -> Optional[str]:
        path = self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")

        path_without_labels = [entry for entry in path if not isinstance(entry, LabelEntry)]
        new_session_id = uuid4().hex
        timestamp = _now_iso()
        header = SessionHeader(
            type="session",
            id=new_session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            version=CURRENT_SESSION_VERSION,
            parentSession=self._session_file if self._persist else None,
        )

        path_entry_ids = {entry.id for entry in path_without_labels}
        labels_to_write = [
            {"targetId": target_id, "label": label}
            for target_id, label in self._labels_by_id.items()
            if target_id in path_entry_ids
        ]

        label_entries: List[LabelEntry] = []
        parent_id = path_without_labels[-1].id if path_without_labels else None
        for item in labels_to_write:
            label_entry = LabelEntry(
                type="label",
                id=_generate_id(path_entry_ids | {e.id for e in label_entries}),
                parentId=parent_id,
                timestamp=_now_iso(),
                targetId=item["targetId"],
                label=item["label"],
            )
            label_entries.append(label_entry)
            parent_id = label_entry.id

        self._file_entries = [header, *path_without_labels, *label_entries]
        self._session_id = new_session_id
        self._build_index()

        if not self._persist:
            return None

        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_file = str(Path(self._session_dir) / f"{file_timestamp}_{new_session_id}.jsonl")
        self._session_file = new_file
        self._rewrite_file()
        return new_file

    def load_messages(self) -> List[Message]:
        messages: List[Message] = []
        for entry in self.get_entries():
            if not isinstance(entry, SessionMessageEntry):
                continue
            payload = entry.message
            role = payload.get("role")
            if role == "user":
                messages.append(UserMessage.model_validate(payload))
            elif role == "assistant":
                messages.append(AssistantMessage.model_validate(payload))
            elif role == "toolResult":
                messages.append(ToolResultMessage.model_validate(payload))
        return messages

    def build_session_context(self) -> SessionContext:
        return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

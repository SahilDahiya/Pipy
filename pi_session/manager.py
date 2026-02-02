"""JSONL session persistence for pi-python with tree metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pi_ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage

from .tree import SessionTree

CURRENT_SESSION_VERSION = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return uuid4().hex[:8]


@dataclass
class SessionHeader:
    session_id: str
    timestamp: str
    cwd: str
    parent_session: Optional[str] = None
    version: int = CURRENT_SESSION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": "session",
            "version": self.version,
            "id": self.session_id,
            "timestamp": self.timestamp,
            "cwd": self.cwd,
        }
        if self.parent_session:
            data["parentSession"] = self.parent_session
        return data


@dataclass
class SessionEntry:
    entry_type: str
    entry_id: str
    parent_id: Optional[str]
    timestamp: str
    message: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.entry_type,
            "id": self.entry_id,
            "parentId": self.parent_id,
            "timestamp": self.timestamp,
        }
        if self.entry_type == "message" and self.message is not None:
            data["message"] = self.message
        if self.payload:
            data.update(self.payload)
        return data


class SessionManager:
    def __init__(self, path: str, *, cwd: Optional[str] = None, parent_session: Optional[str] = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[SessionEntry] = []
        self._entries_by_id: Dict[str, SessionEntry] = {}
        self._tree = SessionTree()
        self._leaf_id: Optional[str] = None
        self._header: SessionHeader

        if self._path.exists():
            self._header = self._load_existing()
        else:
            self._header = SessionHeader(
                session_id=_generate_id(),
                timestamp=_now_iso(),
                cwd=cwd or str(Path.cwd()),
                parent_session=parent_session,
            )
            self._write_line(self._header.to_dict())

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def session_id(self) -> str:
        return self._header.session_id

    def header(self) -> SessionHeader:
        return self._header

    def get_entries(self) -> List[SessionEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> SessionEntry:
        if entry_id not in self._entries_by_id:
            raise KeyError(f"Entry not found: {entry_id}")
        return self._entries_by_id[entry_id]

    def get_tree(self) -> SessionTree:
        return self._tree

    def get_leaf_id(self) -> Optional[str]:
        return self._leaf_id

    def append_message(self, message: Message, parent_id: Optional[str] = None) -> SessionEntry:
        payload: Dict[str, Any]
        if hasattr(message, "model_dump"):
            payload = message.model_dump()  # type: ignore[assignment]
        elif isinstance(message, dict):
            payload = message
        else:
            payload = {"message": str(message)}

        return self.append_entry("message", {"message": payload}, parent_id=parent_id)

    def append_entry(
        self,
        entry_type: str,
        payload: Dict[str, Any],
        *,
        parent_id: Optional[str] = None,
    ) -> SessionEntry:
        entry_id = _generate_id()
        parent_id = parent_id if parent_id is not None else self._leaf_id
        entry = SessionEntry(
            entry_type=entry_type,
            entry_id=entry_id,
            parent_id=parent_id,
            timestamp=_now_iso(),
            message=payload.get("message") if entry_type == "message" else None,
            payload=None if entry_type == "message" else payload,
        )
        self._entries.append(entry)
        self._entries_by_id[entry_id] = entry
        self._leaf_id = entry_id
        self._add_to_tree(entry)
        self._write_line(entry.to_dict())
        return entry

    def load_messages(self) -> List[Message]:
        messages: List[Message] = []
        for entry in self._entries:
            if entry.entry_type != "message" or entry.message is None:
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

    def _load_existing(self) -> SessionHeader:
        header: Optional[SessionHeader] = None
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry_type = data.get("type")
                if entry_type == "session":
                    header = SessionHeader(
                        session_id=data.get("id", _generate_id()),
                        timestamp=data.get("timestamp", _now_iso()),
                        cwd=data.get("cwd", ""),
                        parent_session=data.get("parentSession"),
                        version=int(data.get("version", CURRENT_SESSION_VERSION)),
                    )
                    continue

                entry = SessionEntry(
                    entry_type=entry_type or "message",
                    entry_id=data.get("id", _generate_id()),
                    parent_id=data.get("parentId"),
                    timestamp=data.get("timestamp", _now_iso()),
                    message=data.get("message"),
                    payload={
                        key: value
                        for key, value in data.items()
                        if key not in {"type", "id", "parentId", "timestamp", "message"}
                    }
                    or None,
                )
                self._entries.append(entry)
                self._entries_by_id[entry.entry_id] = entry
                self._leaf_id = entry.entry_id
                self._add_to_tree(entry)

        if header is None:
            header = SessionHeader(
                session_id=_generate_id(),
                timestamp=_now_iso(),
                cwd=str(Path.cwd()),
            )
        return header

    def _add_to_tree(self, entry: SessionEntry) -> None:
        if entry.parent_id is None:
            if self._tree.root_id() is None:
                self._tree.add_root(entry.entry_id)
            else:
                root_id = self._tree.root_id()
                if root_id is not None:
                    self._tree.add_child(root_id, entry.entry_id)
        else:
            try:
                self._tree.get_node(entry.parent_id)
            except KeyError:
                if self._tree.root_id() is None:
                    self._tree.add_root(entry.parent_id)
                else:
                    root_id = self._tree.root_id()
                    if root_id is not None:
                        self._tree.add_child(root_id, entry.parent_id)
            self._tree.add_child(entry.parent_id, entry.entry_id)

    def _write_line(self, data: Dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data) + "\n")

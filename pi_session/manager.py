"""JSONL session persistence for pi-python."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pi_ai.types import Message


@dataclass
class SessionEntry:
    type: str
    payload: Dict[str, Any]


class SessionManager:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> str:
        return str(self._path)

    def append_message(self, message: Message) -> None:
        self.append_entry(SessionEntry(type="message", payload=message.model_dump()))

    def append_entry(self, entry: SessionEntry) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": entry.type, "payload": entry.payload}) + "\n")

    def read_entries(self) -> List[SessionEntry]:
        if not self._path.exists():
            return []
        entries: List[SessionEntry] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(SessionEntry(type=data.get("type", "message"), payload=data.get("payload", {})))
        return entries

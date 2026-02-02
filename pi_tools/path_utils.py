"""Path helpers for tools."""

from __future__ import annotations

from pathlib import Path


def resolve_to_cwd(path: str, cwd: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return str(p.resolve())


def resolve_read_path(path: str, cwd: str) -> str:
    return resolve_to_cwd(path, cwd)

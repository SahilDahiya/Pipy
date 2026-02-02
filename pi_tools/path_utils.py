"""Path helpers for tools."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_unicode_spaces(value: str) -> str:
    return UNICODE_SPACES.sub(" ", value)


def _try_macos_screenshot_path(path: str) -> str:
    return re.sub(r" (AM|PM)\.", f"{NARROW_NO_BREAK_SPACE}\\1.", path)


def _try_nfd_variant(path: str) -> str:
    return unicodedata.normalize("NFD", path)


def _try_curly_quote_variant(path: str) -> str:
    return path.replace("'", "\u2019")


def expand_path(path: str) -> str:
    normalized = _normalize_unicode_spaces(path)
    if normalized == "~":
        return os.path.expanduser("~")
    if normalized.startswith("~/"):
        return os.path.join(os.path.expanduser("~"), normalized[2:])
    return normalized


def resolve_to_cwd(path: str, cwd: str) -> str:
    expanded = expand_path(path)
    if os.path.isabs(expanded):
        return expanded
    return str(Path(cwd, expanded).resolve())


def resolve_read_path(path: str, cwd: str) -> str:
    resolved = resolve_to_cwd(path, cwd)
    if os.path.exists(resolved):
        return resolved

    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and os.path.exists(am_pm_variant):
        return am_pm_variant

    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and os.path.exists(nfd_variant):
        return nfd_variant

    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and os.path.exists(curly_variant):
        return curly_variant

    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and os.path.exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved

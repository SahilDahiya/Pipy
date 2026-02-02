"""Shared edit tool diff and fuzzy matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


_SMART_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201A\u201B]")
_SMART_DOUBLE_QUOTES = re.compile(r"[\u201C\u201D\u201E\u201F]")
_UNICODE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_UNICODE_SPACES = re.compile(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]")


def normalize_for_fuzzy_match(text: str) -> str:
    trimmed = "\n".join(line.rstrip() for line in text.split("\n"))
    trimmed = _SMART_SINGLE_QUOTES.sub("'", trimmed)
    trimmed = _SMART_DOUBLE_QUOTES.sub('"', trimmed)
    trimmed = _UNICODE_DASHES.sub("-", trimmed)
    trimmed = _UNICODE_SPACES.sub(" ", trimmed)
    return trimmed


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> tuple[str, Optional[int]]:
    import difflib

    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines), 1)
    line_num_width = len(str(max_line_num))

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()

    output: list[str] = []
    old_line_num = 1
    new_line_num = 1
    first_changed_line: Optional[int] = None

    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            prev_change = idx > 0 and opcodes[idx - 1][0] != "equal"
            next_change = idx < len(opcodes) - 1 and opcodes[idx + 1][0] != "equal"

            if not (prev_change or next_change):
                old_line_num += i2 - i1
                new_line_num += j2 - j1
                continue

            lines = old_lines[i1:i2]
            skip_start = 0
            skip_end = 0

            if not prev_change and len(lines) > context_lines:
                skip_start = len(lines) - context_lines
                lines = lines[skip_start:]

            if not next_change and len(lines) > context_lines:
                skip_end = len(lines) - context_lines
                lines = lines[:context_lines]

            if skip_start > 0:
                output.append(f" {' ' * line_num_width} ...")
                old_line_num += skip_start
                new_line_num += skip_start

            for line in lines:
                line_num = str(old_line_num).rjust(line_num_width)
                output.append(f" {line_num} {line}")
                old_line_num += 1
                new_line_num += 1

            if skip_end > 0:
                output.append(f" {' ' * line_num_width} ...")
                old_line_num += skip_end
                new_line_num += skip_end
            continue

        if first_changed_line is None:
            first_changed_line = new_line_num

        if tag in ("replace", "delete"):
            for line in old_lines[i1:i2]:
                line_num = str(old_line_num).rjust(line_num_width)
                output.append(f"-{line_num} {line}")
                old_line_num += 1

        if tag in ("replace", "insert"):
            for line in new_lines[j1:j2]:
                line_num = str(new_line_num).rjust(line_num_width)
                output.append(f"+{line_num} {line}")
                new_line_num += 1

    return "\n".join(output), first_changed_line

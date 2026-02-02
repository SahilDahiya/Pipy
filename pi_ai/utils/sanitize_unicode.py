"""Unicode sanitation helpers."""

from __future__ import annotations


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired UTF-16 surrogate code points."""
    output: list[str] = []
    i = 0
    while i < len(text):
        code = ord(text[i])
        if 0xD800 <= code <= 0xDBFF:
            if i + 1 < len(text):
                next_code = ord(text[i + 1])
                if 0xDC00 <= next_code <= 0xDFFF:
                    output.append(text[i])
                    output.append(text[i + 1])
                    i += 2
                    continue
            i += 1
            continue
        if 0xDC00 <= code <= 0xDFFF:
            i += 1
            continue
        output.append(text[i])
        i += 1
    return "".join(output)

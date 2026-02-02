"""PKCE helper utilities."""

from __future__ import annotations

import base64
import hashlib
import os


def _base64_url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_pkce() -> tuple[str, str]:
    verifier = _base64_url_encode(os.urandom(32))
    challenge = _base64_url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge

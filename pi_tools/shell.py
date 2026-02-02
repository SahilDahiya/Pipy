"""Shell helpers for bash tool execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


_CACHED_SHELL: Dict[str, object] | None = None


def get_shell_config() -> Dict[str, object]:
    global _CACHED_SHELL
    if _CACHED_SHELL is not None:
        return _CACHED_SHELL

    shell_path = os.environ.get("SHELL")
    if shell_path and Path(shell_path).exists():
        _CACHED_SHELL = {"shell": shell_path, "args": ["-c"]}
        return _CACHED_SHELL

    if Path("/bin/bash").exists():
        _CACHED_SHELL = {"shell": "/bin/bash", "args": ["-c"]}
        return _CACHED_SHELL

    _CACHED_SHELL = {"shell": "sh", "args": ["-c"]}
    return _CACHED_SHELL


def get_shell_env() -> Dict[str, str]:
    return dict(os.environ)


def kill_process_tree(pid: int) -> None:
    try:
        os.killpg(pid, 9)
        return
    except Exception:
        pass
    try:
        os.kill(pid, 9)
    except Exception:
        return

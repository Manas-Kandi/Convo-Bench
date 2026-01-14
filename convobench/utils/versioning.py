"""Versioning helpers."""

from __future__ import annotations

import subprocess
from typing import Optional


def get_git_commit_hash() -> Optional[str]:
    """Return current git commit hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None

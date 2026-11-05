"""Output paths: every writer creates its parent directory."""
from __future__ import annotations

from pathlib import Path


def prepared(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

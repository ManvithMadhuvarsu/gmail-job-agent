"""Atomic file writes so partial writes can't corrupt JSON/JSONL state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))


def append_jsonl(path: Path, entry: dict) -> None:
    """Append a single JSON record as one line. Best-effort durability via flush+fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

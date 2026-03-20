"""
io.py

File I/O utilities for the Model 2 training pipeline.

Provides clean, safe wrappers for common I/O operations:
  - Reading and writing JSONL files
  - Reading and writing JSON files
  - Reading and writing plain text files
  - Safe directory creation
  - Atomic-style file writing (write to tmp, then rename)

All functions use pathlib.Path internally.
"""

from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory (and all parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    """
    Read a JSONL file and return a list of parsed objects.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dicts (skips blank lines).

    Raises:
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def write_jsonl(records: list[Any], path: str | Path, atomic: bool = True) -> Path:
    """
    Write a list of objects to a JSONL file.

    Args:
        records: List of JSON-serializable objects.
        path: Output file path.
        atomic: If True, write to a temp file first then rename (safer).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    if atomic:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
    else:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug(f"Wrote {len(records)} records → {path}")
    return path


def read_json(path: str | Path) -> dict | list:
    """
    Read a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object (dict or list).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """
    Write an object to a JSON file.

    Args:
        obj: JSON-serializable object.
        path: Output file path.
        indent: JSON indentation level.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Wrote JSON → {path}")
    return path


def read_text(path: str | Path) -> str:
    """Read a plain text file and return its content as a string."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text(text: str, path: str | Path) -> Path:
    """Write a string to a plain text file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    logger.debug(f"Wrote text → {path}")
    return path

"""Frozen notebooks: SHA-256 lock file and verification."""

from __future__ import annotations

import hashlib
import json
from .catalog import all_notebooks, notebook_path, stem_of
from .config import LOCK_FILE, ROOT


def serialize(notebook: dict) -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_lock() -> dict:
    if not LOCK_FILE.exists():
        return {}
    return json.loads(LOCK_FILE.read_text(encoding="utf-8"))["notebooks"]


def save_lock(lock: dict) -> None:
    data = {"_comment": "SHA-256 of each notebook that has a recorded test run. tools/build_notebooks.py never "
                        "overwrites these; see its docstring. Edit only with --freeze / --promote.",
            "notebooks": dict(sorted(lock.items()))}
    LOCK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def locked_problems(lock: dict) -> list:
    problems = []
    for key, _, model in all_notebooks():
        stem = stem_of(key, model)
        if stem not in lock:
            continue
        path = notebook_path(key, model)
        if not path.exists():
            problems.append(f"{path.relative_to(ROOT)}: locked but missing")
        elif digest(path.read_text(encoding="utf-8")) != lock[stem]:
            problems.append(f"{path.relative_to(ROOT)}: changed since it was locked")
    return problems

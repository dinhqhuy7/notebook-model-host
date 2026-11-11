"""The commands: build everything, freeze, promote."""

from __future__ import annotations

import json
import sys
from .catalog import all_notebooks, find_notebook, model_dir, notebook_path, pending_update, stem_of, update_name, update_path
from .config import BANNER_ID, ROOT, TYPES, UPDATE_SUFFIX
from .kinds import MODELS, TESTED
from .lock import digest, load_lock, locked_problems, save_lock, serialize
from .readmes import model_readme, overview_readme, type_readme
from .template import build


def read_proxy() -> str:
    proxy_src = (ROOT / "proxy" / "proxy.py").read_text()
    assert "'''" not in proxy_src, "proxy.py must not contain ''' (it is embedded in r'''...''')"
    return proxy_src


def freeze(names: list) -> None:
    lock = load_lock()
    for name in names:
        key, _, model = find_notebook(name)
        if (key, model["slug"]) not in TESTED:
            sys.exit(f"{name}: no TESTED entry; record the test run first")
        path = notebook_path(key, model)
        lock[name] = digest(path.read_text(encoding="utf-8"))
        print("locked", path.relative_to(ROOT))
    save_lock(lock)


def promote(names: list) -> None:
    """Replace the official notebook by its tested update (delete the old one, keep the official name)."""
    proxy_src = read_proxy()
    lock = load_lock()
    for name in names:
        key, platform, model = find_notebook(name)
        update = update_path(key, model)
        if not update.exists():
            sys.exit(f"{name}: no update notebook {update.name}")
        if (key, model["slug"]) not in TESTED:
            sys.exit(f"{name}: no TESTED entry; record the test run of the update first")
        notebook = json.loads(update.read_text(encoding="utf-8"))
        notebook["cells"] = [c for c in notebook["cells"] if c.get("id") != BANNER_ID]
        tested = serialize(notebook).replace(UPDATE_SUFFIX + ".ipynb", ".ipynb")
        wanted = serialize(build(key, platform, model, proxy_src))
        if tested != wanted:
            sys.exit(f"{name}: the update differs from what the template builds now, so the tested content would "
                     "not be what is promoted. Regenerate the update and test it again")
        notebook_path(key, model).write_text(wanted, encoding="utf-8")
        update.unlink()
        lock[name] = digest(wanted)
        print(f"promoted {update.name} -> {name}.ipynb (old notebook replaced, now locked)")
    save_lock(lock)


def build_all() -> None:
    proxy_src = read_proxy()
    lock = load_lock()
    problems = locked_problems(lock)
    if problems:
        sys.exit("locked notebooks were modified, nothing written:\n  " + "\n  ".join(problems))
    for key, platform, model in all_notebooks():
        model_dir(model).mkdir(parents=True, exist_ok=True)
        stem = stem_of(key, model)
        text = serialize(build(key, platform, model, proxy_src))
        if stem in lock and digest(text) != lock[stem]:
            # tested notebook: leave it alone; the new content goes to an update file marked as not tested
            path = update_path(key, model)
            update_text = serialize(build(key, platform, model, proxy_src, update=True))
            if path.exists() and path.read_text(encoding="utf-8") != update_text:
                print(f"WARNING: {path.relative_to(ROOT)} is replaced by newer content (it was not promoted)")
            path.write_text(update_text, encoding="utf-8")
            print(f"locked {stem}.ipynb kept; wrote UPDATE (not tested) {path.relative_to(ROOT)}")
            continue
        path = notebook_path(key, model)
        path.write_text(text, encoding="utf-8")
        print("wrote", path.relative_to(ROOT))
        if pending_update(key, model):
            print(f"note: {update_name(key, model)} is now identical to the official notebook; delete it")
    for vi in (False, True):
        sfx = "README.vi.md" if vi else "README.md"
        for model in MODELS:
            path = model_dir(model) / sfx
            path.write_text(model_readme(model, vi), encoding="utf-8")
            print("wrote", path.relative_to(ROOT))
        for kind in TYPES:
            if any(m["kind"] == kind for m in MODELS):
                path = ROOT / "notebooks" / kind / sfx
                path.write_text(type_readme(kind, vi), encoding="utf-8")
                print("wrote", path.relative_to(ROOT))
        path = ROOT / "notebooks" / sfx
        path.write_text(overview_readme(vi), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))

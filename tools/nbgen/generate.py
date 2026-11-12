"""Generate notebooks for the models and platforms you pick, into a folder of your own."""

from __future__ import annotations

import sys
from pathlib import Path
from .catalog import all_notebooks, available, min_gpu_mib, notebook_name
from .config import PLATFORMS, ROOT
from .kinds import MODELS
from .template import build, serialize

SAMPLES = ROOT / "notebooks"  # the ready-made sample notebooks of this repository: never written by this tool


def read_proxy() -> str:
    proxy_src = (ROOT / "proxy" / "proxy.py").read_text(encoding="utf-8")
    assert "'''" not in proxy_src, "proxy.py must not contain ''' (it is embedded in r'''...''')"
    return proxy_src


def check_out_dir(out: Path) -> Path:
    """The output folder must not be, or be inside, the sample notebooks folder."""
    out = out.resolve()
    if out == SAMPLES or SAMPLES in out.parents:
        sys.exit(f"{out} is inside the sample notebooks ({SAMPLES}); choose another --out folder")
    return out


def list_models() -> list:
    lines = [f"{'slug':26} {'model':26} {'type':11} {'min GPU memory':15} platforms"]
    for model in MODELS:
        platforms = ", ".join(k for k in PLATFORMS if available(model, k))
        memory = f"~{-(-min_gpu_mib(model) // 1024)} GB"
        lines.append(f"{model['slug']:26} {model['served_name']:26} {model['kind']:11} {memory:15} {platforms}")
    return lines


def generate(out: Path, slugs: list | None = None, platforms: list | None = None) -> list:
    """Write <out>/<platform>_<slug>.ipynb for every chosen model and platform; returns the paths written."""
    out = check_out_dir(out)
    known = {m["slug"] for m in MODELS}
    unknown = sorted(set(slugs or []) - known)
    if unknown:
        sys.exit(f"unknown model: {', '.join(unknown)}. Run with --list to see the models")
    proxy_src = read_proxy()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key, platform, model in all_notebooks():
        if (slugs and model["slug"] not in slugs) or (platforms and key not in platforms):
            continue
        path = out / notebook_name(key, model)
        path.write_text(serialize(build(key, platform, model, proxy_src)), encoding="utf-8")
        written.append(path)
    return written

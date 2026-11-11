"""Models and platforms: what exists, what fits."""

from __future__ import annotations

import math
import sys
from .config import GPU_MEMORY, MIN_KV_GIB, OVERHEAD_GIB, PLATFORMS
from .kinds import MODELS


def kv_gib(model: dict, count: int, mib: int) -> float:
    """Estimated memory left for the KV cache when running the model on `count` GPUs of `mib` MiB."""
    return count * (mib / 1024 * GPU_MEMORY - OVERHEAD_GIB) - model["weights_gib"] - model["extra_gib"]


def min_gpu_mib(model: dict) -> int:
    return math.ceil((model["weights_gib"] + model["extra_gib"] + OVERHEAD_GIB + MIN_KV_GIB) / GPU_MEMORY * 1024)


def runnable(model: dict, gpu: tuple) -> bool:
    _, count, mib, cap = gpu
    return cap >= 7.0 and kv_gib(model, count, mib) >= MIN_KV_GIB


def gpu_table(model: dict, platform: dict) -> str:
    rows = ["| GPU | Runs? | Memory left for context |", "|---|---|---|"]
    for gpu in platform["gpus"]:
        name, count, mib, cap = gpu
        if cap < 7.0:
            rows.append(f"| {name} | ✗ not supported by vLLM | |")
        elif runnable(model, gpu):
            rows.append(f"| {name} | ✓ | ~{kv_gib(model, count, mib):.0f} GiB |")
        else:
            rows.append(f"| {name} | ✗ not enough memory | |")
    return "\n".join(rows)


def notebook_name(key: str, model: dict) -> str:
    return f"{key}_{model['slug']}.ipynb"


def stem_of(key: str, model: dict) -> str:
    return notebook_name(key, model)[: -len(".ipynb")]


def available(model: dict, key: str) -> bool:
    """Whether the model gets a notebook on this platform (default: every platform)."""
    return key in model.get("platforms", PLATFORMS)


def all_notebooks():
    """Every (platform key, platform, model) that gets a notebook."""
    for key, platform in PLATFORMS.items():
        for model in (m for m in MODELS if available(m, key)):
            yield key, platform, model


def find_notebook(name: str):
    """'colab_qwen3_8b' -> (key, platform, model)."""
    for key, platform, model in all_notebooks():
        if stem_of(key, model) == name:
            return key, platform, model
    sys.exit(f"unknown notebook: {name}")

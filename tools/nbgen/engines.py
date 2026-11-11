"""Per-engine differences in the notebook template.

A model runs on plain vLLM unless it sets "engine": "vllm-omni" (models that also hear or speak, whose answer
needs more than one stage). The default engine leaves the template exactly as it is.
"""

from __future__ import annotations

from .config import VLLM_VERSION


def vllm_version(model: dict) -> str:
    return model.get("vllm_version", VLLM_VERSION)


def pip_packages(model: dict) -> str:
    """The packages of the install cell, quoted for `!pip install -q`."""
    packages = [f'"vllm=={vllm_version(model)}"']
    if model.get("engine") == "vllm-omni":
        packages.append(f'"vllm-omni=={model["omni_version"]}"')
    return " ".join(packages)


def _replace(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"template changed, cannot patch: {old[:60]!r}"
    return text.replace(old, new)


def _between(text: str, start: str, end: str, new: str) -> str:
    """Replace everything from `start` to the end of `end` (both included)."""
    a = text.index(start)
    b = text.index(end, a) + len(end)
    return text[:a] + new + text[b:]


_OMNI_SERVE = '''\
if len(rows) > 2:
    raise RuntimeError("vLLM-Omni layouts here cover one or two GPUs")
scale = GPU_MEMORY / 0.90
if len(rows) == 1:
    # one GPU: the three stages share it, each with a budget in GiB (Thinker, Talker, Code2Wav)
    share = [round(min(gib / (GPU_MIB / 1024) * scale, 0.9), 3) for gib in (12.5, 3.5, 4.0)]
    stages = {str(i): {"devices": "0", "gpu_memory_utilization": s} for i, s in enumerate(share)}
else:
    # two GPUs: the Thinker has the first, the Talker and Code2Wav share the second
    stages = {"0": {"devices": "0", "gpu_memory_utilization": round(0.8 * scale, 3)},
              "1": {"devices": "1", "gpu_memory_utilization": round(0.5 * scale, 3)},
              "2": {"devices": "1", "gpu_memory_utilization": round(0.3 * scale, 3)}}
if MAX_MODEL_LEN != "auto":
    stages["0"]["max_model_len"] = int(MAX_MODEL_LEN)
if ENFORCE_EAGER:
    for stage in ("0", "1"):
        stages[stage]["enforce_eager"] = True
cmd = [
    "vllm", "serve", MODEL_DIR, "--omni", "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1", "--port", str(PORT), "--api-key", API_KEY,
    "--dtype", DTYPE, "--disable-uvicorn-access-log",
    "--stage-overrides", json.dumps(stages),
    {{SERVE_FLAGS}},
]
'''


def patch_omni(cells: list) -> list:
    """The template with the changes for a model served by vLLM-Omni."""
    out = []
    for kind, text in cells:
        if text.startswith("# Host {{SERVED_NAME}}"):
            text = _replace(text, "with vLLM and exposes it", "with vLLM-Omni and exposes it")
        elif text.startswith("## 3. Install vLLM"):
            text = _replace(text, "## 3. Install vLLM", "## 3. Install vLLM and vLLM-Omni")
            text = _replace(text, "Takes a few minutes.", "Takes several minutes: vLLM-Omni pulls many extra packages.")
        elif text.startswith("## 5. Start the server"):
            text = _replace(
                text, "The first start takes a few minutes.",
                "The first start takes several minutes. With one GPU the three stages share it; with two GPUs the\n"
                "Thinker has the first and the other two stages the second.")
        elif text.startswith("import json, time, urllib.request"):
            text = _between(text, "cmd = [\n", 'cmd.append("--enforce-eager")\n', _OMNI_SERVE)
            text = _replace(text, '["data"][0]["max_model_len"]', '["data"][0].get("max_model_len", "unknown")')
        elif text.startswith("started = time.time()"):
            text = _replace(text, '"max_tokens": 256, "ignore_eos": True,',
                            '"max_tokens": 256, "ignore_eos": True, "modalities": ["text"],')
        elif text.startswith("import re\n\nCLOUDFLARED"):
            text = _replace(
                text, '"max_tokens": 512}}\'\n\'\'\')',
                '"max_tokens": 512}}\'\n\n'
                'Speech: add "modalities": ["text", "audio"] for the answer as speech too (WAV, base64, in the second\n'
                'choice) and "speaker": "Chelsie" or "Ethan". An answer with audio cannot stream: keep it short.\n\'\'\')')
        out.append((kind, text))
    return out


def cells_for(model: dict, cells: list) -> list:
    """The template cells for this model's engine."""
    if model.get("engine") == "vllm-omni":
        return patch_omni(cells)
    return cells

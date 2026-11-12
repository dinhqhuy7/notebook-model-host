"""Per-engine and per-task differences in the notebook template.

A model runs on plain vLLM unless it sets "engine": "vllm-omni" (models that also hear or speak, whose answer
needs more than one stage). A model with "task": "rerank" scores text instead of writing it. A model whose proxy profile
is not "chat" gets the proxy with that profile. The default leaves the template exactly as it is.
"""

from __future__ import annotations

import json
from .config import VLLM_VERSION, profile_of


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


# A path of each profile that must need the key (a POST without key gets 401) to check the proxy in step 7.
_PROBE = {"retrieval": "/rerank", "audio": "/v1/audio/transcriptions"}


def patch_profile(cells: list, profile: str) -> list:
    """The proxy of a model that is not a chat model: it runs with its profile and the check uses one of its paths."""
    out = []
    for kind, text in cells:
        if text.startswith("import sys, urllib.error"):
            text = _replace(text, "str(PROXY_PORT), str(PORT), API_KEY],",
                            f'str(PROXY_PORT), str(PORT), API_KEY, "{profile}"],')
            text = _replace(text, 'for path, expected in [("/v1/chat/completions", 401), ("/invocations", 404)]:',
                            f'for path, expected in [("{_PROBE[profile]}", 401), ("/invocations", 404),\n'
                            '                       ("/v1/chat/completions", 404)]:')
        out.append((kind, text))
    return out


_RERANK_SPEED = '''\
def pairs(label, n):  # distinct documents (~200 tokens each) so that the prefix cache does not answer for them
    return [f"{label}{i} " + "word " * 200 for i in range(n)]


call(LOCAL, "/rerank", {"model": SERVED_NAME, "query": ".", "documents": pairs("warm", 4)})  # the first request is slower
started = time.time()
reply = call(LOCAL, "/rerank", {"model": SERVED_NAME, "query": ".", "documents": pairs("doc", 64)})
elapsed = time.time() - started
DOCS_PER_SECOND = 64 / elapsed
print(f"~{DOCS_PER_SECOND:.0f} documents/second ({reply['usage']['total_tokens'] / elapsed:.0f} tokens/second, 200-token documents)")'''

_RERANK_INFO = '''\
print(f\'\'\'
Base URL : {PUBLIC}
API key  : {API_KEY}
Model    : {SERVED_NAME}
Context  : {CONTEXT} tokens per query + document
Speed    : ~{DOCS_PER_SECOND:.0f} documents/second (200-token documents); a request must finish within 100 seconds

curl {PUBLIC}/rerank \\\\
  -H "Authorization: Bearer {API_KEY}" -H "Content-Type: application/json" \\\\
  -d \'{{"model": "{SERVED_NAME}", "query": "What is the capital of France?", "documents": ["Paris is the capital of France.", "Berlin is a city in Germany."]}}\'

Also POST /score (scores in the order you send). Both accept "instruction" to describe your task.
\'\'\')'''


def notes_for(model: dict, key: str) -> str:
    """The notes of the notebook intro. "notes_by_platform" pins the text of a platform whose notebook is already tested
    and locked, so improving `notes` (the README text, the other platforms) does not change that notebook."""
    return model.get("notes_by_platform", {}).get(key, model["notes"])


def patch_rerank(cells: list, model: dict, key: str) -> list:
    """The template for a model that scores text: no chat request, a rerank speed test, the prompt template file."""
    out = []
    for kind, text in cells:
        if text.startswith("# Host {{SERVED_NAME}}"):
            text = _replace(text, "as a key-protected, OpenAI-compatible API.",
                            "as a key-protected reranking API (`/rerank` and `/score`).")
        elif text.startswith("## 1. Check the GPU"):
            if key in model.get("one_gpu_on", ()):
                text = _replace(text, "and splits the model across all GPUs\nwhen there are several.",
                                "and uses one GPU: the model is small.")
        elif text.startswith("import os, secrets") or text.startswith("import socket, subprocess"):
            if "TP = len(rows)" in text and key in model.get("one_gpu_on", ()):
                text = _replace(text, "TP = len(rows) if len(rows) in (1, 2, 4, 8) else 1", "TP = 1  # a small model: one GPU")
                text = _replace(text, "GPU_MIB = min(int(r[1]) for r in rows)", "GPU_MIB = int(rows[0][1])")
        elif text.startswith("import json, time, urllib.request"):
            from .kinds.retrieval import QWEN3_RERANKER_TEMPLATE
            text = _replace(text, 'LOG = f"{WORK}/server.log"\n',
                            'TEMPLATE_FILE = f"{WORK}/qwen3_reranker.jinja"  # the prompt the reranker was trained with\n'
                            "with open(TEMPLATE_FILE, \"w\") as f:\n"
                            "    f.write(r\'\'\'" + QWEN3_RERANKER_TEMPLATE + "\'\'\')\n\n"
                            'LOG = f"{WORK}/server.log"\n')
            text = _replace(text, "    {{SERVE_FLAGS}},\n]", '    {{SERVE_FLAGS}},\n    "--chat-template", TEMPLATE_FILE,\n]')
            text = _replace(text, '["data"][0]["max_model_len"]', '["data"][0].get("max_model_len", "unknown")')
        elif text.startswith("## 6. Measure speed"):
            text = _replace(text, "One request with no content generates 256 tokens (`ignore_eos`) to measure speed. Nothing is printed\nfrom the answer.",
                            "One query is scored against 64 documents of ~200 tokens to measure speed (a short request before it\n"
                            "warms the server up). Nothing is printed from the scores.")
        elif text.startswith("started = time.time()"):
            text = _RERANK_SPEED
        elif text.startswith("## 8. Expose to the Internet"):
            text = _replace(text, "Requests without\nstreaming are cut after 100 seconds (error 524); with `stream: true` the answer arrives as it is\n"
                            "generated and is not cut (tested: 124 seconds, 9000 tokens).",
                            "A request that takes more\nthan 100 seconds is cut (error 524): scoring is fast, but send very large lists in several requests.")
        elif text.startswith("import re\n\nCLOUDFLARED"):
            text = _between(text, "print(f\'\'\'\nBase URL", "\'\'\')", _RERANK_INFO.rstrip("\n"))
        out.append((kind, text))
    return out


def cells_for(model: dict, cells: list, key: str) -> list:
    """The template cells for this model's engine, task and proxy profile."""
    if model.get("engine") == "vllm-omni":
        cells = patch_omni(cells)
    if model.get("task") == "rerank":
        cells = patch_rerank(cells, model, key)
    if profile_of(model) != "chat":
        cells = patch_profile(cells, profile_of(model))
    return cells

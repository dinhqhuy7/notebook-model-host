"""Text models: model families and models.

Add a model: a line in MODELS (its "kind" must be "text"), then
`python tools/build_notebooks.py --list` shows it.
"""

# Model families: serving flags (enable features only) and what to tell the reader about them.
QWEN3 = {
    "kind": "text",
    "input": ("text", "văn bản"),
    "context": 40960,
    "flags": ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
    "flag_notes": """\
- `--reasoning-parser qwen3`: thinking is returned in the `reasoning` field, separate from `content`.
- `--enable-auto-tool-choice --tool-call-parser hermes`: function calls are returned in `tool_calls`.""",
    "sampling": "requests that omit sampling parameters get the model's `generation_config.json`",
    "notes": "",
    "notes_vi": "",
    "extra_gib": 0.0,
}

MODELS = [
    {"slug": "qwen3_0_6b", "model_id": "Qwen/Qwen3-0.6B", "served_name": "Qwen3-0.6B",
     "weights_gib": 1.40, "download_gb": 2, **QWEN3},
    {"slug": "qwen3_1_7b", "model_id": "Qwen/Qwen3-1.7B", "served_name": "Qwen3-1.7B",
     "weights_gib": 3.78, "download_gb": 4, **QWEN3},
    {"slug": "qwen3_4b", "model_id": "Qwen/Qwen3-4B", "served_name": "Qwen3-4B",
     "weights_gib": 7.49, "download_gb": 8, **QWEN3},
    {"slug": "qwen3_8b", "model_id": "Qwen/Qwen3-8B", "served_name": "Qwen3-8B",
     "weights_gib": 15.26, "download_gb": 16, **QWEN3},
    {"slug": "qwen3_14b", "model_id": "Qwen/Qwen3-14B", "served_name": "Qwen3-14B",
     "weights_gib": 27.51, "download_gb": 30, **QWEN3, "platforms": ("colab",),
     "unavailable": {"kaggle": ("needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total",
                                "cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB")}},
]

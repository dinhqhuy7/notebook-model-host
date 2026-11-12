"""Language models: model families and models (chat, thinking, tools, code writing and completion).

Add a model: a line in MODELS (its "kind" must be "language"), then `python tools/build_notebooks.py --list` shows it.
"""

# Model families: serving flags (enable features only) and what to tell the reader about them.
QWEN3 = {
    "kind": "language",
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

# Qwen2.5-Coder (Qwen2ForCausalLM, 36 layers, 2 KV heads): the KV cache costs 36 KiB per token, so the full 32 768-token
# context needs only ~1.1 GiB and fits on every listed GPU. License: Qwen Research License, not Apache 2.0.
_CODER_NOTES = {
    "kind": "language",
    "input": ("text", "văn bản"),
    "context": 32768,
    "extra_gib": 0.0,
}
QWEN2_5_CODER_INSTRUCT = {
    **_CODER_NOTES,
    "flags": ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
    "flag_notes": """\
- `--enable-auto-tool-choice --tool-call-parser hermes`: function calls are returned in `tool_calls`.
- No reasoning parser: the model never thinks.""",
    "sampling": "requests that omit sampling parameters get the model's `generation_config.json`",
    "notes": "Code model (Instruct): chat, tool calling and code writing, fixing and explaining. The KV cache costs "
             "36 KiB per token, so the full 32 768-token context fits on every listed GPU. It never thinks. If you send "
             "no system message its chat template adds \"You are Qwen, created by Alibaba Cloud. You are a helpful "
             "assistant.\"; send your own to replace it. License: Qwen Research License, not Apache 2.0. `float16` on "
             "T4 has not been tested with this model yet.",
    "notes_vi": "Model code (Instruct): chat, gọi hàm, viết, sửa và giải thích code. KV cache tốn 36 KiB mỗi token, nên "
                "đủ context 32 768 token trên mọi GPU trong bảng. Không bao giờ suy nghĩ. Nếu bạn không gửi system "
                "message, chat template tự thêm \"You are Qwen, created by Alibaba Cloud. You are a helpful "
                "assistant.\"; hãy gửi system riêng để thay thế. Giấy phép: Qwen Research License, không phải "
                "Apache 2.0. `float16` trên T4 chưa thử với model này.",
}
QWEN2_5_CODER_BASE = {
    **_CODER_NOTES,
    "flags": [],
    "flag_notes": """\
- No tool parser and no reasoning parser: this is the base model, for `/v1/completions` (code completion and
  fill-in-the-middle). It is not tuned for chat or tools.""",
    "sampling": "requests that omit sampling parameters get vLLM's defaults (temperature 1.0)",
    "notes": "Code model (base): code completion and fill-in-the-middle through `/v1/completions`, with the prompt "
             "`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`. It is not tuned for chat or tools (use the Instruct "
             "model for those). The KV cache costs 36 KiB per token, so the full 32 768-token context fits on every "
             "listed GPU. Always send `temperature`: without it vLLM samples at 1.0. License: Qwen Research License, "
             "not Apache 2.0. `float16` on T4 has not been tested with this model yet.",
    "notes_vi": "Model code (base): hoàn thành code và điền vào giữa qua `/v1/completions`, với prompt "
                "`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`. Không tinh chỉnh để chat hay gọi hàm (dùng bản "
                "Instruct cho việc đó). KV cache tốn 36 KiB mỗi token, nên đủ context 32 768 token trên mọi GPU trong "
                "bảng. Luôn gửi `temperature`: không gửi thì vLLM lấy mẫu ở 1.0. Giấy phép: Qwen Research License, "
                "không phải Apache 2.0. `float16` trên T4 chưa thử với model này.",
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
    {"slug": "qwen2_5_coder_3b_instruct", "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
     "served_name": "Qwen2.5-Coder-3B-Instruct", "weights_gib": 5.75, "download_gb": 7, **QWEN2_5_CODER_INSTRUCT},
    {"slug": "qwen2_5_coder_3b", "model_id": "Qwen/Qwen2.5-Coder-3B", "served_name": "Qwen2.5-Coder-3B",
     "weights_gib": 5.75, "download_gb": 7, **QWEN2_5_CODER_BASE},
]

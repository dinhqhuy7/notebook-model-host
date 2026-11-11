"""Vision-language models: model families and models.

Add a model: a line in MODELS (its "kind" must be "vision-language"), then
`python tools/build_notebooks.py --list` shows it.
"""

QWEN3_5 = {
    "kind": "vision-language",
    "input": ("text, images, video", "văn bản, ảnh, video"),
    "context": 262144,
    "flags": ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
              "--allowed-media-domains", "none.invalid", "--max-num-seqs", "128"],
    "flag_notes": """\
- `--reasoning-parser qwen3`: thinking is returned in the `reasoning` field, separate from `content`.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`: function calls are returned in `tool_calls`.
- `--allowed-media-domains none.invalid`: the server never downloads image/video URLs itself;
  callers send media from their machine as base64 `data:` URLs.
- `--max-num-seqs 128`: at most 128 requests are generated at once. Each one needs its own
  linear-attention state, and small GPUs (Kaggle 2 × T4) fit only ~180 of them, below vLLM's default 256.""",
    "sampling": "the model ships no `generation_config.json`: requests that omit them get vLLM's defaults, "
                "so callers should send the values Qwen recommends, see docs/api.md",
    "notes": "Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache "
             "costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the "
             "model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first "
             "start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.",
    # Image encoder, encoder cache and linear-attention state (measured: KV cache 13.2 GiB on A100 40 GB).
    "extra_gib": 3.0,
    "notes_vi": "Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache "
                "chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy "
                "`float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất "
                "~11–12 phút (biên dịch kernel): bước 5 không bị treo.",
}

# Qwen3-VL (dense attention, vision): the full 262 144 context needs ~28 GiB of KV cache, so small GPUs get less.
_VL_FLAGS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--allowed-media-domains", "none.invalid"]
_VL_NOTES = {
    "kind": "vision-language",
    "input": ("text, images, video", "văn bản, ảnh, video"),
    "context": 262144,
    "sampling": "requests that omit sampling parameters get the model's `generation_config.json`",
    "extra_gib": 1.0,  # estimate, not measured yet (the vision encoder is already inside the weights)
}
QWEN3_VL_INSTRUCT = {
    **_VL_NOTES,
    "flags": _VL_FLAGS,
    "flag_notes": """\
- `--enable-auto-tool-choice --tool-call-parser hermes`: function calls are returned in `tool_calls`.
- `--allowed-media-domains none.invalid`: the server never downloads image/video URLs itself;
  callers send media from their machine as base64 `data:` URLs.""",
    "notes": "Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full "
             "262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs "
             "vLLM lowers the context to what fits (see the memory table). **It never thinks** (no reasoning).",
    "notes_vi": "Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ "
                "context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm "
                "context cho vừa (xem bảng bộ nhớ). **Không bao giờ suy nghĩ** (không có reasoning).",
}
QWEN3_VL_THINKING = {
    **_VL_NOTES,
    "flags": ["--reasoning-parser", "qwen3", *_VL_FLAGS],
    "flag_notes": """\
- `--reasoning-parser qwen3`: thinking is returned in the `reasoning` field, separate from `content`.
""" + QWEN3_VL_INSTRUCT["flag_notes"],
    "notes": "Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full "
             "262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs "
             "vLLM lowers the context to what fits (see the memory table). **It always thinks**: the chat template "
             "has no switch, so `enable_thinking` changes nothing; give `max_tokens` room for the thinking.",
    "notes_vi": "Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ "
                "context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm "
                "context cho vừa (xem bảng bộ nhớ). **Luôn suy nghĩ**: chat template không có công tắc nên "
                "`enable_thinking` không đổi gì; hãy để `max_tokens` đủ chỗ cho phần suy nghĩ.",
}

MODELS = [
    {"slug": "qwen3_5_0_8b", "model_id": "Qwen/Qwen3.5-0.8B", "served_name": "Qwen3.5-0.8B",
     "weights_gib": 1.63, "download_gb": 2, **QWEN3_5,
     "notes": "Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the "
              "full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send "
              "`chat_template_kwargs: {\"enable_thinking\": true}` to turn it on. On T4 the model runs in "
              "`float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel "
              "compilation): step 5 is not stuck.",
     "notes_vi": "Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ "
                 "context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi "
                 "`chat_template_kwargs: {\"enable_thinking\": true}` để bật. Trên T4 model chạy `float16` thay "
                 "vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo."},
    {"slug": "qwen3_5_2b", "model_id": "Qwen/Qwen3.5-2B", "served_name": "Qwen3.5-2B",
     "weights_gib": 4.24, "download_gb": 5, **QWEN3_5,
     "notes": "Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the "
              "full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send "
              "`chat_template_kwargs: {\"enable_thinking\": true}` to turn it on. On T4 the model runs in "
              "`float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel "
              "compilation): step 5 is not stuck.",
     "notes_vi": "Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ "
                 "context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi "
                 "`chat_template_kwargs: {\"enable_thinking\": true}` để bật. Trên T4 model chạy `float16` thay "
                 "vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo."},
    {"slug": "qwen3_5_4b", "model_id": "Qwen/Qwen3.5-4B", "served_name": "Qwen3.5-4B",
     "weights_gib": 8.68, "download_gb": 9, **QWEN3_5},
    {"slug": "qwen3_5_9b", "model_id": "Qwen/Qwen3.5-9B", "served_name": "Qwen3.5-9B",
     "weights_gib": 17.98, "download_gb": 20, **QWEN3_5},
    {"slug": "qwen3_vl_2b_instruct", "model_id": "Qwen/Qwen3-VL-2B-Instruct", "served_name": "Qwen3-VL-2B-Instruct",
     "weights_gib": 3.96, "download_gb": 5, **QWEN3_VL_INSTRUCT},
    {"slug": "qwen3_vl_2b_thinking", "model_id": "Qwen/Qwen3-VL-2B-Thinking", "served_name": "Qwen3-VL-2B-Thinking",
     "weights_gib": 3.96, "download_gb": 5, **QWEN3_VL_THINKING},
]

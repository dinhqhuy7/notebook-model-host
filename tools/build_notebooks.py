"""Generate notebooks/<platform>/<model>.ipynb from one template (every PLATFORMS × MODELS pair).

    python tools/build_notebooks.py

Also writes notebooks/README(.vi).md (overview of every model × platform) and
notebooks/<platform>/README(.vi).md (notebook specs, GPU table and test results).

Add a model: add an entry to MODELS and rerun; the GPU table and minimum memory are computed. A model that
cannot run on a platform sets "platforms" (where it runs) and "unavailable" (reason, English and Vietnamese).
Record a test run: add an entry to TESTED and rerun.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO = "dinhqhuy7/notebook-model-host"  # GitHub repo, used by the "Open in Colab/Kaggle" badges
VLLM_VERSION = "0.33.0"
CLOUDFLARED_VERSION = "2026.10.7"

GPU_MEMORY = 0.90   # fraction of GPU memory given to vLLM
OVERHEAD_GIB = 2.0  # per GPU: activations, CUDA graphs... (measured ~1.9 GiB on A100)
MIN_KV_GIB = 1.5    # minimum KV cache to be usable

# Model families: serving flags (enable features only) and what to tell the reader about them.
QWEN3 = {
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
QWEN3_5 = {
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
]

# Real test runs, shown in notebooks/<platform>/README*.md. Key: (platform, model slug).
# Rows: (English label, Vietnamese label, English value, Vietnamese value).
TESTED = {
    ("colab", "qwen3_5_2b"): {
        "summary": ("T4, 2026-11-11", "T4, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07", "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 1, `--max-num-seqs 128`",
             "0.33.0 / float16 / 1, `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "6.15 GiB = 530 734 tokens, 2.02 full-length requests at once (same in both runs)",
             "6,15 GiB = 530 734 token, 2,02 request dài tối đa cùng lúc (giống nhau ở cả hai lần chạy)"),
            ("GPU memory", "Bộ nhớ GPU", "weights 4.18 GiB, CUDA graphs 0.20 + 0.12 GiB",
             "trọng số 4,18 GiB, CUDA graph 0,20 + 0,12 GiB"),
            ("Time", "Thời gian",
             "server start 12 min 53 s and 11 min 50 s (two runs 8.5 hours apart), of which ~8 min is unlogged "
             "kernel compilation after torch.compile",
             "bật server 12 phút 53 giây và 11 phút 50 giây (hai lần cách nhau 8,5 tiếng), trong đó ~8 phút là biên "
             "dịch kernel không ghi log sau torch.compile"),
            ("Speed", "Tốc độ", "~300–332 tokens/second total with 8 requests at once",
             "~300–332 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "12/13 checks, twice. no/wrong key → 401; `/invocations`, `/sleep` → 404; tool calling via "
             "`qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL "
             "rejected; secret found in a 16 000-token prompt (~16 s). The thinking check failed only because this "
             "model sometimes thinks past the check's 4 000-token budget: through streaming with 12 000 tokens it "
             "answered correctly 3 times out of 3 (1 300–2 100 tokens). Base64 image: text right, the rectangle "
             "called a square",
             "12/13 phép thử, hai lần. không key/sai key → 401; `/invocations`, `/sleep` → 404; gọi hàm qua "
             "`qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ "
             "chối; tìm đúng mã trong prompt 16 000 token (~16 giây). Phép thử suy nghĩ trượt chỉ vì model này đôi "
             "khi nghĩ quá ngân sách 4 000 token của phép thử: qua stream với 12 000 token thì trả lời đúng 3/3 lần "
             "(1 300–2 100 token). Ảnh base64: đúng chữ, hình chữ nhật bị gọi là hình vuông"),
        ],
    },
    ("kaggle", "qwen3_5_2b"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`",
             "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "17.3 GiB (8.64 GiB per GPU) = 1 490 675 tokens, 5.69 full-length requests at once",
             "17,3 GiB (8,64 GiB mỗi GPU) = 1 490 675 token, 5,69 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)",
             "weights 2.12 GiB, other non-torch ~0.9 GiB, CUDA graphs 0.22 + 0.12 GiB",
             "trọng số 2,12 GiB, non-torch khác ~0,9 GiB, CUDA graph 0,22 + 0,12 GiB"),
            ("Time", "Thời gian",
             "server start ~10 min (weights 7 s, torch.compile 25 s, ~6.6 min of unlogged kernel compilation, "
             "CUDA graphs 32 s)",
             "bật server ~10 phút (trọng số 7 giây, torch.compile 25 giây, ~6,6 phút biên dịch kernel không ghi log, "
             "CUDA graph 32 giây)"),
            ("Speed", "Tốc độ", "~86 tokens/second for one request (with thinking); ~272 tokens/second total with 8 at once",
             "~86 token/giây cho một request (có suy nghĩ); ~272 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking with `enable_thinking: true`; tool "
             "calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` "
             "image URL rejected; secret found in a 16 000-token prompt (9.6 s). Base64 image: text right, the "
             "rectangle called a square",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; suy nghĩ khi gửi `enable_thinking: true`; "
             "gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh "
             "`http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (9,6 giây). Ảnh base64: đúng chữ, hình "
             "chữ nhật bị gọi là hình vuông"),
            ("Sampling parameters", "Tham số sinh",
             "a short factual question, 20 tries each: temperature 0 right 20/20; Qwen's general values (0.7/0.8, "
             "top_k 20, presence 1.5) 17/20; Qwen's text values for 0.8B/2B (1.0/1.0, top_k 20, presence 2.0) 9/20. "
             "Use a low temperature for short exact answers",
             "một câu hỏi ngắn, 20 lần mỗi cách: temperature 0 đúng 20/20; tham số Qwen chung (0.7/0.8, top_k 20, "
             "presence 1.5) 17/20; tham số Qwen cho văn bản của 0.8B/2B (1.0/1.0, top_k 20, presence 2.0) 9/20. "
             "Câu trả lời ngắn cần chính xác thì dùng temperature thấp"),
        ],
    },
    ("kaggle", "qwen3_5_0_8b"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`",
             "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "20.4 GiB (10.19 GiB per GPU) = 1 758 728 tokens, 6.71 full-length requests at once",
             "20,4 GiB (10,19 GiB mỗi GPU) = 1 758 728 token, 6,71 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)",
             "weights 0.84 GiB, other non-torch ~0.9 GiB, CUDA graphs 0.21 + 0.12 GiB",
             "trọng số 0,84 GiB, non-torch khác ~0,9 GiB, CUDA graph 0,21 + 0,12 GiB"),
            ("Time", "Thời gian",
             "server start ~11.5 min (weights 7 s, torch.compile 30 s, ~7.7 min of unlogged kernel compilation, "
             "CUDA graphs 33 s)",
             "bật server ~11,5 phút (trọng số 7 giây, torch.compile 30 giây, ~7,7 phút biên dịch kernel không ghi log, "
             "CUDA graph 33 giây)"),
            ("Speed", "Tốc độ",
             "~120 tokens/second for one request (with thinking); ~289 tokens/second total with 8 at once, slower "
             "than a single Colab T4 (~640): for a model this small, splitting over 2 GPUs costs more than it saves",
             "~120 token/giây cho một request (có suy nghĩ); ~289 token/giây tổng khi 8 request, chậm hơn Colab "
             "1 × T4 (~640): model nhỏ thế này thì chia 2 GPU tốn hơn lợi"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking with `enable_thinking: true`; tool "
             "calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` "
             "image URL rejected; secret found in a 16 000-token prompt (7.9 s, faster than a single T4). Base64 "
             "image: text and red rectangle right, the blue circle called green (same as on Colab)",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; suy nghĩ khi gửi `enable_thinking: true`; "
             "gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh "
             "`http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (7,9 giây, nhanh hơn 1 × T4). Ảnh base64: "
             "đúng chữ và hình chữ nhật đỏ, hình tròn xanh dương bị gọi là xanh lá (giống trên Colab)"),
        ],
    },
    ("colab", "qwen3_5_0_8b"): {
        "summary": ("T4, 2026-11-11", "T4, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07", "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 1, `--max-num-seqs 128`",
             "0.33.0 / float16 / 1, `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "9.11 GiB = 785 357 tokens, 3.00 full-length requests at once",
             "9,11 GiB = 785 357 token, 3,00 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU", "weights 1.63 GiB, other non-torch ~1 GiB, CUDA graphs 0.20 + 0.12 GiB",
             "trọng số 1,63 GiB, non-torch khác ~1 GiB, CUDA graph 0,20 + 0,12 GiB"),
            ("Time", "Thời gian",
             "server start ~11 min (weights 15 s, torch.compile 27 s, ~7.5 min of unlogged kernel compilation, "
             "CUDA graphs 47 s)",
             "bật server ~11 phút (trọng số 15 giây, torch.compile 27 giây, ~7,5 phút biên dịch kernel không ghi log, "
             "CUDA graph 47 giây)"),
            ("Speed", "Tốc độ", "~100 tokens/second for one request (with thinking); ~640 tokens/second total with 8 at once",
             "~100 token/giây cho một request (có suy nghĩ); ~640 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; no thinking by default, thinking with "
             "`enable_thinking: true`; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (13.5 s). "
             "Base64 image: text and red rectangle right, but the blue circle was called green",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định không suy nghĩ, suy nghĩ khi gửi "
             "`enable_thinking: true`; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (13,5 giây). "
             "Ảnh base64: đọc đúng chữ và hình chữ nhật đỏ, nhưng nói hình tròn xanh dương là xanh lá"),
            ("Sampling parameters", "Tham số sinh",
             "a short factual question was right 9/20 times when the request sent no sampling parameters (vLLM "
             "defaults) and 20/20 with Qwen's recommended values: always send them",
             "một câu hỏi ngắn đúng 9/20 lần khi request không gửi tham số sinh (mặc định vLLM) và 20/20 khi gửi "
             "tham số Qwen khuyên: luôn gửi tham số"),
        ],
    },
    ("kaggle", "qwen3_5_9b"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel), --max-num-seqs 128",
             "0.33.0 / float16 / 2 (tensor parallel), --max-num-seqs 128"),
            ("Context (`auto`)", "Context (`auto`)", "109 824 tokens (reduced from 262 144 to fit)",
             "109 824 token (tự giảm từ 262 144 cho vừa)"),
            ("KV cache", "KV cache", "3.5 GiB (1.74 GiB per GPU) = 110 337 tokens: one full-length request at a time",
             "3,5 GiB (1,74 GiB mỗi GPU) = 110 337 token: một request dài tối đa mỗi lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)",
             "weights 8.83 GiB, other non-torch ~0.9 GiB, peak activations 1.63 GiB, CUDA graphs 0.28 + 0.14 GiB",
             "trọng số 8,83 GiB, non-torch khác ~0,9 GiB, activation đỉnh 1,63 GiB, CUDA graph 0,28 + 0,14 GiB"),
            ("Time", "Thời gian",
             "server start ~12 min (weights 76 s, torch.compile 32 s, ~7 min of unlogged warm-up, CUDA graphs 62 s)",
             "bật server ~12 phút (trọng số 76 giây, torch.compile 32 giây, ~7 phút khởi động không ghi log, "
             "CUDA graph 62 giây)"),
            ("Speed", "Tốc độ", "~17–19 tokens/second for one request; ~167 tokens/second total with 8 at once",
             "~17–19 token/giây cho một request; ~167 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling via `qwen3_coder`; "
             "JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; base64 image described correctly; "
             "`http(s)` image URL rejected; secret found in an 8 000-token prompt (10 s). One short answer was "
             "wrong (\"Toàn\" for the capital of Japan with `reasoning_effort=\"none\"`). That request sent no sampling "
             "parameters; on Qwen3.5-0.8B the same question was right 9/20 times without them and 20/20 with Qwen's "
             "recommended values, so the cause is vLLM's default sampling, not float16",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm qua `qwen3_coder`; "
             "JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; mô tả đúng ảnh base64; link ảnh "
             "`http(s)` bị từ chối; tìm đúng mã trong prompt 8 000 token (10 giây). Một câu trả lời ngắn bị sai "
             "(\"Toàn\" cho thủ đô Nhật Bản với `reasoning_effort=\"none\"`). Request đó không gửi tham số sinh; "
             "trên Qwen3.5-0.8B cùng câu hỏi đúng 9/20 lần khi không gửi và 20/20 khi gửi tham số Qwen khuyên, nên "
             "nguyên nhân là tham số sinh mặc định của vLLM, không phải float16"),
        ],
    },
    ("kaggle", "qwen3_8b"): {
        "summary": ("T4 x2, 2026-11-10", "T4 x2, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel)",
             "0.33.0 / float16 / 2 (tensor parallel)"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "9.2 GiB (4.58 GiB per GPU) = 66 704 tokens, 1.63 full-length requests at once",
             "9,2 GiB (4,58 GiB mỗi GPU) = 66 704 token, 1,63 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)", "weights 7.64 GiB, CUDA graphs 0.42 + 0.23 GiB",
             "trọng số 7,64 GiB, CUDA graph 0,42 + 0,23 GiB"),
            ("Time", "Thời gian", "loading weights 52 s, engine start 90 s (torch.compile 33 s, CUDA graphs 26 s)",
             "nạp trọng số 52 giây, khởi động engine 90 giây (torch.compile 33 giây, CUDA graph 26 giây)"),
            ("Speed", "Tốc độ", "one request ~8–26 tokens/second, uneven between runs; ~170 tokens/second total with 8 at once",
             "một request ~8–26 token/giây, không đều giữa các lần; ~170 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming. Thinking took 89 s for 1 172 tokens and an "
             "18 000-token prompt 76 s, close to the 100-second limit: use streaming",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream. Suy nghĩ 1 172 token mất 89 giây, prompt 18 000 token "
             "mất 76 giây, sát giới hạn 100 giây: nên dùng stream"),
        ],
    },
    ("kaggle", "qwen3_1_7b"): {
        "summary": ("T4 x2, 2026-11-10", "T4 x2, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel)",
             "0.33.0 / float16 / 2 (tensor parallel)"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "21.4 GiB (10.72 GiB per GPU) = 200 752 tokens, 4.90 full-length requests at once",
             "21,4 GiB (10,72 GiB mỗi GPU) = 200 752 token, 4,90 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)", "weights 1.61 GiB, peak activations 0.56 GiB, CUDA graphs 0.18 GiB",
             "trọng số 1,61 GiB, activation đỉnh 0,56 GiB, CUDA graph 0,18 GiB"),
            ("Time", "Thời gian", "engine start 76 s (torch.compile 27 s, CUDA graphs 19 s)",
             "khởi động engine 76 giây (torch.compile 27 giây, CUDA graph 19 giây)"),
            ("Speed", "Tốc độ", "~73 tokens/second for one request; ~506 tokens/second total with 8 at once",
             "~73 token/giây cho một request; ~506 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming. An 18 000-token prompt took 29.6 s, about half the "
             "time of a single T4",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 18 000 token mất 29,6 giây, khoảng nửa thời "
             "gian trên 1 × T4"),
        ],
    },
    ("colab", "qwen3_14b"): {
        "summary": ("A100 40 GB, 2026-11-10", "A100 40 GB, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07", "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / bfloat16 / 1", "0.33.0 / bfloat16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "38 304 tokens (reduced from 40 960 to fit)",
             "38 304 token (tự giảm từ 40 960 cho vừa)"),
            ("KV cache", "KV cache", "5.85 GiB = 38 320 tokens: one full-length request at a time, more if shorter",
             "5,85 GiB = 38 320 token: một request dài tối đa mỗi lúc, nhiều hơn nếu ngắn hơn"),
            ("GPU memory", "Bộ nhớ GPU", "weights 27.52 GiB, peak activations 2.0 GiB, CUDA graphs 0.28 GiB",
             "trọng số 27,52 GiB, activation đỉnh 2,0 GiB, CUDA graph 0,28 GiB"),
            ("Time", "Thời gian",
             "server start 3 min 47 s (weights 12 s, torch.compile 35 s, FlashInfer JIT ~1 min 50 s, CUDA graphs 16 s)",
             "bật server 3 phút 47 giây (trọng số 12 giây, torch.compile 35 giây, FlashInfer JIT ~1 phút 50 giây, "
             "CUDA graph 16 giây)"),
            ("Speed", "Tốc độ", "~44 tokens/second for one request; ~287 tokens/second total with 8 at once",
             "~44 token/giây cho một request; ~287 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming; a secret found in a 27 044-token prompt in 6 s. "
             "For the full 40 960 context and several long requests at once, use an 80 GB GPU",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream; tìm đúng mã trong prompt 27 044 token (6 giây). Muốn đủ "
             "context 40 960 và nhiều request dài cùng lúc, dùng GPU 80 GB"),
        ],
    },
    ("kaggle", "qwen3_0_6b"): {
        "summary": ("T4 x2, 2026-11-10", "T4 x2, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel)",
             "0.33.0 / float16 / 2 (tensor parallel)"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "23.5 GiB (11.75 GiB per GPU) = 220 080 tokens, 5.37 full-length requests at once",
             "23,5 GiB (11,75 GiB mỗi GPU) = 220 080 token, 5,37 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)", "weights 0.57 GiB, peak activations 0.57 GiB, CUDA graphs 0.18 GiB",
             "trọng số 0,57 GiB, activation đỉnh 0,57 GiB, CUDA graph 0,18 GiB"),
            ("Time", "Thời gian", "engine start 80 s (torch.compile 30 s, CUDA graphs 19 s)",
             "khởi động engine 80 giây (torch.compile 30 giây, CUDA graph 19 giây)"),
            ("Speed", "Tốc độ", "~104 tokens/second for one request; ~834 tokens/second total with 8 at once",
             "~104 token/giây cho một request; ~834 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming. An 18 000-token prompt took 28.5 s, half the time "
             "of a single T4",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 18 000 token mất 28,5 giây, bằng nửa thời gian "
             "trên 1 × T4"),
        ],
    },
    ("colab", "qwen3_0_6b"): {
        "summary": ("T4, 2026-11-10", "T4, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07", "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 1", "0.33.0 / float16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "11.14 GiB = 104 256 tokens, 2.55 full-length requests at once",
             "11,14 GiB = 104 256 token, 2,55 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU", "weights 1.12 GiB, peak activations 0.59 GiB, CUDA graphs 0.17 GiB",
             "trọng số 1,12 GiB, activation đỉnh 0,59 GiB, CUDA graph 0,17 GiB"),
            ("Time", "Thời gian", "server start 2 min 25 s (weights 11 s, torch.compile 28 s, CUDA graphs 18 s)",
             "bật server 2 phút 25 giây (trọng số 11 giây, torch.compile 28 giây, CUDA graph 18 giây)"),
            ("Speed", "Tốc độ", "~77 tokens/second for one request; ~570 tokens/second total with 8 at once",
             "~77 token/giây cho một request; ~570 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming. Long prompts are slow on T4 whatever the model "
             "size: ~17 000–18 000 tokens take ~50–55 s (also with Qwen3-1.7B), so prompts over ~30 000 tokens "
             "need streaming",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream. Prompt dài chậm trên T4 bất kể cỡ model: "
             "~17 000–18 000 token mất ~50–55 giây (Qwen3-1.7B cũng vậy), nên prompt trên ~30 000 token cần stream"),
        ],
    },
    ("kaggle", "qwen3_4b"): {
        "summary": ("T4 x2, 2026-11-10", "T4 x2, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel)",
             "0.33.0 / float16 / 2 (tensor parallel)"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "16.8 GiB (8.41 GiB per GPU) = 122 416 tokens, 2.99 full-length requests at once",
             "16,8 GiB (8,41 GiB mỗi GPU) = 122 416 token, 2,99 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)", "weights 3.82 GiB, peak activations 0.67 GiB, CUDA graphs 0.2 GiB",
             "trọng số 3,82 GiB, activation đỉnh 0,67 GiB, CUDA graph 0,2 GiB"),
            ("Time", "Thời gian", "engine start 85 s (weights 10 s, torch.compile 32 s, CUDA graphs 21 s)",
             "khởi động engine 85 giây (trọng số 10 giây, torch.compile 32 giây, CUDA graph 21 giây)"),
            ("Speed", "Tốc độ", "~42 tokens/second for one request; ~290 tokens/second total with 8 at once "
             "(about 1.7× a single Colab T4)",
             "~42 token/giây cho một request; ~290 token/giây tổng khi 8 request cùng lúc "
             "(khoảng 1,7 lần Colab 1 × T4)"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming. A 27 000-token prompt got error 524 without "
             "streaming and was answered correctly with streaming (first token after 15 s): stream long prompts",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 27 000 token bị lỗi 524 khi không stream, "
             "trả lời đúng khi stream (token đầu sau 15 giây): prompt dài nên dùng stream"),
        ],
    },
    ("colab", "qwen3_5_9b"): {
        "summary": ("A100 40 GB 2026-11-10, A100 80 GB 2026-11-11", "A100 40 GB 10/11/2026, A100 80 GB 11/11/2026"),
        "rows": [
            ("A100 80 GB run", "Lần chạy A100 80 GB",
             "1 × NVIDIA A100-SXM4-80GB, `--max-num-seqs 128`: context 262 144, KV cache 49.25 GiB = 1 592 668 tokens "
             "(6.08 full-length requests at once), server start ~5 min 50 s, ~88 tokens/second for one request, "
             "~505 total with 8 at once; 13/13 checks passed, a 24 000-token prompt in 3.2 s. "
             "The rows below are from the A100 40 GB run",
             "1 × NVIDIA A100-SXM4-80GB, `--max-num-seqs 128`: context 262 144, KV cache 49,25 GiB = 1 592 668 token "
             "(6,08 request dài tối đa cùng lúc), bật server ~5 phút 50 giây, ~88 token/giây cho một request, "
             "~505 tổng khi 8 request; qua 13/13 phép thử, prompt 24 000 token trong 3,2 giây. "
             "Các dòng dưới là lần chạy A100 40 GB"),
            ("GPU", "GPU", "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07",
             "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / bfloat16 / 1", "0.33.0 / bfloat16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "13.17 GiB = 425 788 tokens, 1.62 full-length requests at once",
             "13,17 GiB = 425 788 token, 1,62 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU",
             "weights 17.57 GiB, other non-torch ~2.5 GiB (image encoder, linear-attention state), peak activations "
             "2.27 GiB, CUDA graphs 0.27 GiB",
             "trọng số 17,57 GiB, non-torch khác ~2,5 GiB (bộ mã hoá ảnh, trạng thái linear attention), activation "
             "đỉnh 2,27 GiB, CUDA graph 0,27 GiB"),
            ("Time", "Thời gian",
             "server start 6 min (weights 10 s, torch.compile 28 s, FlashInfer JIT ~1 min 50 s, ~1 min 40 s of "
             "unlogged warm-up)",
             "bật server 6 phút (trọng số 10 giây, torch.compile 28 giây, FlashInfer JIT ~1 phút 50 giây, ~1 phút "
             "40 giây khởi động không ghi log)"),
            ("Speed", "Tốc độ", "~73 tokens/second for one request; ~500 tokens/second total with 8 at once",
             "~73 token/giây cho một request; ~500 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "tool calling via `qwen3_coder`; JSON schema; base64 image correctly described; `http(s)` image URL "
             "rejected (HTTP 400); Anthropic `/v1/messages`; streaming; a secret found in the middle of a "
             "57 242-token prompt (6 s)",
             "không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "gọi hàm qua `qwen3_coder`; JSON schema; mô tả đúng ảnh base64; link ảnh `http(s)` bị từ chối "
             "(HTTP 400); Anthropic `/v1/messages`; stream; tìm đúng mã giấu giữa prompt 57 242 token (6 giây)"),
            ("11 Wikimedia images", "11 ảnh Wikimedia",
             "all served (10 in parallel in 8.7 s; 3 images in one request, 2 125 tokens, 3.9 s). Scenes, food, "
             "landmarks and a handwritten blackboard formula read correctly; on a folded receipt the amounts were "
             "right but the shop name and date were not; a Vietnamese no-overtaking sign was misread",
             "đều chạy (10 ảnh song song 8,7 giây; 3 ảnh trong một request, 2 125 token, 3,9 giây). Nhận đúng cảnh "
             "vật, món ăn, địa danh và công thức viết tay trên bảng đen; hoá đơn bị gấp: số tiền đúng nhưng tên "
             "cửa hàng và ngày sai; đọc sai biển cấm vượt của Việt Nam"),
            ("13 text tasks (auto-graded)", "13 tác vụ văn bản (chấm tự động)",
             "correct: multi-step math and a logic puzzle (with thinking), summary, spelling fixes, JSON extraction, "
             "multi-turn memory, two-step tool calling. Wrong: an idiom translated literally with a typo, \"longest "
             "river in Vietnam\" answered as the Red River, a lục bát poem with the wrong 6-8 pattern. Code without "
             "thinking: 11/14 tests in 8 s; with thinking the model looped for 30 000 tokens (7 min) and never wrote "
             "the code, so disable thinking for coding",
             "đúng: toán nhiều bước và câu đố logic (có suy nghĩ), tóm tắt, sửa chính tả, trích JSON, nhớ hội thoại "
             "nhiều lượt, gọi hàm 2 bước. Sai: dịch thành ngữ theo nghĩa đen kèm lỗi chính tả, \"sông dài nhất chảy "
             "qua Việt Nam\" trả lời sông Hồng, thơ lục bát sai luật 6-8. Viết code khi tắt suy nghĩ: qua 11/14 test "
             "trong 8 giây; khi bật suy nghĩ model lặp 30 000 token (7 phút) mà không ra code, nên tắt suy nghĩ khi "
             "viết code"),
        ],
    },
    ("colab", "qwen3_1_7b"): {
        "summary": ("T4, 2026-11-10", "T4, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07", "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 1", "0.33.0 / float16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "8.72 GiB = 81 632 tokens, 1.99 full-length requests at once",
             "8,72 GiB = 81 632 token, 1,99 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU", "weights 3.22 GiB, peak activations 0.92 GiB, CUDA graphs 0.17 GiB",
             "trọng số 3,22 GiB, activation đỉnh 0,92 GiB, CUDA graph 0,17 GiB"),
            ("Time", "Thời gian", "server start 2 min 52 s (weights 26 s, torch.compile 29 s, CUDA graphs 20 s)",
             "bật server 2 phút 52 giây (trọng số 26 giây, torch.compile 29 giây, CUDA graph 20 giây)"),
            ("Speed", "Tốc độ", "~50 tokens/second for one request; ~330 tokens/second total with 8 at once",
             "~50 token/giây cho một request; ~330 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "Vietnamese text in float16 reads normally; tool calling; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; streaming. Thinking on a simple question: 1 231 tokens in 22 s. Quality is limited: one "
             "answer mixed in a few Thai characters",
             "không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "văn bản tiếng Việt ở float16 bình thường; gọi hàm; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; stream. Suy nghĩ với câu đơn giản: 1 231 token trong 22 giây. Chất lượng có giới hạn: "
             "một câu trả lời lẫn vài chữ tiếng Thái"),
        ],
    },
    ("colab", "qwen3_4b"): {
        "summary": ("T4, 2026-11-10", "T4, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07", "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 1", "0.33.0 / float16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "30 592 tokens (reduced from 40 960 to fit)",
             "30 592 token (tự giảm từ 40 960 cho vừa)"),
            ("KV cache", "KV cache", "4.2 GiB = 30 608 tokens: one full-length request, or many short ones",
             "4,2 GiB = 30 608 token: một request dài tối đa, hoặc nhiều request ngắn"),
            ("GPU memory", "Bộ nhớ GPU", "weights 7.56 GiB, peak activations 1.15 GiB, CUDA graphs 0.22 GiB",
             "trọng số 7,56 GiB, activation đỉnh 1,15 GiB, CUDA graph 0,22 GiB"),
            ("Speed", "Tốc độ", "~25 tokens/second for one request; ~170 tokens/second total with 8 at once",
             "~25 token/giây cho một request; ~170 token/giây tổng khi 8 request cùng lúc"),
            ("Time", "Thời gian", "server start 3 min 30 s (weights 42 s, torch.compile 37 s, CUDA graphs 30 s)",
             "bật server 3 phút 30 giây (trọng số 42 giây, torch.compile 37 giây, CUDA graph 30 giây)"),
            ("Expected log noise", "Log trông như lỗi nhưng bình thường",
             "`ERROR ... Cannot use FA version 2` (T4 falls back to Triton attention), FlashInfer sampler "
             "unavailable, and many `bfloat16 into float16` weight warnings",
             "`ERROR ... Cannot use FA version 2` (T4 chuyển sang Triton attention), FlashInfer sampler "
             "không dùng được, và nhiều cảnh báo nạp trọng số `bfloat16 into float16`"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "Vietnamese text in float16 reads normally; tool calling; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; streaming. Thinking is slow on T4 (a simple question used 2 000 tokens, 81 s): disable "
             "it or stream",
             "không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort=\"none\"`); "
             "văn bản tiếng Việt ở float16 bình thường; gọi hàm; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; stream. Suy nghĩ chậm trên T4 (câu đơn giản tốn 2 000 token, 81 giây): nên tắt "
             "hoặc dùng stream"),
        ],
    },
    ("colab", "qwen3_8b"): {
        "summary": ("A100 40 GB, 2026-11-10", "A100 40 GB, 10/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07",
             "1 × NVIDIA A100-SXM4-40GB, driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / bfloat16 / 1", "0.33.0 / bfloat16 / 1"),
            ("Context (`auto`)", "Context (`auto`)", "40 960 tokens (the model maximum)", "40 960 token (tối đa của model)"),
            ("KV cache", "KV cache", "18.47 GiB = 134 448 tokens, 3.28 full-length requests at once",
             "18,47 GiB = 134 448 token, 3,28 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU", "weights 15.27 GiB, peak activations 1.64 GiB, CUDA graphs 0.23 GiB",
             "trọng số 15,27 GiB, activation đỉnh 1,64 GiB, CUDA graph 0,23 GiB"),
            ("Speed", "Tốc độ", "~75 tokens/second for one request", "~75 token/giây cho một request"),
            ("Time", "Thời gian",
             "install 2 min, download 43 s, server start 3 min 35 s (FlashInfer JIT ~1 min 45 s, "
             "torch.compile 31 s); ~7 min from Run all to a public URL",
             "cài 2 phút, tải 43 giây, bật server 3 phút 35 giây (FlashInfer JIT ~1 phút 45 giây, "
             "torch.compile 31 giây); ~7 phút từ Run all tới khi có URL công khai"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off (`enable_thinking`, "
             "`reasoning_effort=\"none\"`, `/no_think`); tool calling; JSON schema; completions; `/tokenize`; "
             "Anthropic `/v1/messages`; `cache_salt`; streaming for 124 s / 9 000 tokens without being cut",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ (`enable_thinking`, "
             "`reasoning_effort=\"none\"`, `/no_think`); gọi hàm; JSON schema; completions; `/tokenize`; "
             "Anthropic `/v1/messages`; `cache_salt`; stream 124 giây / 9 000 token không bị cắt"),
        ],
    },
}

# gpus: (name, GPU count, MiB per GPU, compute capability) commonly offered on the platform.
PLATFORMS = {
    "colab": {
        "name": "Google Colab",
        "badge": "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 "(https://colab.research.google.com/github/{{REPO}}/blob/main/notebooks/colab/colab_{{SLUG}}.ipynb)",
        "gpus": [("T4", 1, 15360, 7.5), ("L4", 1, 23034, 8.9), ("A100 40 GB", 1, 40960, 8.0),
                 ("A100 80 GB", 1, 81920, 8.0), ("H100 80 GB", 1, 81559, 9.0)],
        "setup": """\
1. **Runtime → Change runtime type:** the notebook opens with {{DEFAULT_GPU}}; pick a stronger GPU if you have one.
2. (Recommended) 🔑 **Secrets** in the left sidebar: add `LLM_API_KEY`, enable Notebook access.
3. **Runtime → Run all.** Keep the tab open while in use.""",
        "gpu_fail": "Runtime → Change runtime type → pick a GPU with enough memory (table at the top)",
        "no_internet": "check the runtime's network, then Runtime → Disconnect and delete runtime and run again",
        "base": "/content",
        "secret_code": """\
try:
    from google.colab import userdata
    API_KEY = userdata.get("LLM_API_KEY")
except Exception:
    API_KEY = None""",
        "share_warning": "Clear outputs (Edit → Clear all outputs) before sharing the notebook: the key is in them.",
        "shutdown": "Then Runtime → Disconnect and delete runtime to release the GPU.",
        "metadata": lambda gpu: {"accelerator": "GPU", "colab": {"gpuType": gpu.split()[0]}},
    },
    "kaggle": {
        "name": "Kaggle",
        "badge": "[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
                 "(https://kaggle.com/kernels/welcome?src=https://github.com/{{REPO}}/blob/main/notebooks/kaggle/kaggle_{{SLUG}}.ipynb)",
        "gpus": [("T4 x2", 2, 15360, 7.5), ("P100", 1, 16384, 6.0)],
        "setup": """\
1. **Settings → Accelerator:** a GPU with enough memory per the table above (not P100).
2. **Settings → Internet → On** (requires a phone-verified account).
3. (Recommended) **Add-ons → Secrets:** add `LLM_API_KEY` and attach it to the notebook.
4. **Run All.** Keep the tab open while in use.""",
        "gpu_fail": "Settings → Accelerator → pick a GPU with enough memory (table at the top)",
        "no_internet": "Settings → Internet → On (requires a phone-verified account), then run all cells again",
        "base": "/tmp",
        "secret_code": """\
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("LLM_API_KEY")
except Exception:
    API_KEY = None""",
        "share_warning": "Keep the notebook Private and clear outputs before Save Version: the key is in them.",
        "shutdown": "Then click Stop session to stop using GPU quota.",
        "metadata": lambda gpu: {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                            "isInternetEnabled": True}},
    },
}


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


# Notebook template: ("md" | "code", text). {{NAME}} is substituted at build time.
CELLS = [
    ("md", """\
# Host {{SERVED_NAME}} on {{PLATFORM_NAME}}

{{BADGE}}

Runs [{{MODEL_ID}}](https://huggingface.co/{{MODEL_ID}}) with vLLM and exposes it to the Internet
as a key-protected, OpenAI-compatible API. The notebook only hosts the model: prompts, parameters
and answers stay on the caller's side, and the server records no content.

```
caller ──HTTPS──▶ Cloudflare Tunnel ──▶ proxy (checks key) ──▶ vLLM ──▶ GPU
```

{{MODEL_NOTES}}**GPU** (weights {{WEIGHTS_GIB}} GiB; other GPUs are checked in step 1):

{{GPU_TABLE}}

**How to run:**

{{SETUP}}

Step 8 prints the URL and API key. How to call the API:
[docs/api.md](https://github.com/{{REPO}}/blob/main/docs/api.md)."""),

    ("md", """\
## 1. Check the GPU and Internet

The notebook downloads vLLM, the model and cloudflared, so it needs Internet. vLLM {{VLLM_VERSION}} needs NVIDIA driver ≥ 580 (CUDA 13) and GPU compute capability ≥ 7.0. The
notebook picks `bfloat16` (Ampere and newer) or `float16` (T4), and splits the model across all GPUs
when there are several."""),

    ("code", """\
import socket, subprocess

for host in ("pypi.org", "huggingface.co", "github.com"):
    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        raise RuntimeError(f"No Internet (cannot resolve {host}): {{NO_INTERNET}}") from None

out = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=True,
).stdout.strip()
rows = [[s.strip() for s in line.split(",")] for line in out.splitlines()]
GPU_NAME, DRIVER = rows[0][0], rows[0][2]
GPU_MIB = min(int(r[1]) for r in rows)
CAP = min(float(r[3]) for r in rows)
TP = len(rows) if len(rows) in (1, 2, 4, 8) else 1
DTYPE = "bfloat16" if CAP >= 8.0 else "float16"
assert int(DRIVER.split(".")[0]) >= 580, f"Driver {DRIVER} is older than 580: try another GPU type"
assert CAP >= 7.0, f"vLLM does not support {GPU_NAME}: {{GPU_FAIL}}"
assert GPU_MIB * TP >= {{MIN_GPU_MIB}}, f"{GPU_MIB * TP} MiB is not enough: {{GPU_FAIL}}"
print(f"{len(rows)} × {GPU_NAME}, {GPU_MIB} MiB, driver {DRIVER}, compute {CAP} → {DTYPE}, {TP} GPU")"""),

    ("md", """\
## 2. Settings

- `MAX_MODEL_LEN`: max tokens per request. `auto` = the longest that fits on the GPU.
- `GPU_MEMORY`: fraction of GPU memory for vLLM; lower to 0.85 on out-of-memory errors.
- `ENFORCE_EAGER`: disables CUDA graphs; loads faster with less memory but generates slower.

The API key comes from the `LLM_API_KEY` secret; without it a random key is generated on every run.
{{SHARE_WARNING}}"""),

    ("code", """\
import os, secrets

MODEL_ID = "{{MODEL_ID}}"
SERVED_NAME = "{{SERVED_NAME}}"
MODEL_DIR = "{{BASE}}/models/{{SERVED_NAME}}"
WORK = "{{BASE}}/llm_host"
PORT, PROXY_PORT = 8000, 8001

MAX_MODEL_LEN = "auto"  #@param {type:"string"}
GPU_MEMORY = {{GPU_MEMORY}}  #@param {type:"number"}
ENFORCE_EAGER = False  #@param {type:"boolean"}

{{SECRET_CODE}}
print("API key:", "from the LLM_API_KEY secret" if API_KEY else "randomly generated")
API_KEY = API_KEY or secrets.token_urlsafe(32)
os.makedirs(WORK, exist_ok=True)"""),

    ("md", """\
## 3. Install vLLM

Takes a few minutes. pip then prints red `ERROR: pip's dependency resolver...` lines about
preinstalled packages (google-colab, gradio...): this is expected and harmless, since vLLM runs in its
own process. The install succeeded if the last line shows the vLLM version."""),

    ("code", """\
!pip install -q "vllm=={{VLLM_VERSION}}"
!vllm --version"""),

    ("md", """\
## 4. Download the model (~{{DOWNLOAD_GB}} GB)"""),

    ("code", """\
import shutil
from huggingface_hub import snapshot_download

os.makedirs(MODEL_DIR, exist_ok=True)
free_gb = shutil.disk_usage(MODEL_DIR).free / 1e9
assert free_gb > {{DOWNLOAD_GB}} * 1.1, f"Only {free_gb:.0f} GB of disk left, need ~{{DOWNLOAD_GB}} GB"
snapshot_download(MODEL_ID, local_dir=MODEL_DIR)"""),

    ("md", """\
## 5. Start the server

The server adds no system prompt, sets no sampling parameters ({{SAMPLING}}), and logs no prompts
or answers. The flags only enable features that requests can use:

{{FLAG_NOTES}}

The server only listens on `127.0.0.1`. The first start takes a few minutes."""),

    ("code", """\
import json, time, urllib.request

LOG = f"{WORK}/server.log"
cmd = [
    "vllm", "serve", MODEL_DIR, "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1", "--port", str(PORT), "--api-key", API_KEY,
    "--dtype", DTYPE, "--tensor-parallel-size", str(TP),
    "--max-model-len", str(MAX_MODEL_LEN), "--gpu-memory-utilization", str(GPU_MEMORY),
    "--enable-prefix-caching", "--disable-uvicorn-access-log",
    {{SERVE_FLAGS}},
]
if ENFORCE_EAGER:
    cmd.append("--enforce-eager")
server = subprocess.Popen(cmd, stdout=open(LOG, "ab"), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    with urllib.request.urlopen(urllib.request.Request(base + path, data, headers), timeout=300) as r:
        return json.loads(r.read())


end = time.time() + 30 * 60
while time.time() < end and server.poll() is None and not healthy():
    time.sleep(5)
if not healthy():
    log = open(LOG, errors="replace").read()
    print(log[-3000:])
    errors = [l for l in log.splitlines() if "Error" in l and "Traceback" not in l]
    if errors:  # the root cause is often far above the last lines
        print("\\nError lines in the log:\\n" + "\\n".join(errors[-15:]))
    raise RuntimeError("The server did not start; see the log above")

LOCAL = f"http://127.0.0.1:{PORT}"
CONTEXT = call(LOCAL, "/v1/models")["data"][0]["max_model_len"]
print(f"Server ready, max context {CONTEXT} tokens")"""),

    ("md", """\
## 6. Measure speed

One request with no content generates 256 tokens (`ignore_eos`) to measure speed. Nothing is printed
from the answer."""),

    ("code", """\
started = time.time()
reply = call(LOCAL, "/v1/chat/completions", {
    "model": SERVED_NAME, "messages": [{"role": "user", "content": "."}],
    "max_tokens": 256, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False},
})
TOKENS_PER_SECOND = reply["usage"]["completion_tokens"] / (time.time() - started)
print(f"~{TOKENS_PER_SECOND:.0f} tokens/second")"""),

    ("md", """\
## 7. Key-checking proxy

vLLM's `--api-key` does not protect every path (`/invocations`, `/sleep`... stay open). The proxy
is the only way in: it checks the key on every path and only lets the model APIs through. Source and
tests: [proxy/proxy.py](https://github.com/{{REPO}}/blob/main/proxy/proxy.py)."""),

    ("code", """\
import sys, urllib.error

PROXY_FILE = f"{WORK}/proxy.py"
with open(PROXY_FILE, "w") as f:
    f.write(r'''{{PROXY_SRC}}''')
proxy = subprocess.Popen([sys.executable, PROXY_FILE, str(PROXY_PORT), str(PORT), API_KEY],
                         stdout=open(f"{WORK}/proxy.log", "ab"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
PROXY = f"http://127.0.0.1:{PROXY_PORT}"
for _ in range(30):
    try:
        call(PROXY, "/v1/models")
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError(f"The proxy did not start; see {WORK}/proxy.log")

for path, expected in [("/v1/chat/completions", 401), ("/invocations", 404)]:  # POST without key
    try:
        urllib.request.urlopen(urllib.request.Request(PROXY + path, b"{}"))
        raise AssertionError(f"{path} is not blocked")
    except urllib.error.HTTPError as error:
        assert error.code == expected, (path, error.code)
print("Proxy running: no key → 401, admin APIs → 404")"""),

    ("md", """\
## 8. Expose to the Internet

Cloudflare quick tunnel: no account needed, the URL changes on every run. Requests without
streaming are cut after 100 seconds (error 524); with `stream: true` the answer arrives as it is
generated and is not cut (tested: 124 seconds, 9000 tokens)."""),

    ("code", """\
import re

CLOUDFLARED = f"{WORK}/cloudflared"
subprocess.run(["wget", "-q", "-O", CLOUDFLARED, "https://github.com/cloudflare/cloudflared/releases/"
                "download/{{CLOUDFLARED_VERSION}}/cloudflared-linux-amd64"], check=True)
os.chmod(CLOUDFLARED, 0o755)
TUNNEL_LOG = f"{WORK}/tunnel.log"
tunnel = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", PROXY],
                          stdout=open(TUNNEL_LOG, "ab"), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)
PUBLIC = None
for _ in range(60):
    found = re.findall(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open(TUNNEL_LOG).read())
    if found:
        PUBLIC = found[-1]
        break
    time.sleep(2)
assert PUBLIC, f"Could not get the tunnel URL; see {TUNNEL_LOG}"
for _ in range(30):  # DNS for a new URL takes a few seconds
    try:
        call(PUBLIC, "/v1/models")
        break
    except OSError:
        time.sleep(3)
else:
    raise RuntimeError(f"Could not reach the server through the tunnel; see {TUNNEL_LOG}")

print(f'''
Base URL : {PUBLIC}/v1
API key  : {API_KEY}
Model    : {SERVED_NAME}
Context  : {CONTEXT} tokens
Speed    : ~{TOKENS_PER_SECOND:.0f} tokens/second; without streaming keep max_tokens below ~{int(TOKENS_PER_SECOND * 90)} (100-second limit)

curl {PUBLIC}/v1/chat/completions \\\\
  -H "Authorization: Bearer {API_KEY}" -H "Content-Type: application/json" \\\\
  -d '{{"model": "{SERVED_NAME}", "messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 512}}'
''')"""),

    ("md", """\
## 9. Monitor

Prints status and throughput stats every minute. Stop this cell (■) before running Shut down: the
notebook runs one cell at a time. Stopping it does not stop the server."""),

    ("code", """\
PARTS = {"vLLM": server, "proxy": proxy, "tunnel": tunnel}
try:
    while True:
        dead = [name for name, process in PARTS.items() if process.poll() is not None]
        stats = [l for l in open(LOG, errors="replace").read().splitlines() if "throughput" in l]
        state = f"ERROR: {', '.join(dead)} stopped" if dead else ("OK" if healthy() else "vLLM /health not OK")
        print(time.strftime("%H:%M:%S"), state, "|", stats[-1][max(stats[-1].find("Avg prompt"), 0):] if stats else "")
        if dead:
            break
        time.sleep(60)
except KeyboardInterrupt:
    print("Monitor stopped; the server is still running")"""),

    ("md", """\
## 10. Shut down

Stop the Monitor cell first, then run the cell below to stop the tunnel, proxy and vLLM. {{SHUTDOWN}}"""),

    ("code", """\
import signal

for process in (globals().get(n) for n in ("tunnel", "proxy", "server")):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
print("Tunnel, proxy and vLLM stopped")"""),
]


def render(text: str, values: dict) -> str:
    for key, value in values.items():  # in order: earlier values may contain later {{...}}
        text = text.replace("{{" + key + "}}", str(value))
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    assert not left, f"not substituted: {left}"
    return text


def build(platform: dict, model: dict, proxy_src: str) -> dict:
    default_gpu = next(gpu[0] for gpu in platform["gpus"] if runnable(model, gpu))
    values = {
        "BADGE": platform["badge"], "SETUP": platform["setup"], "GPU_TABLE": gpu_table(model, platform),
        "SECRET_CODE": platform["secret_code"], "SHARE_WARNING": platform["share_warning"],
        "SHUTDOWN": platform["shutdown"], "GPU_FAIL": platform["gpu_fail"], "NO_INTERNET": platform["no_internet"],
        "PLATFORM_NAME": platform["name"], "BASE": platform["base"], "DEFAULT_GPU": default_gpu,
        "REPO": REPO, "VLLM_VERSION": VLLM_VERSION, "CLOUDFLARED_VERSION": CLOUDFLARED_VERSION,
        "SLUG": model["slug"], "MODEL_ID": model["model_id"], "SERVED_NAME": model["served_name"],
        "WEIGHTS_GIB": f"{model['weights_gib']:.1f}", "FLAG_NOTES": model["flag_notes"],
        "SAMPLING": model["sampling"], "SERVE_FLAGS": ", ".join(json.dumps(f) for f in model["flags"]),
        "MODEL_NOTES": model["notes"] + "\n\n" if model["notes"] else "",
        "DOWNLOAD_GB": model["download_gb"], "MIN_GPU_MIB": min_gpu_mib(model), "GPU_MEMORY": GPU_MEMORY,
        "PROXY_SRC": proxy_src,  # last: nothing inside the proxy source gets substituted
    }
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        source = render(text, values).split("\n")
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i:02d}", "metadata": {},
                "source": [line + "\n" for line in source[:-1]] + source[-1:]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    metadata = platform["metadata"](default_gpu)
    metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def notebook_name(key: str, model: dict) -> str:
    return f"{key}_{model['slug']}.ipynb"


def available(model: dict, key: str) -> bool:
    """Whether the model gets a notebook on this platform (default: every platform)."""
    return key in model.get("platforms", PLATFORMS)


def test_status(key: str, model: dict, vi: bool) -> str:
    run = TESTED.get((key, model["slug"]))
    return f"✓ {run['summary'][1 if vi else 0]}" if run else ("Chưa thử" if vi else "Not tested yet")


def platform_readme(key: str, platform: dict, vi: bool) -> str:
    """notebooks/<platform>/README(.vi).md: notebook specs, estimated GPU table, test results."""
    t = (lambda en, vi_text: vi_text) if vi else (lambda en, vi_text: en)
    num = (lambda x: f"{x:.1f}".replace(".", ",")) if vi else (lambda x: f"{x:.1f}")
    badge = render(platform["badge"], {"REPO": REPO, "SLUG": "SLUG"})
    other = "[English](README.md) · **Tiếng Việt**" if vi else "**English** · [Tiếng Việt](README.vi.md)"
    main_readme = "../../README.vi.md" if vi else "../../README.md"
    models = [m for m in MODELS if available(m, key)]
    lines = [
        t(f"# Host open models on {platform['name']}: ready-made notebooks",
          f"# Host model mở trên {platform['name']}: notebook dựng sẵn"), "", other, "",
        t(f"Back to the [main README]({main_readme}).", f"Quay lại [README chính]({main_readme})."), "",
        "## Notebook", "",
        t("| Notebook | Model | Weights | Download | Min. total GPU memory | Test status |",
          "| Notebook | Model | Trọng số | Dung lượng tải | Tổng bộ nhớ GPU tối thiểu | Chạy thử |"),
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        name = notebook_name(key, model)
        status = test_status(key, model, vi)
        lines.append(f"| [{name}]({name}) {badge.replace('SLUG', model['slug'])} | `{model['model_id']}` | "
                     f"{num(model['weights_gib'])} GiB | ~{model['download_gb']} GB | "
                     f"~{math.ceil(min_gpu_mib(model) / 1024)} GB | {status} |")
    for model in models:
        note = model["notes_vi"] if vi else model["notes"]
        if note:
            lines += ["", f"**{model['served_name']}:** {note}"]
    for model in MODELS:
        if not available(model, key):
            reason = model["unavailable"][key][1 if vi else 0]
            lines += ["", t(f"**{model['served_name']}:** no {platform['name']} notebook: {reason}.",
                            f"**{model['served_name']}:** không có notebook cho {platform['name']}: {reason}.")]
    lines += [
        "", t("## Shared settings", "## Thông số chung"), "",
        t(f"vLLM {VLLM_VERSION}, cloudflared {CLOUDFLARED_VERSION}. `--max-model-len auto`, "
          f"`--gpu-memory-utilization {GPU_MEMORY}`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); "
          "the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. "
          "The server sets no sampling parameters and logs no prompts or answers.",
          f"vLLM {VLLM_VERSION}, cloudflared {CLOUDFLARED_VERSION}. `--max-model-len auto`, "
          f"`--gpu-memory-utilization {GPU_MEMORY}`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); "
          "model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. "
          "Server không đặt tham số sinh, không ghi prompt hay câu trả lời."),
        "", t("## Estimated memory left for context", "## Bộ nhớ ước tính còn cho context"), "",
        "| GPU | " + " | ".join(m["served_name"] for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for gpu in platform["gpus"]:
        cells = []
        for model in models:
            if gpu[3] < 7.0:
                cells.append(t("✗ not supported by vLLM", "✗ vLLM không hỗ trợ"))
            elif runnable(model, gpu):
                cells.append(f"~{kv_gib(model, gpu[1], gpu[2]):.0f} GiB")
            else:
                cells.append(t("✗ not enough memory", "✗ thiếu bộ nhớ"))
        lines.append(f"| {gpu[0]} | " + " | ".join(cells) + " |")
    lines += ["", t("## Test results", "## Kết quả chạy thử")]
    for model in models:
        name = notebook_name(key, model)
        run = TESTED.get((key, model["slug"]))
        lines += ["", f"### {name}", ""]
        if not run:
            lines.append(t("Not tested yet.", "Chưa thử."))
            continue
        lines += [t("| Item | Result |", "| Mục | Kết quả |"), "|---|---|"]
        lines += [f"| {r[1] if vi else r[0]} | {r[3] if vi else r[2]} |" for r in run["rows"]]
    return "\n".join(lines) + "\n"


def overview_readme(vi: bool) -> str:
    """notebooks/README(.vi).md: every model × platform, with test status."""
    t = (lambda en, vi_text: vi_text) if vi else (lambda en, vi_text: en)
    num = (lambda x: f"{x:.1f}".replace(".", ",")) if vi else (lambda x: f"{x:.1f}")
    sfx = ".vi.md" if vi else ".md"
    other = "[English](README.md) · **Tiếng Việt**" if vi else "**English** · [Tiếng Việt](README.vi.md)"
    platform_links = ", ".join(f"[{p['name']}]({k}/README{sfx})" for k, p in PLATFORMS.items())
    lines = [
        t("# Notebooks", "# Notebook"), "", other, "",
        t(f"Back to the [main README](../README{sfx}).", f"Quay lại [README chính](../README{sfx})."), "",
        t("Each notebook hosts one model on one platform. Pick a model, then a platform where your account has a "
          f"GPU with enough memory. GPU tables and test results per platform: {platform_links}. "
          "Choose the GPU before Run all; it needs at least the model's \"Min. total GPU memory\".",
          "Mỗi notebook host một model trên một nền tảng. Chọn model, rồi chọn nền tảng mà tài khoản của bạn có GPU "
          f"đủ bộ nhớ. Bảng GPU và kết quả chạy thử theo nền tảng: {platform_links}. "
          "Chọn GPU trước khi Run all; GPU cần ít nhất \"Tổng bộ nhớ GPU tối thiểu\" của model."), "",
        t("| Model | Input | Weights | Context | Min. total GPU memory | ", "| Model | Đầu vào | Trọng số | Context | "
          "Tổng bộ nhớ GPU tối thiểu | ") + " | ".join(p["name"] for p in PLATFORMS.values()) + " |",
        "|---|---|---|---|---|" + "---|" * len(PLATFORMS),
    ]
    for model in MODELS:
        cells = []
        for key in PLATFORMS:
            if available(model, key):
                name = notebook_name(key, model)
                cells.append(f"[{name}]({key}/{name})<br>{test_status(key, model, vi)}")
            else:
                cells.append(f"— ({model['unavailable'][key][1 if vi else 0]})")
        context = f"{model['context']:,}".replace(",", " ")
        lines.append(f"| `{model['model_id']}` | {model['input'][1 if vi else 0]} | {num(model['weights_gib'])} GiB | "
                     f"{context} | ~{math.ceil(min_gpu_mib(model) / 1024)} GB | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    proxy_src = (ROOT / "proxy" / "proxy.py").read_text()
    assert "'''" not in proxy_src, "proxy.py must not contain ''' (it is embedded in r'''...''')"
    for key, platform in PLATFORMS.items():
        out = ROOT / "notebooks" / key
        out.mkdir(parents=True, exist_ok=True)
        for model in (m for m in MODELS if available(m, key)):
            path = out / notebook_name(key, model)
            path.write_text(json.dumps(build(platform, model, proxy_src), ensure_ascii=False, indent=1) + "\n")
            print("wrote", path.relative_to(ROOT))
        for vi in (False, True):
            path = out / ("README.vi.md" if vi else "README.md")
            path.write_text(platform_readme(key, platform, vi))
            print("wrote", path.relative_to(ROOT))
    for vi in (False, True):
        path = ROOT / "notebooks" / ("README.vi.md" if vi else "README.md")
        path.write_text(overview_readme(vi))
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

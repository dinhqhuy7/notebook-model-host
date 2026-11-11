"""Text models: model families, models and recorded test runs.

Add a model: a line in MODELS (its "kind" must be "text"). Record a test run: an entry in TESTED, then
`python tools/build_notebooks.py --freeze <notebook>`.
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

# Real test runs of this type, shown in notebooks/<kind>/<slug>/README*.md. Key: (platform, model slug).
# Rows: (English label, Vietnamese label, English value, Vietnamese value).
TESTED = {
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

"""Vision-language models: model families, models and recorded test runs.

Add a model: a line in MODELS (its "kind" must be "vision-language"). Record a test run: an entry in TESTED, then
`python tools/build_notebooks.py --freeze <notebook>`.
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

# Real test runs of this type, shown in notebooks/<kind>/<slug>/README*.md. Key: (platform, model slug).
# Rows: (English label, Vietnamese label, English value, Vietnamese value).
TESTED = {
    ("colab", "qwen3_vl_2b_instruct"): {
        "summary": ("T4 & A100 40 GB, 2026-11-11", "T4 & A100 40 GB, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB) and 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07",
             "1 × Tesla T4 (15 GB) và 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 (T4), bfloat16 (A100) / 1",
             "0.33.0 / float16 (T4), bfloat16 (A100) / 1"),
            ("Context (`auto`)", "Context (`auto`)", "T4: 61 808 tokens; A100: 262 144 tokens (model maximum)",
             "T4: 61 808 token; A100: 262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "T4: 6.6 GiB = 61 823 tokens, 1.00×; A100: 29.03 GiB = 271 808 tokens, 1.04×",
             "T4: 6,6 GiB = 61 823 token, 1,00×; A100: 29,03 GiB = 271 808 token, 1,04×"),
            ("GPU memory", "Bộ nhớ GPU",
             "T4: weights + non-torch 4.38 GiB, peak activation 2.12 GiB, CUDA graphs 0.30 + 0.18 GiB; "
             "A100: 4.37 GiB, 2.15 GiB, 0.30 + 0.20 GiB",
             "T4: trọng số + non-torch 4,38 GiB, activation đỉnh 2,12 GiB, CUDA graph 0,30 + 0,18 GiB; "
             "A100: 4,37 GiB, 2,15 GiB, 0,30 + 0,20 GiB"),
            ("Time", "Thời gian", "T4: server start 4 min 5 s; A100: 3 min 47 s (weights 1.3 s, multimodal "
             "warmup 14.5 s, torch.compile 40.9 s, first FlashInfer JIT ~1 min 46 s, CUDA graphs ~11 s)",
             "T4: bật server 4 phút 5 giây; A100: 3 phút 47 giây (trọng số 1,3 giây, warmup đa phương thức "
             "14,5 giây, torch.compile 40,9 giây, FlashInfer JIT lần đầu ~1 phút 46 giây, CUDA graph ~11 giây)"),
            ("Speed", "Tốc độ", "T4: ~44 tokens/s single, ~318–335 tokens/s with 8; A100: ~216 single, "
             "~1 165–1 365 with 8", "T4: ~44 token/s một request, ~318–335 token/s với 8; A100: ~216 một request, "
             "~1 165–1 365 với 8"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "26/26 on both GPUs: no reasoning; tool calls via `hermes`; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; streaming; 8 concurrent requests; image base64 and multiple "
             "images; document/table OCR to JSON; 2D grounding; counting and spatial reasoning; chart reasoning; "
             "visual coding; multi-turn with an image; tool call from image content; Responses API with an image; "
             "video OCR/order/approximate timestamp and streaming; image + video; `http(s)` media URLs rejected; "
             "secret found in an 18 044-token prompt (T4: 56.6 s uncached; A100: 0.9 s on the first test). On A100, "
             "the final run located the 2-second video boundary exactly and mixed image + video passed twice; on T4, "
             "these two quality checks varied across runs",
             "26/26 trên cả hai GPU: không có reasoning; gọi hàm qua `hermes`; JSON schema; "
             "Anthropic `/v1/messages`; `/tokenize`; stream; 8 request song song; ảnh base64 và nhiều ảnh; OCR tài "
             "liệu/bảng sang JSON; grounding 2D; đếm và suy luận không gian; hiểu biểu đồ; visual coding; đa lượt "
             "có ảnh; tool call từ nội dung ảnh; Responses API với ảnh; OCR/thứ tự/timestamp gần đúng và stream "
             "video; trộn ảnh + video; link media `http(s)` bị từ chối; tìm đúng mã trong prompt 18 044 token "
             "(T4: 56,6 giây khi chưa cache; A100: 0,9 giây ở lần thử đầu). Trên A100, lần xác nhận cuối định vị đúng "
             "mốc video giây 2 và trộn ảnh + video qua hai lần; trên T4, hai phép chất lượng này thay đổi giữa các lần"),
        ],
    },
    ("colab", "qwen3_vl_2b_thinking"): {
        "summary": ("T4, 2026-11-11", "T4, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × Tesla T4 (15 GB), driver 580.82.07",
             "1 × Tesla T4 (15 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU",
             "0.33.0 / float16 / 1, reasoning parser `qwen3`", "0.33.0 / float16 / 1, reasoning parser `qwen3`"),
            ("Context (`auto`)", "Context (`auto`)", "61 808 tokens (reduced from the model maximum 262 144)",
             "61 808 token (giảm từ tối đa 262 144 của model)"),
            ("KV cache", "KV cache", "6.6 GiB = 61 823 tokens, 1.00 full-length request at once",
             "6,6 GiB = 61 823 token, 1,00 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU",
             "weights + non-torch 4.38 GiB, peak activation 2.12 GiB, CUDA graphs 0.30 + 0.18 GiB",
             "trọng số + non-torch 4,38 GiB, activation đỉnh 2,12 GiB, CUDA graph 0,30 + 0,18 GiB"),
            ("Time", "Thời gian", "server start 4 min 2 s (weights 17 s, multimodal warmup 28 s, "
             "torch.compile 45 s, CUDA graphs ~23 s)", "bật server 4 phút 2 giây (trọng số 17 giây, "
             "warmup đa phương thức 28 giây, torch.compile 45 giây, CUDA graph ~23 giây)"),
            ("Speed", "Tốc độ", "~42 tokens/second for one request; ~185–357 tokens/second total with 8 at once",
             "~42 token/giây cho một request; ~185–357 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "24/26 checks passed. Reasoning is always on and is separated into `message.reasoning`; tool calls via "
             "`hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; image base64 and multiple "
             "images; document/table OCR to JSON; counting and spatial reasoning; chart reasoning; visual coding; "
             "tool call from image content; Responses API with an image; video OCR/order/timestamp and streaming; "
             "image + video; `http(s)` media URLs rejected; secret found in an 18 046-token prompt (17–75 s, "
             "depending on reasoning length). Limitations: 2D grounding returned pixel coordinates or exhausted "
             "the reasoning budget before a box; image multi-turn was correct in 2/4 runs and otherwise confused "
             "the visible code with HTML or repeated until the token limit. It also sometimes copied a requested "
             "answer placeholder literally. Deterministic sampling is used for capability checks",
             "24/26 phép thử qua. Luôn suy nghĩ và reasoning được tách vào `message.reasoning`; gọi hàm qua "
             "`hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; ảnh base64 và nhiều ảnh; OCR tài "
             "liệu/bảng sang JSON; đếm và suy luận không gian; hiểu biểu đồ; visual coding; tool call từ nội dung "
             "ảnh; Responses API với ảnh; OCR/thứ tự/timestamp và stream video; trộn ảnh + video; link media "
             "`http(s)` bị từ chối; tìm đúng mã trong prompt 18 046 token (17–75 giây tuỳ độ dài reasoning). Giới "
             "hạn: grounding 2D trả tọa độ pixel hoặc hết ngân sách reasoning trước bbox; multi-turn có ảnh đúng "
             "2/4 lần, các lần khác nhầm mã nhìn thấy thành HTML hoặc lặp tới giới hạn token. Model cũng có lúc chép "
             "nguyên placeholder câu trả lời. Phép thử chức năng dùng sampling xác định"),
        ],
    },
    ("kaggle", "qwen3_vl_2b_instruct"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel)",
             "0.33.0 / float16 / 2 (tensor parallel)"),
            ("Context (`auto`)", "Context (`auto`)", "170 256 tokens (reduced from the model maximum 262 144)",
             "170 256 token (giảm từ tối đa 262 144 của model)"),
            ("KV cache", "KV cache", "18.2 GiB total (9.09 GiB per GPU) = 170 272 tokens, "
             "1.00 full-length request at once", "18,2 GiB tổng (9,09 GiB mỗi GPU) = 170 272 token, "
             "1,00 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)",
             "weights + non-torch 2.29 GiB, peak activation 1.72 GiB, CUDA graphs 0.30 + 0.18 GiB",
             "trọng số + non-torch 2,29 GiB, activation đỉnh 1,72 GiB, CUDA graph 0,30 + 0,18 GiB"),
            ("Time", "Thời gian", "server start 3 min 39 s (weights 8 s, multimodal warmup 16 s, "
             "torch.compile 45 s, CUDA graphs ~23 s)", "bật server 3 phút 39 giây (trọng số 8 giây, "
             "warmup đa phương thức 16 giây, torch.compile 45 giây, CUDA graph ~23 giây)"),
            ("Speed", "Tốc độ", "~71 tokens/second for one request; ~498 tokens/second total with 8 at once",
             "~71 token/giây cho một request; ~498 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "12/12 standard checks: no reasoning; tool calls via `hermes`; JSON schema; Anthropic `/v1/messages`; "
             "`/tokenize`; streaming; image base64; `http(s)` image URL rejected; secret found in an 18 000-token "
             "prompt (29.5 s). Extended checks passed: multiple images, document OCR to JSON, 2D grounding, spatial "
             "reasoning, visual coding, multi-turn with an image, tool call from image content, Responses API with "
             "text and images, video OCR/order/timestamp, image + video, and video streaming; `http(s)` video URL "
             "rejected. A short video question confused the last colour, while the full three-segment prompt was correct",
             "12/12 phép thử chuẩn: không có reasoning; gọi hàm qua `hermes`; JSON schema; Anthropic "
             "`/v1/messages`; `/tokenize`; stream; ảnh base64; link ảnh `http(s)` bị từ chối; tìm đúng mã "
             "trong prompt 18 000 token (29,5 giây). Phép thử mở rộng qua: nhiều ảnh, OCR tài liệu sang "
             "JSON, grounding 2D, không gian, visual coding, đa lượt có ảnh, tool call từ nội dung ảnh, Responses "
             "API với văn bản và ảnh, OCR/thứ tự/timestamp video, trộn ảnh + video và stream video; link video "
             "`http(s)` bị từ chối. Câu hỏi video quá ngắn nhầm màu cuối, prompt đủ ba đoạn trả đúng"),
        ],
    },
    ("kaggle", "qwen3_vl_2b_thinking"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU",
             "0.33.0 / float16 / 2 (tensor parallel), reasoning parser `qwen3`",
             "0.33.0 / float16 / 2 (tensor parallel), reasoning parser `qwen3`"),
            ("Context (`auto`)", "Context (`auto`)", "170 256 tokens (reduced from the model maximum 262 144)",
             "170 256 token (giảm từ tối đa 262 144 của model)"),
            ("KV cache", "KV cache", "18.2 GiB total (9.09 GiB per GPU) = 170 272 tokens, "
             "1.00 full-length request at once", "18,2 GiB tổng (9,09 GiB mỗi GPU) = 170 272 token, "
             "1,00 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)",
             "weights + non-torch 2.29 GiB, peak activation 1.72 GiB, CUDA graphs 0.30 + 0.18 GiB",
             "trọng số + non-torch 2,29 GiB, activation đỉnh 1,72 GiB, CUDA graph 0,30 + 0,18 GiB"),
            ("Time", "Thời gian", "server start ~3 min 35 s, same in two runs (weights 7 s, multimodal warmup "
             "15 s, torch.compile 41 s, CUDA graphs ~21 s)", "bật server ~3 phút 35 giây, hai lần chạy như nhau "
             "(trọng số 7 giây, warmup đa phương thức 15 giây, torch.compile 41 giây, CUDA graph ~21 giây)"),
            ("Speed", "Tốc độ", "~71–74 tokens/second for one request; ~452 tokens/second total with 8 at once",
             "~71–74 token/giây cho một request; ~452 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "24/26 checks passed. Reasoning is always on and is separated into `message.reasoning`; tool calls via "
             "`hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; image base64 and multiple "
             "images; document/table OCR to JSON; counting and spatial reasoning; chart reasoning; visual coding; "
             "tool call from image content; Responses API with an image; video OCR/order/timestamp and streaming; "
             "image + video; `http(s)` media URLs rejected; secret found in an 18 046-token prompt (38.9 s). "
             "Limitations: 2D grounding returned wrong coordinates (`0,0,200,200` instead of about "
             "`100,200,350,533`); image multi-turn gave the wrong answer; it also sometimes copied a requested "
             "answer placeholder literally. Deterministic sampling is used for capability checks",
             "24/26 phép thử qua. Luôn suy nghĩ và reasoning được tách vào `message.reasoning`; gọi hàm qua "
             "`hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; ảnh base64 và nhiều ảnh; OCR tài "
             "liệu/bảng sang JSON; đếm và suy luận không gian; hiểu biểu đồ; visual coding; tool call từ nội dung "
             "ảnh; Responses API với ảnh; OCR/thứ tự/timestamp và stream video; trộn ảnh + video; link media "
             "`http(s)` bị từ chối; tìm đúng mã trong prompt 18 046 token (38,9 giây). Giới hạn: grounding 2D trả "
             "sai tọa độ (`0,0,200,200` thay vì khoảng `100,200,350,533`); multi-turn có ảnh trả sai; model cũng có "
             "lúc chép nguyên placeholder câu trả lời. Phép thử chức năng dùng sampling xác định"),
        ],
    },
    ("colab", "qwen3_5_4b"): {
        "summary": ("L4, 2026-11-11", "L4, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "1 × NVIDIA L4 (22.5 GB), driver 580.82.07",
             "1 × NVIDIA L4 (22,5 GB), driver 580.82.07"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / bfloat16 / 1, `--max-num-seqs 128`",
             "0.33.0 / bfloat16 / 1, `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "9.19 GiB = 297 061 tokens, 1.13 full-length requests at once",
             "9,19 GiB = 297 061 token, 1,13 request dài tối đa cùng lúc"),
            ("GPU memory", "Bộ nhớ GPU",
             "weights + non-torch 8.73 GiB, peak activation 1.91 GiB, CUDA graphs 0.27 + 0.14 GiB",
             "trọng số + non-torch 8,73 GiB, activation đỉnh 1,91 GiB, CUDA graph 0,27 + 0,14 GiB"),
            ("Time", "Thời gian",
             "server start 5 min 53 s (weights 7 s, torch.compile 32 s, initial profiling/warmup 94 s, "
             "FlashInfer JIT ~1 min 50 s, CUDA graphs 24 s)",
             "bật server 5 phút 53 giây (trọng số 7 giây, torch.compile 32 giây, profiling/warmup ban đầu "
             "94 giây, FlashInfer JIT ~1 phút 50 giây, CUDA graph 24 giây)"),
            ("Speed", "Tốc độ", "~29 tokens/second for one request (with thinking); ~194 tokens/second total with 8 at once",
             "~29 token/giây cho một request (có suy nghĩ); ~194 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "13/13 checks. no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on by default (911 tokens, "
             "32 s); thinking off via both switches; tool calling via `qwen3_coder`; JSON schema; Anthropic "
             "`/v1/messages`; `/tokenize`; streaming; 8 concurrent requests; `http(s)` image URL rejected; secret "
             "found in a 16 000-token prompt (3.7 s). Base64 image: text and both colours right, the rectangle "
             "called a square",
             "13/13 phép thử. không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định có suy nghĩ "
             "(911 token, 32 giây); tắt suy nghĩ bằng cả hai cách; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic "
             "`/v1/messages`; `/tokenize`; stream; 8 request song song; link ảnh `http(s)` bị từ chối; tìm đúng mã "
             "trong prompt 16 000 token (3,7 giây). Ảnh base64: đúng chữ và cả hai màu, hình chữ nhật bị gọi là "
             "hình vuông"),
        ],
    },
    ("kaggle", "qwen3_5_4b"): {
        "summary": ("T4 x2, 2026-11-11", "T4 x2, 11/11/2026"),
        "rows": [
            ("GPU", "GPU", "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13",
             "2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13"),
            ("vLLM / dtype / GPUs", "vLLM / dtype / số GPU", "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`",
             "0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128`"),
            ("Context (`auto`)", "Context (`auto`)", "262 144 tokens (the model maximum)", "262 144 token (tối đa của model)"),
            ("KV cache", "KV cache", "11.9 GiB (5.96 GiB per GPU) = 385 659 tokens, 1.47 full-length requests at once",
             "11,9 GiB (5,96 GiB mỗi GPU) = 385 659 token, 1,47 request dài tối đa cùng lúc"),
            ("GPU memory (per GPU)", "Bộ nhớ GPU (mỗi GPU)", "weights 4.38 GiB, CUDA graphs 0.58 + 0.22 GiB",
             "trọng số 4,38 GiB, CUDA graph 0,58 + 0,22 GiB"),
            ("Time", "Thời gian",
             "server start ~11 min (weights 25 s, torch.compile 31 s, ~7 min of unlogged kernel compilation, "
             "CUDA graphs 55 s)",
             "bật server ~11 phút (trọng số 25 giây, torch.compile 31 giây, ~7 phút biên dịch kernel không ghi log, "
             "CUDA graph 55 giây)"),
            ("Speed", "Tốc độ", "~44 tokens/second for one request (with thinking); ~258 tokens/second total with 8 at once",
             "~44 token/giây cho một request (có suy nghĩ); ~258 token/giây tổng khi 8 request cùng lúc"),
            ("Checked through the public URL", "Đã kiểm qua URL công khai",
             "no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on by default (791 tokens, 18 s); tool "
             "calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image "
             "URL rejected; secret found in a 16 000-token prompt (~20 s). Base64 image: text and both colours right, "
             "the rectangle called a square",
             "không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định có suy nghĩ (791 token, 18 giây); gọi "
             "hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị "
             "từ chối; tìm đúng mã trong prompt 16 000 token (~20 giây). Ảnh base64: đúng chữ và cả hai màu, hình chữ "
             "nhật bị gọi là hình vuông"),
        ],
    },
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
}

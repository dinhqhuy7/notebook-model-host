"""Retrieval models: model families and models (they score or embed text for search).

Add a model: a line in MODELS (its "kind" must be "retrieval"), then `python tools/build_notebooks.py --list` shows it.
"""

import json

# The prompt Qwen trained Qwen3-Reranker with, as a vLLM chat template for scoring. The text below is copied byte for byte
# from vLLM (examples/pooling/score/template/qwen3_reranker.jinja, v0.33.0, Apache-2.0). `instruction` comes from the
# request; without it the default instruction in the template is used.
_VLLM_TEMPLATE = '''\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction | default(instruct | default(messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query", true), true), true) }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>

'''
# Jinja drops the last newline of a template, so vLLM's file renders a prompt that ends with "</think>\n". Qwen's prompt
# ends with "</think>\n\n" (a different token), so one more newline keeps the prompt exactly as the model was trained.
QWEN3_RERANKER_TEMPLATE = _VLLM_TEMPLATE + "\n"
QWEN3_RERANKER_PROMPT = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n")  # the prompt in Qwen's model card, for the test
VLLM_TEMPLATE_SHA256 = "e1ee98e69aab7b2da366edf1c50efcef37e34b4a0c50fb816336213e68d9047a"  # of _VLLM_TEMPLATE

# Qwen3-Reranker is a Qwen3ForCausalLM that answers "yes" or "no". The overrides turn it into a one-output classifier on
# those two tokens (relevance = p(yes) / (p(yes) + p(no))), which is how vLLM serves the original checkpoint.
_RERANKER_OVERRIDES = {"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"],
                       "is_original_qwen3_reranker": True}

# The prompt of Qwen3-VL-Reranker as a vLLM chat template for scoring, copied byte for byte from vLLM
# (examples/pooling/score/template/qwen3_vl_reranker.jinja, v0.33.0, Apache-2.0). It renders Qwen's prompt for the model
# (system line, `<Instruct>: ...<Query>:...\n<Document>:...`, then `<|im_start|>assistant\n`; no `<think>` block, unlike the text
# rerankers). vLLM puts the image and video placeholders into the query and document text before the template runs. Here
# the last newline of the file is the one Jinja drops, so the file renders exactly Qwen's prompt.
QWEN3_VL_RERANKER_TEMPLATE = '''\
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction | default(instruct | default(messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a search query, retrieve relevant candidates that answer the query.", true), true), true) }}<Query>:{{
    messages
    | selectattr("role", "eq", "query")
    | map(attribute="content")
    | first
}}
<Document>:{{
    messages
    | selectattr("role", "eq", "document")
    | map(attribute="content")
    | first
}}<|im_end|>
<|im_start|>assistant

'''
VL_TEMPLATE_SHA256 = "df956086e3ef710a6f792a99ad3f7a7ec717f19114aba061304117fa688dee26"  # of QWEN3_VL_RERANKER_TEMPLATE
QWEN3_VL_RERANKER_PROMPT = (  # Qwen's prompt for one pair (qwen3_vl_reranker.py of the model card), for the test
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    "<Instruct>: {instruction}<Query>:{query}\n<Document>:{document}<|im_end|>\n<|im_start|>assistant\n")

# Same recipe as the text rerankers, on the vision-language model (Qwen3VLForSequenceClassification).
_VL_RERANKER_OVERRIDES = {"architectures": ["Qwen3VLForSequenceClassification"], "classifier_from_token": ["no", "yes"],
                          "is_original_qwen3_reranker": True}

# Qwen3-Reranker family (0.6B, 4B): the same serving, the same functions; each model adds its own notes and memory.
_QWEN3_RERANKER = {
    "kind": "retrieval",
    "task": "rerank",
    "template_file": "qwen3_reranker.jinja",
    "template": QWEN3_RERANKER_TEMPLATE,
    "input": ("text (a query and the documents to rank)", "văn bản (câu hỏi và các tài liệu cần xếp hạng)"),
    "context": 40960,
    "flags": ["--runner", "pooling", "--hf-overrides", json.dumps(_RERANKER_OVERRIDES)],
    "flag_notes": """\
- `--runner pooling`: the model is served for scoring, not for text generation.
- `--hf-overrides ...`: turns the original `Qwen3ForCausalLM` checkpoint into a scorer of the `yes` and `no` tokens, the
  way Qwen defines this reranker. The weights are the official ones.
- `--chat-template`: the cell below writes the prompt Qwen trained the reranker with (system line, `<Instruct>`,
  `<Query>`, `<Document>`) to a file. Callers can set the instruction in each request.
- No reasoning or tool parser: the model writes no text, so `/v1/chat/completions` does not exist.""",
    "sampling": "this model generates no text: it returns a relevance score for each query-document pair",
    # What the model can do, and whether this host gives it: (function, function in Vietnamese, API, True or
    # (reason, reason in Vietnamese)). Every function of the model card must be listed.
    "functions": [
        ("Rerank documents for a query", "Xếp hạng tài liệu theo câu hỏi", "`POST /rerank`, `/v1/rerank`, `/v2/rerank`", True),
        ("Score query-document pairs", "Chấm điểm từng cặp câu hỏi - tài liệu", "`POST /score`, `/v1/score`", True),
        ("Task instruction per request", "Instruction theo tác vụ trong từng request", ("`instruction` in `/rerank` and `/score`", "`instruction` trong `/rerank` và `/score`"), True),
        ("Cut long queries and documents", "Cắt câu hỏi và tài liệu dài", "`max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`)", True),
        ("Text generation / chat", "Sinh văn bản / chat", "`/v1/chat/completions`",
         ("the model does not generate text", "model không sinh văn bản")),
        ("Embeddings", "Embedding", "`/v1/embeddings`",
         ("not an embedding model; Qwen3-Embedding is", "không phải model embedding; Qwen3-Embedding mới là")),
    ],
}

QWEN3_RERANKER_0_6B = {
    **_QWEN3_RERANKER,
    # The tested Colab notebook has "one GPU" and its notes written in, and is locked: keep that notebook as it is. Kaggle
    # always gives 2 x T4 and counts both against the quota, so the Kaggle notebook uses both.
    "one_gpu_on": ("colab",),
    "notes_by_platform": {"colab": "Reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one "
             "(`/rerank` returns them sorted, `/score` in the order you sent). Each pair is scored with the prompt Qwen "
             "trained it with; set `instruction` in the request to describe your task (Qwen reports 1% to 5% better "
             "results with a task-specific instruction, in English); without it the default is \"Given a web search "
             "query, retrieve relevant passages that answer the query\". Multilingual. The model is small, so the "
             "notebook uses one GPU, even on Kaggle's 2 × T4. It writes no text (no chat). License: Apache 2.0. "
             "`float16` on T4 has not been tested with this model yet."},
    "notes": "Reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one "
             "(`/rerank` returns them sorted, `/score` in the order you sent). Each pair is scored with the prompt Qwen "
             "trained it with; set `instruction` in the request to describe your task (Qwen reports 1% to 5% better "
             "results with a task-specific instruction, in English); without it the default is \"Given a web search "
             "query, retrieve relevant passages that answer the query\". Multilingual. On Kaggle's 2 × T4 the model is "
             "split across both GPUs (Kaggle counts both against your quota). It writes no text (no chat). License: Apache 2.0. "
             "Tested on Colab T4 (`float16`); the split across two GPUs on Kaggle has not been tested yet.",
    "notes_vi": "Reranker: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu "
                "(`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Mỗi cặp được chấm bằng đúng prompt "
                "Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn (Qwen cho biết kết quả tốt "
                "hơn 1% đến 5% khi có instruction riêng cho tác vụ, viết bằng tiếng Anh); không gửi thì mặc định là "
                "\"Given a web search query, retrieve relevant passages that answer the query\". Đa ngôn ngữ. Trên 2 × T4 của "
                "Kaggle model được chia ra cả hai GPU (Kaggle tính quota cho cả hai). Model không viết văn bản (không "
                "chat). Giấy phép: Apache 2.0. Đã chạy thử trên Colab T4 (`float16`); việc chia hai GPU trên Kaggle chưa chạy thử.",
    "extra_gib": 0.3,  # estimate, not measured yet: the classifier head is built from the full-vocabulary output matrix
}

# Qwen3-Reranker-4B (36 layers, 8 KV heads x 128): the KV cache costs 144 KiB per token, so a T4 (~3 GiB left) gets ~24 000
# tokens of context and the notebook lowers the context to what fits. Scores of the model card: MTEB-R 69.76 (0.6B: 65.80),
# CMTEB-R 75.94 (71.31), MMTEB-R 72.74 (66.36), MTEB-Code 81.20 (73.42).
QWEN3_RERANKER_4B = {
    **_QWEN3_RERANKER,
    # The tested Colab notebook is locked and has its first estimate (about 24 000 tokens on a T4) written in: that notebook stays as it is.
    "notes_by_platform": {"colab": "Reranker, the larger and more accurate one of the family: send a query and a list of documents, get a "
             "relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). "
             "Qwen's tables put it ahead of Qwen3-Reranker-0.6B (for example MTEB-R 69.76 against 65.80). Each pair is "
             "scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, "
             "otherwise the default is \"Given a web search query, retrieve relevant passages that answer the query\". "
             "Multilingual. The KV cache costs 144 KiB per token, so on a single T4 vLLM lowers the context to what fits "
             "(about 24 000 tokens; the model allows 40 960): see the memory table. On Kaggle's 2 × T4 the model is split "
             "across both GPUs (tensor parallel), which leaves room for the full context. It writes no text (no chat). "
             "License: Apache 2.0. `float16` on T4 and the split across two GPUs have not been tested with this model yet."},
    "notes": "Reranker, the larger and more accurate one of the family: send a query and a list of documents, get a "
             "relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). "
             "Qwen's tables put it ahead of Qwen3-Reranker-0.6B (for example MTEB-R 69.76 against 65.80). Each pair is "
             "scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, "
             "otherwise the default is \"Given a web search query, retrieve relevant passages that answer the query\". "
             "Multilingual. The KV cache costs 144 KiB per token, so on a single T4 vLLM lowers the context to 32 128 "
             "tokens (measured; the model allows 40 960). Scoring is slower than the 0.6B (about 9 documents of 200 tokens a "
             "second on a T4): keep each pair well under ~10 000 tokens on a T4, because the tunnel cuts a request after "
             "100 seconds. On Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel), which leaves room for "
             "the full context. It writes no text (no chat). License: Apache 2.0. Tested on Colab T4 (`float16`); the split "
             "across two GPUs on Kaggle has not been tested yet.",
    "notes_vi": "Reranker, cỡ lớn và chính xác hơn của họ này: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ "
                "0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Bảng điểm "
                "của Qwen đặt nó trên Qwen3-Reranker-0.6B (ví dụ MTEB-R 69,76 so với 65,80). Mỗi cặp được chấm bằng đúng "
                "prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc "
                "định là \"Given a web search query, retrieve relevant passages that answer the query\". Đa ngôn ngữ. KV "
                "cache tốn 144 KiB mỗi token, nên trên một T4 vLLM tự giảm context còn 32 128 token (đã đo; "
                "model cho phép 40 960). Chấm điểm chậm hơn bản 0,6B (khoảng 9 tài liệu 200 token mỗi giây trên T4): trên T4 "
                "nên giữ mỗi cặp dưới ~10 000 token vì tunnel cắt request sau 100 giây. Trên 2 × T4 của Kaggle model được chia "
                "ra cả hai GPU (tensor parallel), đủ chỗ cho context đầy đủ. Model không viết văn bản (không chat). Giấy "
                "phép: Apache 2.0. Đã chạy thử trên Colab T4 (`float16`); việc chia hai GPU trên Kaggle chưa chạy thử.",
    "extra_gib": 0.7,  # estimate, not measured yet: the classifier head is built from the 151 669 x 2 560 output matrix
}

# Qwen3-Reranker-8B (36 layers, 8 KV heads x 128, 144 KiB of KV cache per token, 15.25 GiB of weights, embeddings NOT tied unlike the
# 0.6B and 4B). It does not fit one T4, so Kaggle's 2 x T4 need the split across both GPUs. Scores of the model card: MTEB-R 69.02
# (4B: 69.76), CMTEB-R 77.45 (75.94), MMTEB-R 72.94 (72.74), MLDR 70.19 (69.97), MTEB-Code 81.22 (81.20).
QWEN3_RERANKER_8B = {
    **_QWEN3_RERANKER,
    "notes": "Reranker, the largest of the family: send a query and a list of documents, get a relevance score from 0 to 1 "
             "for each one (`/rerank` returns them sorted, `/score` in the order you sent). Qwen's tables put it only "
             "slightly ahead of Qwen3-Reranker-4B (CMTEB-R 77.45 against 75.94, MTEB-R 69.02 against 69.76) for twice the "
             "memory: try the 4B first. Each pair is scored with the prompt Qwen trained it with; set `instruction` in the "
             "request to describe your task, otherwise the default is \"Given a web search query, retrieve relevant "
             "passages that answer the query\". Multilingual. The weights need 15.25 GiB, so **a T4 is not enough**: on "
             "Colab use an A100 (an L4 leaves only about 2 GiB for the KV cache), on Kaggle's 2 × T4 the model is split "
             "across both GPUs (tensor parallel) and about 48 000 tokens of context fit. The KV cache costs 144 KiB per "
             "token. It writes no text (no chat). License: Apache 2.0. Not tested yet: `float16` on T4, the split across two "
             "GPUs, and this model's separate (not tied) output embeddings.",
    "notes_vi": "Reranker, cỡ lớn nhất của họ này: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho "
                "từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Bảng điểm của Qwen cho "
                "thấy nó chỉ nhỉnh hơn Qwen3-Reranker-4B một chút (CMTEB-R 77,45 so với 75,94, MTEB-R 69,02 so với 69,76) "
                "mà tốn gấp đôi bộ nhớ: nên thử bản 4B trước. Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt "
                "`instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là \"Given a web search query, "
                "retrieve relevant passages that answer the query\". Đa ngôn ngữ. Trọng số 15,25 GiB nên **một T4 không "
                "đủ**: trên Colab dùng A100 (L4 chỉ còn khoảng 2 GiB cho KV cache), trên 2 × T4 của Kaggle model được chia "
                "ra cả hai GPU (tensor parallel) và đủ khoảng 48 000 token context. KV cache tốn 144 KiB mỗi token. Model "
                "không viết văn bản (không chat). Giấy phép: Apache 2.0. Chưa thử: `float16` trên T4, chia hai GPU, và đầu ra "
                "embedding riêng (không gắn chung) của model này.",
    "extra_gib": 1.2,  # estimate, not measured yet: the classifier head is built from the 151 669 x 4 096 output matrix
}

# Qwen3-VL-Reranker-2B (28 layers, 8 KV heads x 128, 112 KiB of KV cache per token, 3.96 GiB with the vision encoder). The card says
# 32K of context; the config says 262 144, so vLLM's `auto` may open more than the card promises where memory allows.
_QWEN3_VL_RERANKER_FUNCTIONS = [
    ("Rerank text documents for a text query", "Xếp hạng tài liệu văn bản theo câu hỏi văn bản",
     "`POST /rerank`, `/v1/rerank`, `/v2/rerank`", True),
    ("Rerank images and screenshots", "Xếp hạng ảnh và ảnh chụp màn hình",
     ("`documents`: `{\"content\": [{\"type\": \"image_url\", ...}]}` in `/rerank`, `/score` (a base64 `data:` URL)",
      "`documents`: `{\"content\": [{\"type\": \"image_url\", ...}]}` trong `/rerank`, `/score` (URL `data:` base64)"), True),
    ("Rerank videos", "Xếp hạng video",
     ("`documents`: `{\"content\": [{\"type\": \"video_url\", ...}]}` in `/rerank`, `/score` (a base64 `data:` URL)",
      "`documents`: `{\"content\": [{\"type\": \"video_url\", ...}]}` trong `/rerank`, `/score` (URL `data:` base64)"), True),
    ("Mixed documents (text and image together) and image or video queries", "Tài liệu hỗn hợp (văn bản kèm ảnh) và câu hỏi là ảnh hoặc video",
     ("`query` or `documents` as `{\"content\": [text, image_url, video_url ...]}` in `/rerank`, `/score`",
      "`query` hoặc `documents` dạng `{\"content\": [text, image_url, video_url ...]}` trong `/rerank`, `/score`"), True),
    ("Score query-document pairs", "Chấm điểm từng cặp câu hỏi - tài liệu", "`POST /score`, `/v1/score`", True),
    ("Task instruction per request", "Instruction theo tác vụ trong từng request",
     ("`instruction` in `/rerank` and `/score`", "`instruction` trong `/rerank` và `/score`"), True),
    ("Cut long queries and documents", "Cắt câu hỏi và tài liệu dài", "`max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`)", True),
    ("Image and video processor settings", "Thiết lập bộ xử lý ảnh và video",
     ("`mm_processor_kwargs` in `/rerank`, `/score` (needs `--trust-request-mm-kwargs`: set in the Colab `_update` and Kaggle notebooks, not in the tested Colab notebook yet)",
      "`mm_processor_kwargs` trong `/rerank`, `/score` (cần `--trust-request-mm-kwargs`: đã bật ở bản `_update` Colab và bản Kaggle, chưa có ở bản Colab đã chạy thử)"), True),
    ("Image and video by http(s) link", "Ảnh và video qua link http(s)", "`image_url` / `video_url` `https://...` / `/rerank`",
     ("blocked on purpose: the server never downloads links; send base64",
      "chặn có chủ đích: server không tự tải link; hãy gửi base64")),
    ("Text generation / chat", "Sinh văn bản / chat", "`/v1/chat/completions`",
     ("the model does not generate text", "model không sinh văn bản")),
    ("Embeddings", "Embedding", "`/v1/embeddings`",
     ("not an embedding model; Qwen3-VL-Embedding is", "không phải model embedding; Qwen3-VL-Embedding mới là")),
]
QWEN3_VL_RERANKER_2B = {
    "kind": "retrieval",
    "task": "rerank",
    "template_file": "qwen3_vl_reranker.jinja",
    "template": QWEN3_VL_RERANKER_TEMPLATE,
    "input": ("text, images, video (queries and documents)", "văn bản, ảnh, video (câu hỏi và tài liệu)"),
    "context": 32768,
    "flags": ["--runner", "pooling", "--hf-overrides", json.dumps(_VL_RERANKER_OVERRIDES),
              "--allowed-media-domains", "none.invalid", "--trust-request-mm-kwargs"],
    "flag_notes": """\
- `--runner pooling`: the model is served for scoring, not for text generation.
- `--hf-overrides ...`: turns the original `Qwen3VLForConditionalGeneration` checkpoint into a scorer of the `yes` and `no`
  tokens, the way Qwen defines this reranker. The weights are the official ones.
- `--chat-template`: the cell below writes the prompt Qwen trained the reranker with (system line, `<Instruct>`,
  `<Query>`, `<Document>`) to a file. Callers can set the instruction in each request.
- `--allowed-media-domains none.invalid`: the server never downloads image/video URLs itself;
  callers send media from their machine as base64 `data:` URLs.
- `--trust-request-mm-kwargs`: a request may set how images and video are processed (`mm_processor_kwargs`), for example
  the frame rate of a video. vLLM refuses these by default; here every caller holds the API key.
- No reasoning or tool parser: the model writes no text, so `/v1/chat/completions` does not exist.""",
    "sampling": "this model generates no text: it returns a relevance score for each query-document pair",
    "notes": "Multimodal reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one "
             "(`/rerank` returns them sorted, `/score` in the order you sent). The query and each document can be text, "
             "an image or a screenshot, a video, or a mix (`{\"content\": [...]}`); media goes in as base64 `data:` URLs. "
             "Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your "
             "task, otherwise the default is \"Given a search query, retrieve relevant candidates that answer the query.\" "
             "Qwen states 32K of context; vLLM's `auto` may open more where memory allows, the memory table shows what "
             "fits. vLLM reads at most 32 frames of a video by default; a request can set how images and video are processed with `mm_processor_kwargs` (the notebook starts the server with `--trust-request-mm-kwargs`, which vLLM needs for that). "
             "Multilingual (30+ languages). It writes no text (no chat). On Kaggle's 2 × T4 the model is split across "
             "both GPUs (tensor parallel). License: Apache 2.0. Tested on Colab T4 (`float16`) without the `--trust-request-mm-kwargs` flag; not tested yet: that flag "
             "and the split across two GPUs.",
    "notes_vi": "Reranker đa phương thức: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài "
                "liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Câu hỏi và từng tài liệu có thể là "
                "văn bản, ảnh hoặc ảnh chụp màn hình, video, hoặc kết hợp (`{\"content\": [...]}`); media gửi dạng URL "
                "`data:` base64. Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để "
                "mô tả tác vụ của bạn, không gửi thì mặc định là \"Given a search query, retrieve relevant candidates that "
                "answer the query.\" Qwen ghi context 32K; `auto` của vLLM có thể mở nhiều hơn khi đủ bộ nhớ, bảng bộ nhớ "
                "cho biết vừa bao nhiêu. vLLM mặc định đọc tối đa 32 khung hình của một video; request có thể đặt cách xử lý ảnh và video bằng "
                "`mm_processor_kwargs` (notebook bật `--trust-request-mm-kwargs` cho server, vLLM cần cờ này). Đa ngôn ngữ (hơn 30 ngôn ngữ). Model không viết văn bản (không chat). Trên 2 × T4 "
                "của Kaggle model được chia ra cả hai GPU (tensor parallel). Giấy phép: Apache 2.0. Đã chạy thử trên Colab T4 (`float16`) khi chưa có cờ "
                "`--trust-request-mm-kwargs`; chưa thử: cờ đó và chia hai GPU.",
    "extra_gib": 1.5,  # estimate, not measured yet: vision encoder activations and the classifier head built from the vocabulary
    "rerank_info_extra": "Queries and documents can also be images or videos: {{\"content\": [{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"data:image/png;base64,...\"}}}}]}}",
    "functions": _QWEN3_VL_RERANKER_FUNCTIONS,
}


MODELS = [
    {"slug": "qwen3_reranker_0_6b", "model_id": "Qwen/Qwen3-Reranker-0.6B", "served_name": "Qwen3-Reranker-0.6B",
     "weights_gib": 1.12, "download_gb": 2, **QWEN3_RERANKER_0_6B},
    {"slug": "qwen3_reranker_4b", "model_id": "Qwen/Qwen3-Reranker-4B", "served_name": "Qwen3-Reranker-4B",
     "weights_gib": 7.49, "download_gb": 8, **QWEN3_RERANKER_4B},
    {"slug": "qwen3_reranker_8b", "model_id": "Qwen/Qwen3-Reranker-8B", "served_name": "Qwen3-Reranker-8B",
     "weights_gib": 15.25, "download_gb": 17, **QWEN3_RERANKER_8B},
    {"slug": "qwen3_vl_reranker_2b", "model_id": "Qwen/Qwen3-VL-Reranker-2B", "served_name": "Qwen3-VL-Reranker-2B",
     "weights_gib": 3.96, "download_gb": 5, **QWEN3_VL_RERANKER_2B},
]

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

QWEN3_RERANKER = {
    "kind": "retrieval",
    "task": "rerank",
    "single_gpu": True,  # a 0.6B model gains nothing from splitting across GPUs
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
    "notes": "Reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one "
             "(`/rerank` returns them sorted, `/score` in the order you sent). Each pair is scored with the prompt Qwen "
             "trained it with; set `instruction` in the request to describe your task (Qwen reports 1% to 5% better "
             "results with a task-specific instruction, in English); without it the default is \"Given a web search "
             "query, retrieve relevant passages that answer the query\". Multilingual. The model is small, so the "
             "notebook uses one GPU, even on Kaggle's 2 × T4. It writes no text (no chat). License: Apache 2.0. "
             "`float16` on T4 has not been tested with this model yet.",
    "notes_vi": "Reranker: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu "
                "(`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Mỗi cặp được chấm bằng đúng prompt "
                "Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn (Qwen cho biết kết quả tốt "
                "hơn 1% đến 5% khi có instruction riêng cho tác vụ, viết bằng tiếng Anh); không gửi thì mặc định là "
                "\"Given a web search query, retrieve relevant passages that answer the query\". Đa ngôn ngữ. Model "
                "nhỏ nên notebook chỉ dùng một GPU, kể cả trên 2 × T4 của Kaggle. Model không viết văn bản (không "
                "chat). Giấy phép: Apache 2.0. `float16` trên T4 chưa thử với model này.",
    "extra_gib": 0.3,  # estimate, not measured yet: the classifier head is built from the full-vocabulary output matrix
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

MODELS = [
    {"slug": "qwen3_reranker_0_6b", "model_id": "Qwen/Qwen3-Reranker-0.6B", "served_name": "Qwen3-Reranker-0.6B",
     "weights_gib": 1.12, "download_gb": 2, **QWEN3_RERANKER},
]

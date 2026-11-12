# Qwen3-Reranker-4B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) · text (a query and the documents to rank) · weights 7.5 GiB (download ~8 GB) · context 40 960 tokens · needs at least ~13 GB of total GPU memory

Reranker, the larger and more accurate one of the family: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). Qwen's tables put it ahead of Qwen3-Reranker-0.6B (for example MTEB-R 69.76 against 65.80). Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, otherwise the default is "Given a web search query, retrieve relevant passages that answer the query". Multilingual. The KV cache costs 144 KiB per token, so on a single T4 vLLM lowers the context to 32 128 tokens (measured; the model allows 40 960). Scoring is slower than the 0.6B (about 9 documents of 200 tokens a second on a T4): keep each pair well under ~10 000 tokens on a T4, because the tunnel cuts a request after 100 seconds. On Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel), which leaves room for the full context. It writes no text (no chat). License: Apache 2.0. Tested on Colab T4 (`float16`); the split across two GPUs on Kaggle has not been tested yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_4b.ipynb](colab_qwen3_reranker_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb) | ✓ T4, 2026-11-13 |
| Kaggle | [kaggle_qwen3_reranker_4b.ipynb](kaggle_qwen3_reranker_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb) | Not tested yet |

## Functions

Everything the model can do, and whether the notebook hosts it. Whether it was run is in the test results below.

| Function | How to call it | Hosted |
|---|---|---|
| Rerank documents for a query | `POST /rerank`, `/v1/rerank`, `/v2/rerank` | ✓ |
| Score query-document pairs | `POST /score`, `/v1/score` | ✓ |
| Task instruction per request | `instruction` in `/rerank` and `/score` | ✓ |
| Cut long queries and documents | `max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`) | ✓ |
| Text generation / chat | `/v1/chat/completions` | ✗ the model does not generate text |
| Embeddings | `/v1/embeddings` | ✗ not an embedding model; Qwen3-Embedding is |

## Estimated memory left for context

| GPU | Platform | Qwen3-Reranker-4B |
|---|---|---|
| T4 | Google Colab | ~3 GiB |
| L4 | Google Colab | ~10 GiB |
| A100 40 GB | Google Colab | ~26 GiB |
| A100 80 GB | Google Colab | ~62 GiB |
| H100 80 GB | Google Colab | ~61 GiB |
| T4 x2 | Kaggle | ~15 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 32 128 tokens (reduced from 40 960 to fit) |
| KV cache | 4.42 GiB = 32 144 tokens: one full-length request at a time |
| GPU memory | weights 7.56 GiB (7.58 GiB with the rest of the non-torch memory), peak activations 1.12 GiB, CUDA graphs 0.37 + 0.08 GiB |
| Time | server start ~3 min 10 s (weights 36 s, torch.compile 37 s, CUDA graphs 15 + 15 s) |
| Speed | ~9 documents/second with 200-token documents (~2 600 tokens/second); 100 short documents in 6.8 s; 8 requests of 16 documents at once in 1.9 s. Two documents totalling ~22 000 tokens took 77 s the first time, close to the tunnel's 100-second limit: on a T4 keep each query-document pair well under ~10 000 tokens or split the work |
| Checked through the public URL | 11/11 checks: no or wrong key → 401 on `/rerank` and `/score`; `/invocations`, `/sleep` and `/v1/chat/completions` → 404; `/rerank` puts the right document first (score 0.998 against 0.0) and `top_n` cuts the list; `/score` gives the same scores in the order sent; a Vietnamese query ranked correctly; `instruction` reaches the prompt (the prompt grows with it, the default text gives identical scores) and moves the score more than on the 0.6B (0.9982 → 0.9587 for a different instruction); `max_tokens_per_doc` cut 1 285 tokens to 108; the right document found among ~22 000 tokens (0.794 against 0.001). The server log shows the chat template was loaded (no missing-template warning) and `float16` was used on T4 |

### Kaggle

Not tested yet.

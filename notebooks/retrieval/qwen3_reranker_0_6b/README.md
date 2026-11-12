# Qwen3-Reranker-0.6B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · text (a query and the documents to rank) · weights 1.1 GiB (download ~2 GB) · context 40 960 tokens · needs at least ~6 GB of total GPU memory

Reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task (Qwen reports 1% to 5% better results with a task-specific instruction, in English); without it the default is "Given a web search query, retrieve relevant passages that answer the query". Multilingual. On Kaggle's 2 × T4 the model is split across both GPUs (Kaggle counts both against your quota). It writes no text (no chat). License: Apache 2.0. Tested on Colab T4 (`float16`); the split across two GPUs on Kaggle has not been tested yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_0_6b.ipynb](colab_qwen3_reranker_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb) | ✓ T4, 2026-11-13 |
| Kaggle | [kaggle_qwen3_reranker_0_6b.ipynb](kaggle_qwen3_reranker_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb) | Not tested yet |

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

| GPU | Platform | Qwen3-Reranker-0.6B |
|---|---|---|
| T4 | Google Colab | ~10 GiB |
| L4 | Google Colab | ~17 GiB |
| A100 40 GB | Google Colab | ~33 GiB |
| A100 80 GB | Google Colab | ~69 GiB |
| H100 80 GB | Google Colab | ~68 GiB |
| T4 x2 | Kaggle | ~22 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 11.35 GiB = 106 240 tokens, 2.59 full-length requests at once |
| GPU memory | weights 1.12 GiB (1.18 GiB with the rest of the non-torch memory), peak activations 0.57 GiB, CUDA graphs 0.05 GiB |
| Time | server start ~2 min (weights 3 s, torch.compile 30 s, CUDA graphs 11 + 5 s) |
| Speed | ~45 documents/second with 200-token documents (~12 000 tokens/second); 100 documents in 2.3 s; 8 requests of 16 documents at once in 0.8 s. Two documents totalling ~22 000 tokens took 30 s the first time (and 1.6 s when repeated, from the prefix cache) |
| Checked through the public URL | 11/11 checks: no or wrong key → 401 on `/rerank` and `/score`; `/invocations`, `/sleep` and `/v1/chat/completions` → 404; `/rerank` puts the right document first (score 0.999 against 0.0) and `top_n` cuts the list; `/score` gives the same scores in the order sent; a Vietnamese query ranked correctly; `instruction` reaches the prompt (the prompt grows with it, and the default text gives identical scores) but moves this small model's score very little (0.9995 → 0.9979 for a different instruction); `max_tokens_per_doc` cut 1 285 tokens to 108; the right document found among ~22 000 tokens (0.645 against 0.002). The server log shows the chat template was loaded (no missing-template warning), `float16` was used on T4 and `--max-model-len auto` worked with the pooling runner |

### Kaggle

Not tested yet.

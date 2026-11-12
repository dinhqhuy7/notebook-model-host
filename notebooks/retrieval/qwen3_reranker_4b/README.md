# Qwen3-Reranker-4B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) · text (a query and the documents to rank) · weights 7.5 GiB (download ~8 GB) · context 40 960 tokens · needs at least ~13 GB of total GPU memory

Reranker, the larger and more accurate one of the family: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). Qwen's tables put it ahead of Qwen3-Reranker-0.6B (for example MTEB-R 69.76 against 65.80). Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, otherwise the default is "Given a web search query, retrieve relevant passages that answer the query". Multilingual. The KV cache costs 144 KiB per token, so on a single T4 vLLM lowers the context to what fits (about 24 000 tokens; the model allows 40 960): see the memory table. On Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel), which leaves room for the full context. It writes no text (no chat). License: Apache 2.0. `float16` on T4 and the split across two GPUs have not been tested with this model yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_4b.ipynb](colab_qwen3_reranker_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb) | Not tested yet |
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

Not tested yet.

### Kaggle

Not tested yet.

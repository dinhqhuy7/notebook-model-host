# Qwen3-Reranker-8B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) · text (a query and the documents to rank) · weights 15.2 GiB (download ~17 GB) · context 40 960 tokens · needs at least ~23 GB of total GPU memory

Reranker, the largest of the family: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). Qwen's tables put it only slightly ahead of Qwen3-Reranker-4B (CMTEB-R 77.45 against 75.94, MTEB-R 69.02 against 69.76) for twice the memory: try the 4B first. Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, otherwise the default is "Given a web search query, retrieve relevant passages that answer the query". Multilingual. The weights need 15.25 GiB, so **a T4 is not enough**: on Colab use an A100 (an L4 leaves only about 2 GiB for the KV cache), on Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel) and about 48 000 tokens of context fit. The KV cache costs 144 KiB per token. It writes no text (no chat). License: Apache 2.0. Not tested yet: `float16` on T4, the split across two GPUs, and this model's separate (not tied) output embeddings.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_8b.ipynb](colab_qwen3_reranker_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_8b/colab_qwen3_reranker_8b.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen3_reranker_8b.ipynb](kaggle_qwen3_reranker_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_8b/kaggle_qwen3_reranker_8b.ipynb) | Not tested yet |

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

| GPU | Platform | Qwen3-Reranker-8B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~2 GiB |
| A100 40 GB | Google Colab | ~18 GiB |
| A100 80 GB | Google Colab | ~54 GiB |
| H100 80 GB | Google Colab | ~53 GiB |
| T4 x2 | Kaggle | ~7 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

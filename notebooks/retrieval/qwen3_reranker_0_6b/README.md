# Qwen3-Reranker-0.6B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · text (a query and the documents to rank) · weights 1.1 GiB (download ~2 GB) · context 40 960 tokens · needs at least ~6 GB of total GPU memory

Reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task (Qwen reports 1% to 5% better results with a task-specific instruction, in English); without it the default is "Given a web search query, retrieve relevant passages that answer the query". Multilingual. The model is small, so the notebook uses one GPU, even on Kaggle's 2 × T4. It writes no text (no chat). License: Apache 2.0. `float16` on T4 has not been tested with this model yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_0_6b.ipynb](colab_qwen3_reranker_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb) | Not tested yet |
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
| T4 x2 | Kaggle | ~10 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

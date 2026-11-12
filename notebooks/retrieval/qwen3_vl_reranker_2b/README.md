# Qwen3-VL-Reranker-2B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · text, images, video (queries and documents) · weights 4.0 GiB (download ~5 GB) · context 32 768 tokens · needs at least ~10 GB of total GPU memory

Multimodal reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). The query and each document can be text, an image or a screenshot, a video, or a mix (`{"content": [...]}`); media goes in as base64 `data:` URLs. Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, otherwise the default is "Given a search query, retrieve relevant candidates that answer the query." Qwen states 32K of context; vLLM's `auto` may open more where memory allows, the memory table shows what fits. vLLM reads at most 32 frames of a video by default; `mm_processor_kwargs` goes to the image and video processor. Multilingual (30+ languages). It writes no text (no chat). On Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel). License: Apache 2.0. Not tested yet: `float16` on T4, the split across two GPUs and video.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen3_vl_reranker_2b.ipynb](kaggle_qwen3_vl_reranker_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb) | Not tested yet |

## Functions

Everything the model can do, and whether the notebook hosts it. Whether it was run is in the test results below.

| Function | How to call it | Hosted |
|---|---|---|
| Rerank text documents for a text query | `POST /rerank`, `/v1/rerank`, `/v2/rerank` | ✓ |
| Rerank images and screenshots | `documents`: `{"content": [{"type": "image_url", ...}]}` in `/rerank`, `/score` (a base64 `data:` URL) | ✓ |
| Rerank videos | `documents`: `{"content": [{"type": "video_url", ...}]}` in `/rerank`, `/score` (a base64 `data:` URL) | ✓ |
| Mixed documents (text and image together) and image or video queries | `query` or `documents` as `{"content": [text, image_url, video_url ...]}` in `/rerank`, `/score` | ✓ |
| Score query-document pairs | `POST /score`, `/v1/score` | ✓ |
| Task instruction per request | `instruction` in `/rerank` and `/score` | ✓ |
| Cut long queries and documents | `max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`) | ✓ |
| Image and video processor settings | `mm_processor_kwargs` in `/rerank`, `/score` (passed to the processor; not tested here yet) | ✓ |
| Image and video by http(s) link | `image_url` / `video_url` `https://...` / `/rerank` | ✗ blocked on purpose: the server never downloads links; send base64 |
| Text generation / chat | `/v1/chat/completions` | ✗ the model does not generate text |
| Embeddings | `/v1/embeddings` | ✗ not an embedding model; Qwen3-VL-Embedding is |

## Estimated memory left for context

| GPU | Platform | Qwen3-VL-Reranker-2B |
|---|---|---|
| T4 | Google Colab | ~6 GiB |
| L4 | Google Colab | ~13 GiB |
| A100 40 GB | Google Colab | ~29 GiB |
| A100 80 GB | Google Colab | ~65 GiB |
| H100 80 GB | Google Colab | ~64 GiB |
| T4 x2 | Kaggle | ~18 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

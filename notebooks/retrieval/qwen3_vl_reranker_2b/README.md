# Qwen3-VL-Reranker-2B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Retrieval models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · text, images, video (queries and documents) · weights 4.0 GiB (download ~5 GB) · context 32 768 tokens · needs at least ~10 GB of total GPU memory

Multimodal reranker: send a query and a list of documents, get a relevance score from 0 to 1 for each one (`/rerank` returns them sorted, `/score` in the order you sent). The query and each document can be text, an image or a screenshot, a video, or a mix (`{"content": [...]}`); media goes in as base64 `data:` URLs. Each pair is scored with the prompt Qwen trained it with; set `instruction` in the request to describe your task, otherwise the default is "Given a search query, retrieve relevant candidates that answer the query." Qwen states 32K of context; vLLM's `auto` may open more where memory allows, the memory table shows what fits. vLLM reads at most 32 frames of a video by default; a request can set how images and video are processed with `mm_processor_kwargs` (the notebook starts the server with `--trust-request-mm-kwargs`, which vLLM needs for that). Multilingual (30+ languages). It writes no text (no chat). On Kaggle's 2 × T4 the model is split across both GPUs (tensor parallel). License: Apache 2.0. Tested on Colab T4 (`float16`) without the `--trust-request-mm-kwargs` flag; not tested yet: that flag and the split across two GPUs.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb) | ✓ T4, 2026-11-13 |
| Kaggle | [kaggle_qwen3_vl_reranker_2b.ipynb](kaggle_qwen3_vl_reranker_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb) | Not tested yet |

## Updates not yet tested

These notebooks are newer than the tested ones. They have not been run yet, so no measurement applies to them. Use the tested notebook unless you want to try the update; once an update is tested it replaces the tested notebook.

| Update (not tested) | Replaces |
|---|---|
| ⚠ [colab_qwen3_vl_reranker_2b_update.ipynb](colab_qwen3_vl_reranker_2b_update.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b_update.ipynb) | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) |

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
| Image and video processor settings | `mm_processor_kwargs` in `/rerank`, `/score` (needs `--trust-request-mm-kwargs`: set in the Colab `_update` and Kaggle notebooks, not in the tested Colab notebook yet) | ✓ |
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

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 68 160 tokens (reduced from the config's 262 144 to fit; Qwen states 32K) |
| KV cache | 7.28 GiB = 68 176 tokens: one full-length request at a time |
| GPU memory | weights 4.08 GiB (4.19 GiB with the rest of the non-torch memory), peak activations 1.63 GiB, CUDA graphs 0.29 + 0.06 GiB |
| Time | server start ~2 min 41 s (weights 3 s, torch.compile 42 s, engine init 109 s) |
| Speed | ~17 documents/second with 200-token text documents (~4 700 tokens/second); 100 short documents in 3.2 s; 8 requests of 16 documents in 0.9 s; 8 requests of 4 images in 1.0 s; a small video cost 1 206 tokens; two text documents totalling ~22 000 tokens took 31.7 s |
| Checked through the public URL | 20/20 checks: key and blocked paths as for the text rerankers; text ranking, Vietnamese, `top_n`, `/score` equal to `/rerank`, `instruction` reaching the prompt, `max_tokens_per_doc`, 100 documents, parallel requests; images (a base64 image document ranked first, an image query against text documents, a document of text and image together, a screenshot of text), a base64 video document ranked first (0.86 against 0.38). An `https://` link is rejected with HTTP 400 and a `file://` link with HTTP 500, nothing is fetched. This notebook version has no `--trust-request-mm-kwargs`, so vLLM refuses `mm_processor_kwargs` in a request (HTTP 400): the update of this notebook enables it. The log shows the chat template was loaded and `float16` worked with the vision encoder; the video processor warns that some videos cost more tokens than in Qwen's reference code |

### Kaggle

Not tested yet.

# Retrieval models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Models that score or embed content for search: reranking a list of documents against a query, and turning content into vectors. Text in (some also take images and video), scores or vectors out; they do not generate text. The server never downloads media itself: send images and video as base64 `data:` URLs.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](qwen3_reranker_0_6b/README.md) | text (a query and the documents to rank) | 1.1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 2026-11-13 | [kaggle_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Not tested yet |
| [Qwen3-Reranker-4B](qwen3_reranker_4b/README.md) | text (a query and the documents to rank) | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb)<br>✓ T4, 2026-11-13 | [kaggle_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb)<br>Not tested yet |
| [Qwen3-Reranker-8B](qwen3_reranker_8b/README.md) | text (a query and the documents to rank) | 15.2 GiB | 40 960 | ~23 GB | [colab_qwen3_reranker_8b.ipynb](qwen3_reranker_8b/colab_qwen3_reranker_8b.ipynb)<br>Not tested yet | [kaggle_qwen3_reranker_8b.ipynb](qwen3_reranker_8b/kaggle_qwen3_reranker_8b.ipynb)<br>Not tested yet |
| [Qwen3-VL-Reranker-2B](qwen3_vl_reranker_2b/README.md) | text, images, video (queries and documents) | 4.0 GiB | 32 768 | ~10 GB | [colab_qwen3_vl_reranker_2b.ipynb](qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb)<br>✓ T4, 2026-11-13<br>⚠ [colab_qwen3_vl_reranker_2b_update.ipynb](qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b_update.ipynb) (update, not tested) | [kaggle_qwen3_vl_reranker_2b.ipynb](qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb)<br>Not tested yet |

⚠ marks an updated notebook that has not been tested yet. The tested notebook next to it stays the official one until the update is tested and replaces it.

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs when there are several; per-model flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen3-Reranker-0.6B | Qwen3-Reranker-4B | Qwen3-Reranker-8B | Qwen3-VL-Reranker-2B |
|---|---|---|---|---|---|
| T4 | Google Colab | ~10 GiB | ~3 GiB | ✗ not enough memory | ~6 GiB |
| L4 | Google Colab | ~17 GiB | ~10 GiB | ~2 GiB | ~13 GiB |
| A100 40 GB | Google Colab | ~33 GiB | ~26 GiB | ~18 GiB | ~29 GiB |
| A100 80 GB | Google Colab | ~69 GiB | ~62 GiB | ~54 GiB | ~65 GiB |
| H100 80 GB | Google Colab | ~68 GiB | ~61 GiB | ~53 GiB | ~64 GiB |
| T4 x2 | Kaggle | ~22 GiB | ~15 GiB | ~7 GiB | ~18 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM |

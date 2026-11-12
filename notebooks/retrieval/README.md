# Retrieval models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Models that score or embed text for search: reranking a list of documents against a query, and turning text into vectors. Text in, scores or vectors out; they do not generate text.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](qwen3_reranker_0_6b/README.md) | text (a query and the documents to rank) | 1.1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 2026-11-13 | [kaggle_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Not tested yet |
| [Qwen3-Reranker-4B](qwen3_reranker_4b/README.md) | text (a query and the documents to rank) | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb)<br>Not tested yet | [kaggle_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb)<br>Not tested yet |

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs (a small model such as the reranker uses one); per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen3-Reranker-0.6B | Qwen3-Reranker-4B |
|---|---|---|---|
| T4 | Google Colab | ~10 GiB | ~3 GiB |
| L4 | Google Colab | ~17 GiB | ~10 GiB |
| A100 40 GB | Google Colab | ~33 GiB | ~26 GiB |
| A100 80 GB | Google Colab | ~69 GiB | ~62 GiB |
| H100 80 GB | Google Colab | ~68 GiB | ~61 GiB |
| T4 x2 | Kaggle | ~22 GiB | ~15 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM |

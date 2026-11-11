# Text models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Language models that read and write text, with optional thinking and function calling.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B](qwen3_0_6b/README.md) | text | 1.4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](qwen3_0_6b/colab_qwen3_0_6b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_0_6b.ipynb](qwen3_0_6b/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-1.7B](qwen3_1_7b/README.md) | text | 3.8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](qwen3_1_7b/colab_qwen3_1_7b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_1_7b.ipynb](qwen3_1_7b/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-4B](qwen3_4b/README.md) | text | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](qwen3_4b/colab_qwen3_4b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_4b.ipynb](qwen3_4b/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-8B](qwen3_8b/README.md) | text | 15.3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](qwen3_8b/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | [kaggle_qwen3_8b.ipynb](qwen3_8b/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-14B](qwen3_14b/README.md) | text | 27.5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](qwen3_14b/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | — (needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total) |

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|---|
| T4 | Google Colab | ~10 GiB | ~8 GiB | ~4 GiB | ✗ not enough memory | ✗ not enough memory |
| L4 | Google Colab | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ not enough memory |
| A100 40 GB | Google Colab | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB | ~6 GiB |
| A100 80 GB | Google Colab | ~69 GiB | ~66 GiB | ~63 GiB | ~55 GiB | ~42 GiB |
| H100 80 GB | Google Colab | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB |
| T4 x2 | Kaggle | ~22 GiB | ~19 GiB | ~16 GiB | ~8 GiB | — |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | — |

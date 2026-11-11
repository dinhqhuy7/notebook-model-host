# Notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to the [main README](../README.md).

Each notebook hosts one model on one platform. Notebooks are grouped by type of model, one folder per model with its notebooks, GPU table and test results. Pick a model, then a platform where your account has a GPU with enough memory; choose the GPU before Run all (at least the model's "Min. total GPU memory"). Speeds in the test results come from the GPU used for each run, often a modest one; a stronger GPU is faster.

## [Text models](text/README.md)

Language models that read and write text, with optional thinking and function calling.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B](text/qwen3_0_6b/README.md) | text | 1.4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](text/qwen3_0_6b/colab_qwen3_0_6b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_0_6b.ipynb](text/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-1.7B](text/qwen3_1_7b/README.md) | text | 3.8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](text/qwen3_1_7b/colab_qwen3_1_7b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_1_7b.ipynb](text/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-4B](text/qwen3_4b/README.md) | text | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](text/qwen3_4b/colab_qwen3_4b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_4b.ipynb](text/qwen3_4b/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-8B](text/qwen3_8b/README.md) | text | 15.3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](text/qwen3_8b/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | [kaggle_qwen3_8b.ipynb](text/qwen3_8b/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-14B](text/qwen3_14b/README.md) | text | 27.5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](text/qwen3_14b/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | — (needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total) |

## [Vision-language models](vision-language/README.md)

Multimodal models that also read images and video. The server never downloads media itself: send images and video as base64 `data:` URLs.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3.5-0.8B](vision-language/qwen3_5_0_8b/README.md) | text, images, video | 1.6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](vision-language/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_0_8b.ipynb](vision-language/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-2B](vision-language/qwen3_5_2b/README.md) | text, images, video | 4.2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](vision-language/qwen3_5_2b/colab_qwen3_5_2b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_2b.ipynb](vision-language/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-4B](vision-language/qwen3_5_4b/README.md) | text, images, video | 8.7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](vision-language/qwen3_5_4b/colab_qwen3_5_4b.ipynb)<br>✓ L4, 2026-11-11 | [kaggle_qwen3_5_4b.ipynb](vision-language/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-9B](vision-language/qwen3_5_9b/README.md) | text, images, video | 18.0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](vision-language/qwen3_5_9b/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 | [kaggle_qwen3_5_9b.ipynb](vision-language/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Instruct](vision-language/qwen3_vl_2b_instruct/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_instruct.ipynb](vision-language/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 & A100 40 GB, 2026-11-11 | [kaggle_qwen3_vl_2b_instruct.ipynb](vision-language/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Thinking](vision-language/qwen3_vl_2b_thinking/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_thinking.ipynb](vision-language/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_vl_2b_thinking.ipynb](vision-language/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb)<br>✓ T4 x2, 2026-11-11 |

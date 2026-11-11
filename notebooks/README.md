# Notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to the [main README](../README.md).

Each notebook hosts one model on one platform. Pick a model, then a platform where your account has a GPU with enough memory. GPU tables and test results per platform: [Google Colab](colab/README.md), [Kaggle](kaggle/README.md). Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory".

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | text | 1.4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](colab/colab_qwen3_0_6b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_0_6b.ipynb](kaggle/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| `Qwen/Qwen3-1.7B` | text | 3.8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](colab/colab_qwen3_1_7b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_1_7b.ipynb](kaggle/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| `Qwen/Qwen3-4B` | text | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](colab/colab_qwen3_4b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_4b.ipynb](kaggle/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| `Qwen/Qwen3-8B` | text | 15.3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](colab/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | [kaggle_qwen3_8b.ipynb](kaggle/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| `Qwen/Qwen3-14B` | text | 27.5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](colab/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | — (needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total) |
| `Qwen/Qwen3.5-0.8B` | text, images, video | 1.6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](colab/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_0_8b.ipynb](kaggle/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| `Qwen/Qwen3.5-2B` | text, images, video | 4.2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](colab/colab_qwen3_5_2b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_2b.ipynb](kaggle/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| `Qwen/Qwen3.5-4B` | text, images, video | 8.7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](colab/colab_qwen3_5_4b.ipynb)<br>Not tested yet | [kaggle_qwen3_5_4b.ipynb](kaggle/kaggle_qwen3_5_4b.ipynb)<br>Not tested yet |
| `Qwen/Qwen3.5-9B` | text, images, video | 18.0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](colab/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 | [kaggle_qwen3_5_9b.ipynb](kaggle/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 2026-11-11 |

# Omni models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Models that hear and speak as well as read: text, images, audio and video in, text and speech out. They run on vLLM-Omni (a vLLM extension) in several stages, so they need more GPU memory than their size suggests. The server never downloads media itself: send images, audio and video as base64 `data:` URLs.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Omni-3B](qwen2_5_omni_3b/README.md) | text, images, audio, video (answers in text and speech) | 11.2 GiB | 32 768 | ~20 GB | [colab_qwen2_5_omni_3b.ipynb](qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb)<br>Not tested yet | [kaggle_qwen2_5_omni_3b.ipynb](qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb)<br>Not tested yet |

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen2.5-Omni-3B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~4 GiB |
| A100 40 GB | Google Colab | ~20 GiB |
| A100 80 GB | Google Colab | ~56 GiB |
| H100 80 GB | Google Colab | ~56 GiB |
| T4 x2 | Kaggle | ~9 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

# Vision-language models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Multimodal models that also read images and video. The server never downloads media itself: send images and video as base64 `data:` URLs.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3.5-0.8B](qwen3_5_0_8b/README.md) | text, images, video | 1.6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_0_8b.ipynb](qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-2B](qwen3_5_2b/README.md) | text, images, video | 4.2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](qwen3_5_2b/colab_qwen3_5_2b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_2b.ipynb](qwen3_5_2b/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-4B](qwen3_5_4b/README.md) | text, images, video | 8.7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](qwen3_5_4b/colab_qwen3_5_4b.ipynb)<br>✓ L4, 2026-11-11 | [kaggle_qwen3_5_4b.ipynb](qwen3_5_4b/kaggle_qwen3_5_4b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-9B](qwen3_5_9b/README.md) | text, images, video | 18.0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](qwen3_5_9b/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 | [kaggle_qwen3_5_9b.ipynb](qwen3_5_9b/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Instruct](qwen3_vl_2b_instruct/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_instruct.ipynb](qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 & A100 40 GB, 2026-11-11 | [kaggle_qwen3_vl_2b_instruct.ipynb](qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Thinking](qwen3_vl_2b_thinking/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_thinking.ipynb](qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_vl_2b_thinking.ipynb](qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen2.5-VL-7B-Instruct](qwen2_5_vl_7b_instruct/README.md) | text, images, video | 15.4 GiB | 128 000 | ~23 GB | [colab_qwen2_5_vl_7b_instruct.ipynb](qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb)<br>Not tested yet | [kaggle_qwen2_5_vl_7b_instruct.ipynb](qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb)<br>Not tested yet |

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct |
|---|---|---|---|---|---|---|---|---|
| T4 | Google Colab | ~7 GiB | ~4 GiB | ✗ not enough memory | ✗ not enough memory | ~7 GiB | ~7 GiB | ✗ not enough memory |
| L4 | Google Colab | ~14 GiB | ~11 GiB | ~7 GiB | ✗ not enough memory | ~13 GiB | ~13 GiB | ~2 GiB |
| A100 40 GB | Google Colab | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB | ~29 GiB | ~29 GiB | ~18 GiB |
| A100 80 GB | Google Colab | ~65 GiB | ~63 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB | ~54 GiB |
| H100 80 GB | Google Colab | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB | ~53 GiB |
| T4 x2 | Kaggle | ~18 GiB | ~16 GiB | ~11 GiB | ~2 GiB | ~18 GiB | ~18 GiB | ~7 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM |

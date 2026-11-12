# Qwen2.5-Omni-3B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Multimodal models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) · text, images, audio, video (answers in text and speech) · weights 11.2 GiB (download ~12 GB) · context 32 768 tokens · needs at least ~20 GB of total GPU memory

Hears and speaks: text, images, audio and video in; text and speech out. It runs on vLLM-Omni 0.32.0 with vLLM 0.32.0 (the other notebooks use vLLM 0.33.0), in three stages that are all loaded: 11.15 GiB of weights, so **a single T4 is not enough** (one L4 or A100, or Kaggle's 2 × T4). Ask for speech with `"modalities": ["text", "audio"]` (a WAV in base64, in the second choice) and a voice with `"speaker"` (`Chelsie` or `Ethan`); an answer with audio cannot stream, so keep it short (the tunnel cuts requests after 100 seconds). Each stage takes one request at a time by default. **No function calling and no thinking mode.** License: Qwen Research License, not Apache 2.0. `float16` on T4 (Kaggle) has not been tested with this model.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen2_5_omni_3b.ipynb](colab_qwen2_5_omni_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/multimodal/qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen2_5_omni_3b.ipynb](kaggle_qwen2_5_omni_3b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/multimodal/qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb) | Not tested yet |

## Estimated memory left for context

| GPU | Platform | Qwen2.5-Omni-3B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~4 GiB |
| A100 40 GB | Google Colab | ~20 GiB |
| A100 80 GB | Google Colab | ~56 GiB |
| H100 80 GB | Google Colab | ~56 GiB |
| T4 x2 | Kaggle | ~9 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

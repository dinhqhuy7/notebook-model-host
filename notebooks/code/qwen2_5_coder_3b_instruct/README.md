# Qwen2.5-Coder-3B-Instruct: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Code models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) · text · weights 5.8 GiB (download ~7 GB) · context 32 768 tokens · needs at least ~11 GB of total GPU memory

Code model (Instruct): chat, tool calling and code writing, fixing and explaining. The KV cache costs 36 KiB per token, so the full 32 768-token context fits on every listed GPU. It never thinks. If you send no system message its chat template adds "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; send your own to replace it. License: Qwen Research License, not Apache 2.0. `float16` on T4 has not been tested with this model yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen2_5_coder_3b_instruct.ipynb](colab_qwen2_5_coder_3b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen2_5_coder_3b_instruct.ipynb](kaggle_qwen2_5_coder_3b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb) | Not tested yet |

## Estimated memory left for context

| GPU | Platform | Qwen2.5-Coder-3B-Instruct |
|---|---|---|
| T4 | Google Colab | ~6 GiB |
| L4 | Google Colab | ~12 GiB |
| A100 40 GB | Google Colab | ~28 GiB |
| A100 80 GB | Google Colab | ~64 GiB |
| H100 80 GB | Google Colab | ~64 GiB |
| T4 x2 | Kaggle | ~17 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

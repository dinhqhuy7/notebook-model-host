# Qwen2.5-Coder-3B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Code models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) · text · weights 5.8 GiB (download ~7 GB) · context 32 768 tokens · needs at least ~11 GB of total GPU memory

Code model (base): code completion and fill-in-the-middle through `/v1/completions`, with the prompt `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`. It is not tuned for chat or tools (use the Instruct model for those). The KV cache costs 36 KiB per token, so the full 32 768-token context fits on every listed GPU. Always send `temperature`: without it vLLM samples at 1.0. License: Qwen Research License, not Apache 2.0. `float16` on T4 has not been tested with this model yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen2_5_coder_3b.ipynb](colab_qwen2_5_coder_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen2_5_coder_3b.ipynb](kaggle_qwen2_5_coder_3b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb) | Not tested yet |

## Estimated memory left for context

| GPU | Platform | Qwen2.5-Coder-3B |
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

# Qwen2.5-VL-7B-Instruct: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision-language models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · text, images, video · weights 15.4 GiB (download ~17 GB) · context 128 000 tokens · needs at least ~23 GB of total GPU memory

Multimodal (text, images, video), plain attention: the KV cache costs 56 KiB per token, so the full 128 000-token context needs ~7 GiB of KV cache. **It never thinks and has no function calling**: its chat template has no tool support, so the server enables no tool parser (use a JSON schema for structured output). Without sampling parameters it is almost greedy (`generation_config.json`). Grounding boxes are expected in absolute pixels (per Qwen's docs; not tested here yet). On T4 it runs in `float16` instead of its native `bfloat16`; that has not been tested with this model yet.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen2_5_vl_7b_instruct.ipynb](colab_qwen2_5_vl_7b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb) | Not tested yet |
| Kaggle | [kaggle_qwen2_5_vl_7b_instruct.ipynb](kaggle_qwen2_5_vl_7b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb) | Not tested yet |

## Estimated memory left for context

| GPU | Platform | Qwen2.5-VL-7B-Instruct |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~2 GiB |
| A100 40 GB | Google Colab | ~18 GiB |
| A100 80 GB | Google Colab | ~54 GiB |
| H100 80 GB | Google Colab | ~53 GiB |
| T4 x2 | Kaggle | ~7 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

Not tested yet.

### Kaggle

Not tested yet.

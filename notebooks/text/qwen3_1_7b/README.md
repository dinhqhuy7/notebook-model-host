# Qwen3-1.7B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Text models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · text · weights 3.8 GiB (download ~4 GB) · context 40 960 tokens · needs at least ~9 GB of total GPU memory

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_1_7b.ipynb](colab_qwen3_1_7b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/colab_qwen3_1_7b.ipynb) | ✓ T4, 2026-11-10 |
| Kaggle | [kaggle_qwen3_1_7b.ipynb](kaggle_qwen3_1_7b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb) | ✓ T4 x2, 2026-11-10 |

## Estimated memory left for context

| GPU | Platform | Qwen3-1.7B |
|---|---|---|
| T4 | Google Colab | ~8 GiB |
| L4 | Google Colab | ~14 GiB |
| A100 40 GB | Google Colab | ~30 GiB |
| A100 80 GB | Google Colab | ~66 GiB |
| H100 80 GB | Google Colab | ~66 GiB |
| T4 x2 | Kaggle | ~19 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 8.72 GiB = 81 632 tokens, 1.99 full-length requests at once |
| GPU memory | weights 3.22 GiB, peak activations 0.92 GiB, CUDA graphs 0.17 GiB |
| Time | server start 2 min 52 s (weights 26 s, torch.compile 29 s, CUDA graphs 20 s) |
| Speed | ~50 tokens/second for one request; ~330 tokens/second total with 8 at once |
| Checked through the public URL | no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort="none"`); Vietnamese text in float16 reads normally; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. Thinking on a simple question: 1 231 tokens in 22 s. Quality is limited: one answer mixed in a few Thai characters |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 21.4 GiB (10.72 GiB per GPU) = 200 752 tokens, 4.90 full-length requests at once |
| GPU memory (per GPU) | weights 1.61 GiB, peak activations 0.56 GiB, CUDA graphs 0.18 GiB |
| Time | engine start 76 s (torch.compile 27 s, CUDA graphs 19 s) |
| Speed | ~73 tokens/second for one request; ~506 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. An 18 000-token prompt took 29.6 s, about half the time of a single T4 |

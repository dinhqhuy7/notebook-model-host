# Qwen3-0.6B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Text models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · text · weights 1.4 GiB (download ~2 GB) · context 40 960 tokens · needs at least ~6 GB of total GPU memory

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_0_6b.ipynb](colab_qwen3_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/colab_qwen3_0_6b.ipynb) | ✓ T4, 2026-11-10 |
| Kaggle | [kaggle_qwen3_0_6b.ipynb](kaggle_qwen3_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb) | ✓ T4 x2, 2026-11-10 |

## Estimated memory left for context

| GPU | Platform | Qwen3-0.6B |
|---|---|---|
| T4 | Google Colab | ~10 GiB |
| L4 | Google Colab | ~17 GiB |
| A100 40 GB | Google Colab | ~33 GiB |
| A100 80 GB | Google Colab | ~69 GiB |
| H100 80 GB | Google Colab | ~68 GiB |
| T4 x2 | Kaggle | ~22 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 11.14 GiB = 104 256 tokens, 2.55 full-length requests at once |
| GPU memory | weights 1.12 GiB, peak activations 0.59 GiB, CUDA graphs 0.17 GiB |
| Time | server start 2 min 25 s (weights 11 s, torch.compile 28 s, CUDA graphs 18 s) |
| Speed | ~77 tokens/second for one request; ~570 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. Long prompts are slow on T4 whatever the model size: ~17 000–18 000 tokens take ~50–55 s (also with Qwen3-1.7B), so prompts over ~30 000 tokens need streaming |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 23.5 GiB (11.75 GiB per GPU) = 220 080 tokens, 5.37 full-length requests at once |
| GPU memory (per GPU) | weights 0.57 GiB, peak activations 0.57 GiB, CUDA graphs 0.18 GiB |
| Time | engine start 80 s (torch.compile 30 s, CUDA graphs 19 s) |
| Speed | ~104 tokens/second for one request; ~834 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. An 18 000-token prompt took 28.5 s, half the time of a single T4 |

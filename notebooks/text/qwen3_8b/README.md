# Qwen3-8B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Text models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · text · weights 15.3 GiB (download ~16 GB) · context 40 960 tokens · needs at least ~21 GB of total GPU memory

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_8b.ipynb](colab_qwen3_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/colab_qwen3_8b.ipynb) | ✓ A100 40 GB, 2026-11-10 |
| Kaggle | [kaggle_qwen3_8b.ipynb](kaggle_qwen3_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/kaggle_qwen3_8b.ipynb) | ✓ T4 x2, 2026-11-10 |

## Estimated memory left for context

| GPU | Platform | Qwen3-8B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~3 GiB |
| A100 40 GB | Google Colab | ~19 GiB |
| A100 80 GB | Google Colab | ~55 GiB |
| H100 80 GB | Google Colab | ~54 GiB |
| T4 x2 | Kaggle | ~8 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 18.47 GiB = 134 448 tokens, 3.28 full-length requests at once |
| GPU memory | weights 15.27 GiB, peak activations 1.64 GiB, CUDA graphs 0.23 GiB |
| Speed | ~75 tokens/second for one request |
| Time | install 2 min, download 43 s, server start 3 min 35 s (FlashInfer JIT ~1 min 45 s, torch.compile 31 s); ~7 min from Run all to a public URL |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off (`enable_thinking`, `reasoning_effort="none"`, `/no_think`); tool calling; JSON schema; completions; `/tokenize`; Anthropic `/v1/messages`; `cache_salt`; streaming for 124 s / 9 000 tokens without being cut |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 9.2 GiB (4.58 GiB per GPU) = 66 704 tokens, 1.63 full-length requests at once |
| GPU memory (per GPU) | weights 7.64 GiB, CUDA graphs 0.42 + 0.23 GiB |
| Time | loading weights 52 s, engine start 90 s (torch.compile 33 s, CUDA graphs 26 s) |
| Speed | one request ~8–26 tokens/second, uneven between runs; ~170 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. Thinking took 89 s for 1 172 tokens and an 18 000-token prompt 76 s, close to the 100-second limit: use streaming |

# Qwen3-4B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Language models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · text · weights 7.5 GiB (download ~8 GB) · context 40 960 tokens · needs at least ~13 GB of total GPU memory

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_4b.ipynb](colab_qwen3_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_4b/colab_qwen3_4b.ipynb) | ✓ T4, 2026-11-10 |
| Kaggle | [kaggle_qwen3_4b.ipynb](kaggle_qwen3_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_4b/kaggle_qwen3_4b.ipynb) | ✓ T4 x2, 2026-11-10 |

## Estimated memory left for context

| GPU | Platform | Qwen3-4B |
|---|---|---|
| T4 | Google Colab | ~4 GiB |
| L4 | Google Colab | ~11 GiB |
| A100 40 GB | Google Colab | ~27 GiB |
| A100 80 GB | Google Colab | ~63 GiB |
| H100 80 GB | Google Colab | ~62 GiB |
| T4 x2 | Kaggle | ~16 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1 |
| Context (`auto`) | 30 592 tokens (reduced from 40 960 to fit) |
| KV cache | 4.2 GiB = 30 608 tokens: one full-length request, or many short ones |
| GPU memory | weights 7.56 GiB, peak activations 1.15 GiB, CUDA graphs 0.22 GiB |
| Speed | ~25 tokens/second for one request; ~170 tokens/second total with 8 at once |
| Time | server start 3 min 30 s (weights 42 s, torch.compile 37 s, CUDA graphs 30 s) |
| Expected log noise | `ERROR ... Cannot use FA version 2` (T4 falls back to Triton attention), FlashInfer sampler unavailable, and many `bfloat16 into float16` weight warnings |
| Checked through the public URL | no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort="none"`); Vietnamese text in float16 reads normally; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. Thinking is slow on T4 (a simple question used 2 000 tokens, 81 s): disable it or stream |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 16.8 GiB (8.41 GiB per GPU) = 122 416 tokens, 2.99 full-length requests at once |
| GPU memory (per GPU) | weights 3.82 GiB, peak activations 0.67 GiB, CUDA graphs 0.2 GiB |
| Time | engine start 85 s (weights 10 s, torch.compile 32 s, CUDA graphs 21 s) |
| Speed | ~42 tokens/second for one request; ~290 tokens/second total with 8 at once (about 1.7× a single Colab T4) |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. A 27 000-token prompt got error 524 without streaming and was answered correctly with streaming (first token after 15 s): stream long prompts |

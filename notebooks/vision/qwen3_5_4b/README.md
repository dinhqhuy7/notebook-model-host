# Qwen3.5-4B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · text, images, video · weights 8.7 GiB (download ~9 GB) · context 262 144 tokens · needs at least ~17 GB of total GPU memory

Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_5_4b.ipynb](colab_qwen3_5_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_4b/colab_qwen3_5_4b.ipynb) | ✓ L4, 2026-11-11 |
| Kaggle | [kaggle_qwen3_5_4b.ipynb](kaggle_qwen3_5_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb) | ✓ T4 x2, 2026-11-11 |

## Functions

Everything the model can do, and whether the notebook hosts it. Whether it was run is in the test results below.

| Function | How to call it | Hosted |
|---|---|---|
| Image input | `image_url` with a base64 `data:` URL in `/v1/chat/completions` | ✓ |
| Video input | `video_url` with a base64 `data:` URL in `/v1/chat/completions` | ✓ |
| Image and video by http(s) link | `image_url` / `video_url` `https://...` / `/v1/chat/completions` | ✗ blocked on purpose: the server never downloads links; send base64 |
| Thinking mode, on or off per request | `chat_template_kwargs.enable_thinking` in `/v1/chat/completions`; the thinking comes back in `reasoning` | ✓ |
| Function calling | `tools` in `/v1/chat/completions`; calls come back in `tool_calls` | ✓ |
| Structured output (JSON schema) | `response_format` in `/v1/chat/completions` | ✓ |

## Estimated memory left for context

| GPU | Platform | Qwen3.5-4B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ~7 GiB |
| A100 40 GB | Google Colab | ~22 GiB |
| A100 80 GB | Google Colab | ~58 GiB |
| H100 80 GB | Google Colab | ~58 GiB |
| T4 x2 | Kaggle | ~11 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × NVIDIA L4 (22.5 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / bfloat16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 9.19 GiB = 297 061 tokens, 1.13 full-length requests at once |
| GPU memory | weights + non-torch 8.73 GiB, peak activation 1.91 GiB, CUDA graphs 0.27 + 0.14 GiB |
| Time | server start 5 min 53 s (weights 7 s, torch.compile 32 s, initial profiling/warmup 94 s, FlashInfer JIT ~1 min 50 s, CUDA graphs 24 s) |
| Speed | ~29 tokens/second for one request (with thinking); ~194 tokens/second total with 8 at once |
| Checked through the public URL | 13/13 checks. no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on by default (911 tokens, 32 s); thinking off via both switches; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; 8 concurrent requests; `http(s)` image URL rejected; secret found in a 16 000-token prompt (3.7 s). Base64 image: text and both colours right, the rectangle called a square |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 11.9 GiB (5.96 GiB per GPU) = 385 659 tokens, 1.47 full-length requests at once |
| GPU memory (per GPU) | weights 4.38 GiB, CUDA graphs 0.58 + 0.22 GiB |
| Time | server start ~11 min (weights 25 s, torch.compile 31 s, ~7 min of unlogged kernel compilation, CUDA graphs 55 s) |
| Speed | ~44 tokens/second for one request (with thinking); ~258 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on by default (791 tokens, 18 s); tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (~20 s). Base64 image: text and both colours right, the rectangle called a square |

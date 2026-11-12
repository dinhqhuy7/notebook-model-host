# Qwen3.5-0.8B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · text, images, video · weights 1.6 GiB (download ~2 GB) · context 262 144 tokens · needs at least ~10 GB of total GPU memory

Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_5_0_8b.ipynb](colab_qwen3_5_0_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb) | ✓ T4, 2026-11-11 |
| Kaggle | [kaggle_qwen3_5_0_8b.ipynb](kaggle_qwen3_5_0_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb) | ✓ T4 x2, 2026-11-11 |

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

| GPU | Platform | Qwen3.5-0.8B |
|---|---|---|
| T4 | Google Colab | ~7 GiB |
| L4 | Google Colab | ~14 GiB |
| A100 40 GB | Google Colab | ~29 GiB |
| A100 80 GB | Google Colab | ~65 GiB |
| H100 80 GB | Google Colab | ~65 GiB |
| T4 x2 | Kaggle | ~18 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

| Item | Result |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 9.11 GiB = 785 357 tokens, 3.00 full-length requests at once |
| GPU memory | weights 1.63 GiB, other non-torch ~1 GiB, CUDA graphs 0.20 + 0.12 GiB |
| Time | server start ~11 min (weights 15 s, torch.compile 27 s, ~7.5 min of unlogged kernel compilation, CUDA graphs 47 s) |
| Speed | ~100 tokens/second for one request (with thinking); ~640 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; no thinking by default, thinking with `enable_thinking: true`; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (13.5 s). Base64 image: text and red rectangle right, but the blue circle was called green |
| Sampling parameters | a short factual question was right 9/20 times when the request sent no sampling parameters (vLLM defaults) and 20/20 with Qwen's recommended values: always send them |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 20.4 GiB (10.19 GiB per GPU) = 1 758 728 tokens, 6.71 full-length requests at once |
| GPU memory (per GPU) | weights 0.84 GiB, other non-torch ~0.9 GiB, CUDA graphs 0.21 + 0.12 GiB |
| Time | server start ~11.5 min (weights 7 s, torch.compile 30 s, ~7.7 min of unlogged kernel compilation, CUDA graphs 33 s) |
| Speed | ~120 tokens/second for one request (with thinking); ~289 tokens/second total with 8 at once, slower than a single Colab T4 (~640): for a model this small, splitting over 2 GPUs costs more than it saves |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking with `enable_thinking: true`; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (7.9 s, faster than a single T4). Base64 image: text and red rectangle right, the blue circle called green (same as on Colab) |

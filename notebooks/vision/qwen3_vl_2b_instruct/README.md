# Qwen3-VL-2B-Instruct: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · text, images, video · weights 4.0 GiB (download ~5 GB) · context 262 144 tokens · needs at least ~10 GB of total GPU memory

Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full 262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs vLLM lowers the context to what fits (see the memory table). **It never thinks** (no reasoning).

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_vl_2b_instruct.ipynb](colab_qwen3_vl_2b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb) | ✓ T4 & A100 40 GB, 2026-11-11 |
| Kaggle | [kaggle_qwen3_vl_2b_instruct.ipynb](kaggle_qwen3_vl_2b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb) | ✓ T4 x2, 2026-11-11 |

## Estimated memory left for context

| GPU | Platform | Qwen3-VL-2B-Instruct |
|---|---|---|
| T4 | Google Colab | ~7 GiB |
| L4 | Google Colab | ~13 GiB |
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
| GPU | 1 × Tesla T4 (15 GB) and 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 (T4), bfloat16 (A100) / 1 |
| Context (`auto`) | T4: 61 808 tokens; A100: 262 144 tokens (model maximum) |
| KV cache | T4: 6.6 GiB = 61 823 tokens, 1.00×; A100: 29.03 GiB = 271 808 tokens, 1.04× |
| GPU memory | T4: weights + non-torch 4.38 GiB, peak activation 2.12 GiB, CUDA graphs 0.30 + 0.18 GiB; A100: 4.37 GiB, 2.15 GiB, 0.30 + 0.20 GiB |
| Time | T4: server start 4 min 5 s; A100: 3 min 47 s (weights 1.3 s, multimodal warmup 14.5 s, torch.compile 40.9 s, first FlashInfer JIT ~1 min 46 s, CUDA graphs ~11 s) |
| Speed | T4: ~44 tokens/s single, ~318–335 tokens/s with 8; A100: ~216 single, ~1 165–1 365 with 8 |
| Checked through the public URL | 26/26 on both GPUs: no reasoning; tool calls via `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; 8 concurrent requests; image base64 and multiple images; document/table OCR to JSON; 2D grounding; counting and spatial reasoning; chart reasoning; visual coding; multi-turn with an image; tool call from image content; Responses API with an image; video OCR/order/approximate timestamp and streaming; image + video; `http(s)` media URLs rejected; secret found in an 18 044-token prompt (T4: 56.6 s uncached; A100: 0.9 s on the first test). On A100, the final run located the 2-second video boundary exactly and mixed image + video passed twice; on T4, these two quality checks varied across runs |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 170 256 tokens (reduced from the model maximum 262 144) |
| KV cache | 18.2 GiB total (9.09 GiB per GPU) = 170 272 tokens, 1.00 full-length request at once |
| GPU memory (per GPU) | weights + non-torch 2.29 GiB, peak activation 1.72 GiB, CUDA graphs 0.30 + 0.18 GiB |
| Time | server start 3 min 39 s (weights 8 s, multimodal warmup 16 s, torch.compile 45 s, CUDA graphs ~23 s) |
| Speed | ~71 tokens/second for one request; ~498 tokens/second total with 8 at once |
| Checked through the public URL | 12/12 standard checks: no reasoning; tool calls via `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; image base64; `http(s)` image URL rejected; secret found in an 18 000-token prompt (29.5 s). Extended checks passed: multiple images, document OCR to JSON, 2D grounding, spatial reasoning, visual coding, multi-turn with an image, tool call from image content, Responses API with text and images, video OCR/order/timestamp, image + video, and video streaming; `http(s)` video URL rejected. A short video question confused the last colour, while the full three-segment prompt was correct |

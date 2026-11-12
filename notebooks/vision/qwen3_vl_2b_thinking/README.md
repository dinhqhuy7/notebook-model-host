# Qwen3-VL-2B-Thinking: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) · text, images, video · weights 4.0 GiB (download ~5 GB) · context 262 144 tokens · needs at least ~10 GB of total GPU memory

Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full 262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs vLLM lowers the context to what fits (see the memory table). **It always thinks**: the chat template has no switch, so `enable_thinking` changes nothing; give `max_tokens` room for the thinking.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_vl_2b_thinking.ipynb](colab_qwen3_vl_2b_thinking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb) | ✓ T4, 2026-11-11 |
| Kaggle | [kaggle_qwen3_vl_2b_thinking.ipynb](kaggle_qwen3_vl_2b_thinking.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb) | ✓ T4 x2, 2026-11-11 |

## Estimated memory left for context

| GPU | Platform | Qwen3-VL-2B-Thinking |
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
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1, reasoning parser `qwen3` |
| Context (`auto`) | 61 808 tokens (reduced from the model maximum 262 144) |
| KV cache | 6.6 GiB = 61 823 tokens, 1.00 full-length request at once |
| GPU memory | weights + non-torch 4.38 GiB, peak activation 2.12 GiB, CUDA graphs 0.30 + 0.18 GiB |
| Time | server start 4 min 2 s (weights 17 s, multimodal warmup 28 s, torch.compile 45 s, CUDA graphs ~23 s) |
| Speed | ~42 tokens/second for one request; ~185–357 tokens/second total with 8 at once |
| Checked through the public URL | 24/26 checks passed. Reasoning is always on and is separated into `message.reasoning`; tool calls via `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; image base64 and multiple images; document/table OCR to JSON; counting and spatial reasoning; chart reasoning; visual coding; tool call from image content; Responses API with an image; video OCR/order/timestamp and streaming; image + video; `http(s)` media URLs rejected; secret found in an 18 046-token prompt (17–75 s, depending on reasoning length). Limitations: 2D grounding returned pixel coordinates or exhausted the reasoning budget before a box; image multi-turn was correct in 2/4 runs and otherwise confused the visible code with HTML or repeated until the token limit. It also sometimes copied a requested answer placeholder literally. Deterministic sampling is used for capability checks |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel), reasoning parser `qwen3` |
| Context (`auto`) | 170 256 tokens (reduced from the model maximum 262 144) |
| KV cache | 18.2 GiB total (9.09 GiB per GPU) = 170 272 tokens, 1.00 full-length request at once |
| GPU memory (per GPU) | weights + non-torch 2.29 GiB, peak activation 1.72 GiB, CUDA graphs 0.30 + 0.18 GiB |
| Time | server start ~3 min 35 s, same in two runs (weights 7 s, multimodal warmup 15 s, torch.compile 41 s, CUDA graphs ~21 s) |
| Speed | ~71–74 tokens/second for one request; ~452 tokens/second total with 8 at once |
| Checked through the public URL | 24/26 checks passed. Reasoning is always on and is separated into `message.reasoning`; tool calls via `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; image base64 and multiple images; document/table OCR to JSON; counting and spatial reasoning; chart reasoning; visual coding; tool call from image content; Responses API with an image; video OCR/order/timestamp and streaming; image + video; `http(s)` media URLs rejected; secret found in an 18 046-token prompt (38.9 s). Limitations: 2D grounding returned wrong coordinates (`0,0,200,200` instead of about `100,200,350,533`); image multi-turn gave the wrong answer; it also sometimes copied a requested answer placeholder literally. Deterministic sampling is used for capability checks |

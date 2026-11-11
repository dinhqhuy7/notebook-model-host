# Qwen3.5-2B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision-language models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) · text, images, video · weights 4.2 GiB (download ~5 GB) · context 262 144 tokens · needs at least ~12 GB of total GPU memory

Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_5_2b.ipynb](colab_qwen3_5_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/colab_qwen3_5_2b.ipynb) | ✓ T4, 2026-11-11 |
| Kaggle | [kaggle_qwen3_5_2b.ipynb](kaggle_qwen3_5_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb) | ✓ T4 x2, 2026-11-11 |

## Estimated memory left for context

| GPU | Platform | Qwen3.5-2B |
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
| vLLM / dtype / GPUs | 0.33.0 / float16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 6.15 GiB = 530 734 tokens, 2.02 full-length requests at once (same in both runs) |
| GPU memory | weights 4.18 GiB, CUDA graphs 0.20 + 0.12 GiB |
| Time | server start 12 min 53 s and 11 min 50 s (two runs 8.5 hours apart), of which ~8 min is unlogged kernel compilation after torch.compile |
| Speed | ~300–332 tokens/second total with 8 requests at once |
| Checked through the public URL | 12/13 checks, twice. no/wrong key → 401; `/invocations`, `/sleep` → 404; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (~16 s). The thinking check failed only because this model sometimes thinks past the check's 4 000-token budget: through streaming with 12 000 tokens it answered correctly 3 times out of 3 (1 300–2 100 tokens). Base64 image: text right, the rectangle called a square |

### Kaggle

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 17.3 GiB (8.64 GiB per GPU) = 1 490 675 tokens, 5.69 full-length requests at once |
| GPU memory (per GPU) | weights 2.12 GiB, other non-torch ~0.9 GiB, CUDA graphs 0.22 + 0.12 GiB |
| Time | server start ~10 min (weights 7 s, torch.compile 25 s, ~6.6 min of unlogged kernel compilation, CUDA graphs 32 s) |
| Speed | ~86 tokens/second for one request (with thinking); ~272 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking with `enable_thinking: true`; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; `http(s)` image URL rejected; secret found in a 16 000-token prompt (9.6 s). Base64 image: text right, the rectangle called a square |
| Sampling parameters | a short factual question, 20 tries each: temperature 0 right 20/20; Qwen's general values (0.7/0.8, top_k 20, presence 1.5) 17/20; Qwen's text values for 0.8B/2B (1.0/1.0, top_k 20, presence 2.0) 9/20. Use a low temperature for short exact answers |

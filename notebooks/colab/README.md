# Host open models on Google Colab: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to the [main README](../../README.md).

## Notebook

| Notebook | Model | Weights | Download | Min. total GPU memory | Test status |
|---|---|---|---|---|---|
| [colab_qwen3_0_6b.ipynb](colab_qwen3_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_0_6b.ipynb) | `Qwen/Qwen3-0.6B` | 1.4 GiB | ~2 GB | ~6 GB | ✓ T4, 2026-11-10 |
| [colab_qwen3_1_7b.ipynb](colab_qwen3_1_7b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_1_7b.ipynb) | `Qwen/Qwen3-1.7B` | 3.8 GiB | ~4 GB | ~9 GB | ✓ T4, 2026-11-10 |
| [colab_qwen3_4b.ipynb](colab_qwen3_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_4b.ipynb) | `Qwen/Qwen3-4B` | 7.5 GiB | ~8 GB | ~13 GB | ✓ T4, 2026-11-10 |
| [colab_qwen3_8b.ipynb](colab_qwen3_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_8b.ipynb) | `Qwen/Qwen3-8B` | 15.3 GiB | ~16 GB | ~21 GB | ✓ A100 40 GB, 2026-11-10 |
| [colab_qwen3_14b.ipynb](colab_qwen3_14b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_14b.ipynb) | `Qwen/Qwen3-14B` | 27.5 GiB | ~30 GB | ~35 GB | ✓ A100 40 GB, 2026-11-10 |
| [colab_qwen3_5_0_8b.ipynb](colab_qwen3_5_0_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_0_8b.ipynb) | `Qwen/Qwen3.5-0.8B` | 1.6 GiB | ~2 GB | ~10 GB | ✓ T4, 2026-11-11 |
| [colab_qwen3_5_2b.ipynb](colab_qwen3_5_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_2b.ipynb) | `Qwen/Qwen3.5-2B` | 4.2 GiB | ~5 GB | ~12 GB | ✓ T4, 2026-11-11 |
| [colab_qwen3_5_4b.ipynb](colab_qwen3_5_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_4b.ipynb) | `Qwen/Qwen3.5-4B` | 8.7 GiB | ~9 GB | ~17 GB | ✓ L4, 2026-11-11 |
| [colab_qwen3_5_9b.ipynb](colab_qwen3_5_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18.0 GiB | ~20 GB | ~28 GB | ✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 |
| [colab_qwen3_vl_2b_instruct.ipynb](colab_qwen3_vl_2b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_vl_2b_instruct.ipynb) | `Qwen/Qwen3-VL-2B-Instruct` | 4.0 GiB | ~5 GB | ~10 GB | ✓ T4 & A100 40 GB, 2026-11-11 |
| [colab_qwen3_vl_2b_thinking.ipynb](colab_qwen3_vl_2b_thinking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_vl_2b_thinking.ipynb) | `Qwen/Qwen3-VL-2B-Thinking` | 4.0 GiB | ~5 GB | ~10 GB | ✓ T4, 2026-11-11 |

**Qwen3.5-0.8B:** Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-2B:** Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-4B:** Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-9B:** Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

**Qwen3-VL-2B-Instruct:** Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full 262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs vLLM lowers the context to what fits (see the memory table). **It never thinks** (no reasoning).

**Qwen3-VL-2B-Thinking:** Multimodal (text, images, video), plain attention: the KV cache costs 112 KiB per token, so the full 262 144-token context needs ~28 GiB of KV cache and fits only on A100-class GPUs; on smaller GPUs vLLM lowers the context to what fits (see the memory table). **It always thinks**: the chat template has no switch, so `enable_thinking` changes nothing; give `max_tokens` room for the thinking.

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T4 | ~10 GiB | ~8 GiB | ~4 GiB | ✗ not enough memory | ✗ not enough memory | ~7 GiB | ~4 GiB | ✗ not enough memory | ✗ not enough memory | ~7 GiB | ~7 GiB |
| L4 | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ not enough memory | ~14 GiB | ~11 GiB | ~7 GiB | ✗ not enough memory | ~13 GiB | ~13 GiB |
| A100 40 GB | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB | ~6 GiB | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB | ~29 GiB | ~29 GiB |
| A100 80 GB | ~69 GiB | ~66 GiB | ~63 GiB | ~55 GiB | ~42 GiB | ~65 GiB | ~63 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB |
| H100 80 GB | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### colab_qwen3_0_6b.ipynb

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

### colab_qwen3_1_7b.ipynb

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

### colab_qwen3_4b.ipynb

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

### colab_qwen3_8b.ipynb

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

### colab_qwen3_14b.ipynb

| Item | Result |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 38 304 tokens (reduced from 40 960 to fit) |
| KV cache | 5.85 GiB = 38 320 tokens: one full-length request at a time, more if shorter |
| GPU memory | weights 27.52 GiB, peak activations 2.0 GiB, CUDA graphs 0.28 GiB |
| Time | server start 3 min 47 s (weights 12 s, torch.compile 35 s, FlashInfer JIT ~1 min 50 s, CUDA graphs 16 s) |
| Speed | ~44 tokens/second for one request; ~287 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; a secret found in a 27 044-token prompt in 6 s. For the full 40 960 context and several long requests at once, use an 80 GB GPU |

### colab_qwen3_5_0_8b.ipynb

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

### colab_qwen3_5_2b.ipynb

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

### colab_qwen3_5_4b.ipynb

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

### colab_qwen3_5_9b.ipynb

| Item | Result |
|---|---|
| A100 80 GB run | 1 × NVIDIA A100-SXM4-80GB, `--max-num-seqs 128`: context 262 144, KV cache 49.25 GiB = 1 592 668 tokens (6.08 full-length requests at once), server start ~5 min 50 s, ~88 tokens/second for one request, ~505 total with 8 at once; 13/13 checks passed, a 24 000-token prompt in 3.2 s. The rows below are from the A100 40 GB run |
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / GPUs | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 262 144 tokens (the model maximum) |
| KV cache | 13.17 GiB = 425 788 tokens, 1.62 full-length requests at once |
| GPU memory | weights 17.57 GiB, other non-torch ~2.5 GiB (image encoder, linear-attention state), peak activations 2.27 GiB, CUDA graphs 0.27 GiB |
| Time | server start 6 min (weights 10 s, torch.compile 28 s, FlashInfer JIT ~1 min 50 s, ~1 min 40 s of unlogged warm-up) |
| Speed | ~73 tokens/second for one request; ~500 tokens/second total with 8 at once |
| Checked through the public URL | no key → 401; `/invocations` → 404; thinking on/off (`enable_thinking`, `reasoning_effort="none"`); tool calling via `qwen3_coder`; JSON schema; base64 image correctly described; `http(s)` image URL rejected (HTTP 400); Anthropic `/v1/messages`; streaming; a secret found in the middle of a 57 242-token prompt (6 s) |
| 11 Wikimedia images | all served (10 in parallel in 8.7 s; 3 images in one request, 2 125 tokens, 3.9 s). Scenes, food, landmarks and a handwritten blackboard formula read correctly; on a folded receipt the amounts were right but the shop name and date were not; a Vietnamese no-overtaking sign was misread |
| 13 text tasks (auto-graded) | correct: multi-step math and a logic puzzle (with thinking), summary, spelling fixes, JSON extraction, multi-turn memory, two-step tool calling. Wrong: an idiom translated literally with a typo, "longest river in Vietnam" answered as the Red River, a lục bát poem with the wrong 6-8 pattern. Code without thinking: 11/14 tests in 8 s; with thinking the model looped for 30 000 tokens (7 min) and never wrote the code, so disable thinking for coding |

### colab_qwen3_vl_2b_instruct.ipynb

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

### colab_qwen3_vl_2b_thinking.ipynb

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

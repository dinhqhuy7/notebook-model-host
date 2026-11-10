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
| [colab_qwen3_5_9b.ipynb](colab_qwen3_5_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18.0 GiB | ~20 GB | ~28 GB | ✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 |

**Qwen3.5-9B:** Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4, where it takes ~12 minutes to start.

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-9B |
|---|---|---|---|---|---|---|
| T4 | ~10 GiB | ~8 GiB | ~4 GiB | ✗ not enough memory | ✗ not enough memory | ✗ not enough memory |
| L4 | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ not enough memory | ✗ not enough memory |
| A100 40 GB | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB | ~6 GiB | ~13 GiB |
| A100 80 GB | ~69 GiB | ~66 GiB | ~63 GiB | ~55 GiB | ~42 GiB | ~49 GiB |
| H100 80 GB | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB | ~49 GiB |

## Test results

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

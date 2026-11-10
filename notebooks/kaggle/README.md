# Host open models on Kaggle: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to the [main README](../../README.md).

## Notebook

| Notebook | Model | Weights | Download | Min. total GPU memory | Test status |
|---|---|---|---|---|---|
| [kaggle_qwen3_0_6b.ipynb](kaggle_qwen3_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_0_6b.ipynb) | `Qwen/Qwen3-0.6B` | 1.4 GiB | ~2 GB | ~6 GB | ✓ T4 x2, 2026-11-10 |
| [kaggle_qwen3_1_7b.ipynb](kaggle_qwen3_1_7b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_1_7b.ipynb) | `Qwen/Qwen3-1.7B` | 3.8 GiB | ~4 GB | ~9 GB | ✓ T4 x2, 2026-11-10 |
| [kaggle_qwen3_4b.ipynb](kaggle_qwen3_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_4b.ipynb) | `Qwen/Qwen3-4B` | 7.5 GiB | ~8 GB | ~13 GB | ✓ T4 x2, 2026-11-10 |
| [kaggle_qwen3_8b.ipynb](kaggle_qwen3_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_8b.ipynb) | `Qwen/Qwen3-8B` | 15.3 GiB | ~16 GB | ~21 GB | ✓ T4 x2, 2026-11-10 |
| [kaggle_qwen3_5_0_8b.ipynb](kaggle_qwen3_5_0_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_0_8b.ipynb) | `Qwen/Qwen3.5-0.8B` | 1.6 GiB | ~2 GB | ~10 GB | ✓ T4 x2, 2026-11-11 |
| [kaggle_qwen3_5_2b.ipynb](kaggle_qwen3_5_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_2b.ipynb) | `Qwen/Qwen3.5-2B` | 4.2 GiB | ~5 GB | ~12 GB | Not tested yet |
| [kaggle_qwen3_5_4b.ipynb](kaggle_qwen3_5_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_4b.ipynb) | `Qwen/Qwen3.5-4B` | 8.7 GiB | ~9 GB | ~17 GB | Not tested yet |
| [kaggle_qwen3_5_9b.ipynb](kaggle_qwen3_5_9b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18.0 GiB | ~20 GB | ~28 GB | ✓ T4 x2, 2026-11-11 |

**Qwen3.5-0.8B:** Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-2B:** Multimodal (text, images, video), hybrid attention: the KV cache costs ~12 KiB per token, so the full 262 144-token context fits on every listed GPU. **Thinking is off by default**: send `chat_template_kwargs: {"enable_thinking": true}` to turn it on. On T4 the model runs in `float16` instead of its native `bfloat16`, and the first start takes ~11 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-4B:** Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

**Qwen3.5-9B:** Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

**Qwen3-14B:** no Kaggle notebook: needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total.

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|---|---|---|---|---|
| T4 x2 | ~22 GiB | ~19 GiB | ~16 GiB | ~8 GiB | ~18 GiB | ~16 GiB | ~11 GiB | ~2 GiB |
| P100 | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM |

## Test results

### kaggle_qwen3_0_6b.ipynb

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

### kaggle_qwen3_1_7b.ipynb

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 tokens (the model maximum) |
| KV cache | 21.4 GiB (10.72 GiB per GPU) = 200 752 tokens, 4.90 full-length requests at once |
| GPU memory (per GPU) | weights 1.61 GiB, peak activations 0.56 GiB, CUDA graphs 0.18 GiB |
| Time | engine start 76 s (torch.compile 27 s, CUDA graphs 19 s) |
| Speed | ~73 tokens/second for one request; ~506 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming. An 18 000-token prompt took 29.6 s, about half the time of a single T4 |

### kaggle_qwen3_4b.ipynb

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

### kaggle_qwen3_8b.ipynb

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

### kaggle_qwen3_5_0_8b.ipynb

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

### kaggle_qwen3_5_2b.ipynb

Not tested yet.

### kaggle_qwen3_5_4b.ipynb

Not tested yet.

### kaggle_qwen3_5_9b.ipynb

| Item | Result |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / GPUs | 0.33.0 / float16 / 2 (tensor parallel), --max-num-seqs 128 |
| Context (`auto`) | 109 824 tokens (reduced from 262 144 to fit) |
| KV cache | 3.5 GiB (1.74 GiB per GPU) = 110 337 tokens: one full-length request at a time |
| GPU memory (per GPU) | weights 8.83 GiB, other non-torch ~0.9 GiB, peak activations 1.63 GiB, CUDA graphs 0.28 + 0.14 GiB |
| Time | server start ~12 min (weights 76 s, torch.compile 32 s, ~7 min of unlogged warm-up, CUDA graphs 62 s) |
| Speed | ~17–19 tokens/second for one request; ~167 tokens/second total with 8 at once |
| Checked through the public URL | no/wrong key → 401; `/invocations`, `/sleep` → 404; thinking on/off; tool calling via `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; streaming; base64 image described correctly; `http(s)` image URL rejected; secret found in an 8 000-token prompt (10 s). One short answer was wrong ("Toàn" for the capital of Japan with `reasoning_effort="none"`). That request sent no sampling parameters; on Qwen3.5-0.8B the same question was right 9/20 times without them and 20/20 with Qwen's recommended values, so the cause is vLLM's default sampling, not float16 |

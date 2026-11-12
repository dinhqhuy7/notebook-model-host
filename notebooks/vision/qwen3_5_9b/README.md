# Qwen3.5-9B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Vision models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · text, images, video · weights 18.0 GiB (download ~20 GB) · context 262 144 tokens · needs at least ~28 GB of total GPU memory

Multimodal (text, images, video). Hybrid attention (24 linear + 8 full layers): the KV cache costs ~32 KiB per token, so the native 262 144-token context fits on 40 GB GPUs. On T4 the model runs in `float16` instead of its native `bfloat16`: tested on Kaggle 2 × T4. On T4 the first start takes ~11–12 minutes (kernel compilation): step 5 is not stuck.

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_5_9b.ipynb](colab_qwen3_5_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_9b/colab_qwen3_5_9b.ipynb) | ✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 |
| Kaggle | [kaggle_qwen3_5_9b.ipynb](kaggle_qwen3_5_9b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb) | ✓ T4 x2, 2026-11-11 |

## Estimated memory left for context

| GPU | Platform | Qwen3.5-9B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ✗ not enough memory |
| A100 40 GB | Google Colab | ~13 GiB |
| A100 80 GB | Google Colab | ~49 GiB |
| H100 80 GB | Google Colab | ~49 GiB |
| T4 x2 | Kaggle | ~2 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

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

### Kaggle

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

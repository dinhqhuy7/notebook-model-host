# Qwen3-14B: host it on Colab or Kaggle

**English** · [Tiếng Việt](README.vi.md)

Back to [Language models](../README.md) · [all notebooks](../../README.md) · [main README](../../../README.md).

[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) · text · weights 27.5 GiB (download ~30 GB) · context 40 960 tokens · needs at least ~35 GB of total GPU memory

## Notebooks

| Platform | Notebook | Test status |
|---|---|---|
| Google Colab | [colab_qwen3_14b.ipynb](colab_qwen3_14b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_14b/colab_qwen3_14b.ipynb) | ✓ A100 40 GB, 2026-11-10 |
| Kaggle | — | needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total |

## Estimated memory left for context

| GPU | Platform | Qwen3-14B |
|---|---|---|
| T4 | Google Colab | ✗ not enough memory |
| L4 | Google Colab | ✗ not enough memory |
| A100 40 GB | Google Colab | ~6 GiB |
| A100 80 GB | Google Colab | ~42 GiB |
| H100 80 GB | Google Colab | ~42 GiB |

## Test results

Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.

### Google Colab

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

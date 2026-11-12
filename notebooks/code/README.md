# Code models: ready-made notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to [all notebooks](../README.md) · [main README](../../README.md).

Language models specialised in code. The Instruct models chat, call tools and write or fix code; the base models complete code and fill in the middle of it (/v1/completions). Text in, text out.

## Models

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-3B-Instruct](qwen2_5_coder_3b_instruct/README.md) | text | 5.8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb)<br>Not tested yet | [kaggle_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb)<br>Not tested yet |
| [Qwen2.5-Coder-3B](qwen2_5_coder_3b/README.md) | text | 5.8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb)<br>Not tested yet | [kaggle_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb)<br>Not tested yet |

## Shared settings

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. The server sets no sampling parameters and logs no prompts or answers.

## Estimated memory left for context

Choose the GPU before Run all; it needs at least the model's "Min. total GPU memory". Per-model details and test results are in each model's README.

| GPU | Platform | Qwen2.5-Coder-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|
| T4 | Google Colab | ~6 GiB | ~6 GiB |
| L4 | Google Colab | ~12 GiB | ~12 GiB |
| A100 40 GB | Google Colab | ~28 GiB | ~28 GiB |
| A100 80 GB | Google Colab | ~64 GiB | ~64 GiB |
| H100 80 GB | Google Colab | ~64 GiB | ~64 GiB |
| T4 x2 | Kaggle | ~17 GiB | ~17 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM |

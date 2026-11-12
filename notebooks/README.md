# Notebooks

**English** · [Tiếng Việt](README.vi.md)

Back to the [main README](../README.md).

Each notebook hosts one model on one platform. Notebooks are grouped by type of model, one folder per model with its notebooks, GPU table and test results. Pick a model, then a platform where your account has a GPU with enough memory; choose the GPU before Run all (at least the model's "Min. total GPU memory"). Speeds in the test results come from the GPU used for each run, often a modest one; a stronger GPU is faster.

## [Language models](language/README.md)

Models that read and write text: chat, optional thinking, function calling, and writing or completing code. Text in, text out.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B](language/qwen3_0_6b/README.md) | text | 1.4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](language/qwen3_0_6b/colab_qwen3_0_6b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_0_6b.ipynb](language/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-1.7B](language/qwen3_1_7b/README.md) | text | 3.8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](language/qwen3_1_7b/colab_qwen3_1_7b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_1_7b.ipynb](language/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-4B](language/qwen3_4b/README.md) | text | 7.5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](language/qwen3_4b/colab_qwen3_4b.ipynb)<br>✓ T4, 2026-11-10 | [kaggle_qwen3_4b.ipynb](language/qwen3_4b/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-8B](language/qwen3_8b/README.md) | text | 15.3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](language/qwen3_8b/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | [kaggle_qwen3_8b.ipynb](language/qwen3_8b/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 2026-11-10 |
| [Qwen3-14B](language/qwen3_14b/README.md) | text | 27.5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](language/qwen3_14b/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 2026-11-10 | — (needs ~35 GB of GPU memory; Kaggle's 2 × T4 have 30 GB in total) |
| [Qwen2.5-Coder-3B-Instruct](language/qwen2_5_coder_3b_instruct/README.md) | text | 5.8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b_instruct.ipynb](language/qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb)<br>Not tested yet | [kaggle_qwen2_5_coder_3b_instruct.ipynb](language/qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb)<br>Not tested yet |
| [Qwen2.5-Coder-3B](language/qwen2_5_coder_3b/README.md) | text | 5.8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b.ipynb](language/qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb)<br>Not tested yet | [kaggle_qwen2_5_coder_3b.ipynb](language/qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb)<br>Not tested yet |

## [Retrieval models](retrieval/README.md)

Models that score or embed text for search: reranking a list of documents against a query, and turning text into vectors. Text in, scores or vectors out; they do not generate text.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](retrieval/qwen3_reranker_0_6b/README.md) | text (a query and the documents to rank) | 1.1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 2026-11-13 | [kaggle_qwen3_reranker_0_6b.ipynb](retrieval/qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Not tested yet |

## [Vision models](vision/README.md)

Models that see: they read images and video as well as text. The server never downloads media itself: send images and video as base64 `data:` URLs.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3.5-0.8B](vision/qwen3_5_0_8b/README.md) | text, images, video | 1.6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](vision/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_0_8b.ipynb](vision/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-2B](vision/qwen3_5_2b/README.md) | text, images, video | 4.2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](vision/qwen3_5_2b/colab_qwen3_5_2b.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_5_2b.ipynb](vision/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-4B](vision/qwen3_5_4b/README.md) | text, images, video | 8.7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](vision/qwen3_5_4b/colab_qwen3_5_4b.ipynb)<br>✓ L4, 2026-11-11 | [kaggle_qwen3_5_4b.ipynb](vision/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3.5-9B](vision/qwen3_5_9b/README.md) | text, images, video | 18.0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](vision/qwen3_5_9b/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 2026-11-10, A100 80 GB 2026-11-11 | [kaggle_qwen3_5_9b.ipynb](vision/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Instruct](vision/qwen3_vl_2b_instruct/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_instruct.ipynb](vision/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 & A100 40 GB, 2026-11-11 | [kaggle_qwen3_vl_2b_instruct.ipynb](vision/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen3-VL-2B-Thinking](vision/qwen3_vl_2b_thinking/README.md) | text, images, video | 4.0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_thinking.ipynb](vision/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb)<br>✓ T4, 2026-11-11 | [kaggle_qwen3_vl_2b_thinking.ipynb](vision/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb)<br>✓ T4 x2, 2026-11-11 |
| [Qwen2.5-VL-7B-Instruct](vision/qwen2_5_vl_7b_instruct/README.md) | text, images, video | 15.4 GiB | 128 000 | ~23 GB | [colab_qwen2_5_vl_7b_instruct.ipynb](vision/qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb)<br>Not tested yet | [kaggle_qwen2_5_vl_7b_instruct.ipynb](vision/qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb)<br>Not tested yet |

## [Multimodal models](multimodal/README.md)

Models that take several kinds of input and answer in more than one: text, images, audio and video in, text and speech out. They run on vLLM-Omni (a vLLM extension) in several stages, so they need more GPU memory than their size suggests. The server never downloads media itself: send images, audio and video as base64 `data:` URLs.

| Model | Input | Weights | Context | Min. total GPU memory | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Omni-3B](multimodal/qwen2_5_omni_3b/README.md) | text, images, audio, video (answers in text and speech) | 11.2 GiB | 32 768 | ~20 GB | [colab_qwen2_5_omni_3b.ipynb](multimodal/qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb)<br>Not tested yet | [kaggle_qwen2_5_omni_3b.ipynb](multimodal/qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb)<br>Not tested yet |

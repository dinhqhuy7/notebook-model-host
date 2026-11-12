# Model ngôn ngữ: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model đọc và viết văn bản: trò chuyện, có thể suy nghĩ, gọi hàm, viết và hoàn thành code. Vào văn bản, ra văn bản.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B](qwen3_0_6b/README.vi.md) | văn bản | 1,4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](qwen3_0_6b/colab_qwen3_0_6b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_0_6b.ipynb](qwen3_0_6b/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-1.7B](qwen3_1_7b/README.vi.md) | văn bản | 3,8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](qwen3_1_7b/colab_qwen3_1_7b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_1_7b.ipynb](qwen3_1_7b/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-4B](qwen3_4b/README.vi.md) | văn bản | 7,5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](qwen3_4b/colab_qwen3_4b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_4b.ipynb](qwen3_4b/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-8B](qwen3_8b/README.vi.md) | văn bản | 15,3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](qwen3_8b/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | [kaggle_qwen3_8b.ipynb](qwen3_8b/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-14B](qwen3_14b/README.vi.md) | văn bản | 27,5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](qwen3_14b/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | — (cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB) |
| [Qwen2.5-Coder-3B-Instruct](qwen2_5_coder_3b_instruct/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử |
| [Qwen2.5-Coder-3B](qwen2_5_coder_3b/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen2.5-Coder-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|---|---|---|---|---|
| T4 | Google Colab | ~10 GiB | ~8 GiB | ~4 GiB | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ | ~6 GiB | ~6 GiB |
| L4 | Google Colab | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ thiếu bộ nhớ | ~12 GiB | ~12 GiB |
| A100 40 GB | Google Colab | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB | ~6 GiB | ~28 GiB | ~28 GiB |
| A100 80 GB | Google Colab | ~69 GiB | ~66 GiB | ~63 GiB | ~55 GiB | ~42 GiB | ~64 GiB | ~64 GiB |
| H100 80 GB | Google Colab | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB | ~64 GiB | ~64 GiB |
| T4 x2 | Kaggle | ~22 GiB | ~19 GiB | ~16 GiB | ~8 GiB | — | ~17 GiB | ~17 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | — | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

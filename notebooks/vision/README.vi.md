# Model thị giác: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model nhìn được: đọc ảnh và video cùng với văn bản. Server không tự tải media: hãy gửi ảnh và video dạng `data:` base64.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3.5-0.8B](qwen3_5_0_8b/README.vi.md) | văn bản, ảnh, video | 1,6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_5_0_8b.ipynb](qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-2B](qwen3_5_2b/README.vi.md) | văn bản, ảnh, video | 4,2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](qwen3_5_2b/colab_qwen3_5_2b.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_5_2b.ipynb](qwen3_5_2b/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-4B](qwen3_5_4b/README.vi.md) | văn bản, ảnh, video | 8,7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](qwen3_5_4b/colab_qwen3_5_4b.ipynb)<br>✓ L4, 11/11/2026 | [kaggle_qwen3_5_4b.ipynb](qwen3_5_4b/kaggle_qwen3_5_4b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-9B](qwen3_5_9b/README.vi.md) | văn bản, ảnh, video | 18,0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](qwen3_5_9b/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 10/11/2026, A100 80 GB 11/11/2026 | [kaggle_qwen3_5_9b.ipynb](qwen3_5_9b/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3-VL-2B-Instruct](qwen3_vl_2b_instruct/README.vi.md) | văn bản, ảnh, video | 4,0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_instruct.ipynb](qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 & A100 40 GB, 11/11/2026 | [kaggle_qwen3_vl_2b_instruct.ipynb](qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3-VL-2B-Thinking](qwen3_vl_2b_thinking/README.vi.md) | văn bản, ảnh, video | 4,0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_thinking.ipynb](qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_vl_2b_thinking.ipynb](qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen2.5-VL-7B-Instruct](qwen2_5_vl_7b_instruct/README.vi.md) | văn bản, ảnh, video | 15,4 GiB | 128 000 | ~23 GB | [colab_qwen2_5_vl_7b_instruct.ipynb](qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb)<br>Chưa thử | [kaggle_qwen2_5_vl_7b_instruct.ipynb](qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU (model nhỏ như reranker chỉ dùng một); cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct |
|---|---|---|---|---|---|---|---|---|
| T4 | Google Colab | ~7 GiB | ~4 GiB | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ | ~7 GiB | ~7 GiB | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~14 GiB | ~11 GiB | ~7 GiB | ✗ thiếu bộ nhớ | ~13 GiB | ~13 GiB | ~2 GiB |
| A100 40 GB | Google Colab | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB | ~29 GiB | ~29 GiB | ~18 GiB |
| A100 80 GB | Google Colab | ~65 GiB | ~63 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB | ~54 GiB |
| H100 80 GB | Google Colab | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB | ~53 GiB |
| T4 x2 | Kaggle | ~18 GiB | ~16 GiB | ~11 GiB | ~2 GiB | ~18 GiB | ~18 GiB | ~7 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

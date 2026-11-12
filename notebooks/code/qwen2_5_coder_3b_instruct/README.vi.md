# Qwen2.5-Coder-3B-Instruct: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model viết code](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) · văn bản · trọng số 5,8 GiB (tải ~7 GB) · context 32 768 token · cần ít nhất ~11 GB tổng bộ nhớ GPU

Model code (Instruct): chat, gọi hàm, viết, sửa và giải thích code. KV cache tốn 36 KiB mỗi token, nên đủ context 32 768 token trên mọi GPU trong bảng. Không bao giờ suy nghĩ. Nếu bạn không gửi system message, chat template tự thêm "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; hãy gửi system riêng để thay thế. Giấy phép: Qwen Research License, không phải Apache 2.0. `float16` trên T4 chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen2_5_coder_3b_instruct.ipynb](colab_qwen2_5_coder_3b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen2_5_coder_3b_instruct.ipynb](kaggle_qwen2_5_coder_3b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb) | Chưa thử |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen2.5-Coder-3B-Instruct |
|---|---|---|
| T4 | Google Colab | ~6 GiB |
| L4 | Google Colab | ~12 GiB |
| A100 40 GB | Google Colab | ~28 GiB |
| A100 80 GB | Google Colab | ~64 GiB |
| H100 80 GB | Google Colab | ~64 GiB |
| T4 x2 | Kaggle | ~17 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

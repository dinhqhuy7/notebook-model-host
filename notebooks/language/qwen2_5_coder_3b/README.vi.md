# Qwen2.5-Coder-3B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ngôn ngữ](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) · văn bản · trọng số 5,8 GiB (tải ~7 GB) · context 32 768 token · cần ít nhất ~11 GB tổng bộ nhớ GPU

Model code (base): hoàn thành code và điền vào giữa qua `/v1/completions`, với prompt `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`. Không tinh chỉnh để chat hay gọi hàm (dùng bản Instruct cho việc đó). KV cache tốn 36 KiB mỗi token, nên đủ context 32 768 token trên mọi GPU trong bảng. Luôn gửi `temperature`: không gửi thì vLLM lấy mẫu ở 1.0. Giấy phép: Qwen Research License, không phải Apache 2.0. `float16` trên T4 chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen2_5_coder_3b.ipynb](colab_qwen2_5_coder_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen2_5_coder_3b.ipynb](kaggle_qwen2_5_coder_3b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb) | Chưa thử |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Hoàn thành code và điền vào giữa | `prompt` trong `/v1/completions`; điền vào giữa bằng `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>` | ✓ |
| Chat và gọi hàm | `/v1/chat/completions` | ✗ model base không được tinh chỉnh cho chat hay gọi hàm; dùng Qwen2.5-Coder-3B-Instruct |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen2.5-Coder-3B |
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

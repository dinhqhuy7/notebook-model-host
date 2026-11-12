# Model viết code: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model ngôn ngữ chuyên cho code. Bản Instruct trò chuyện, gọi hàm, viết và sửa code; bản base hoàn thành code và điền vào giữa đoạn code (/v1/completions). Vào văn bản, ra văn bản.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-3B-Instruct](qwen2_5_coder_3b_instruct/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b_instruct.ipynb](qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử |
| [Qwen2.5-Coder-3B](qwen2_5_coder_3b/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b.ipynb](qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen2.5-Coder-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|
| T4 | Google Colab | ~6 GiB | ~6 GiB |
| L4 | Google Colab | ~12 GiB | ~12 GiB |
| A100 40 GB | Google Colab | ~28 GiB | ~28 GiB |
| A100 80 GB | Google Colab | ~64 GiB | ~64 GiB |
| H100 80 GB | Google Colab | ~64 GiB | ~64 GiB |
| T4 x2 | Kaggle | ~17 GiB | ~17 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

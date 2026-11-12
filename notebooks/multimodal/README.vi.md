# Model đa phương thức: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model nhận nhiều loại đầu vào và trả lời bằng nhiều dạng: vào là văn bản, ảnh, âm thanh, video; ra là văn bản và giọng nói. Chạy trên vLLM-Omni (phần mở rộng của vLLM) qua nhiều giai đoạn nên cần nhiều bộ nhớ GPU hơn kích thước model. Server không tự tải media: hãy gửi ảnh, âm thanh, video dạng `data:` base64.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Omni-3B](qwen2_5_omni_3b/README.vi.md) | văn bản, ảnh, âm thanh, video (trả lời bằng văn bản và giọng nói) | 11,2 GiB | 32 768 | ~20 GB | [colab_qwen2_5_omni_3b.ipynb](qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb)<br>Chưa thử | [kaggle_qwen2_5_omni_3b.ipynb](qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU (model nhỏ như reranker chỉ dùng một); cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen2.5-Omni-3B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~4 GiB |
| A100 40 GB | Google Colab | ~20 GiB |
| A100 80 GB | Google Colab | ~56 GiB |
| H100 80 GB | Google Colab | ~56 GiB |
| T4 x2 | Kaggle | ~9 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

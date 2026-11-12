# Model truy hồi: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model chấm điểm hoặc nhúng văn bản để tìm kiếm: xếp hạng lại danh sách tài liệu theo câu hỏi, và biến văn bản thành vector. Vào văn bản, ra điểm số hoặc vector; không sinh văn bản.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](qwen3_reranker_0_6b/README.vi.md) | văn bản (câu hỏi và các tài liệu cần xếp hạng) | 1,1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 13/11/2026 | [kaggle_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU (model nhỏ như reranker chỉ dùng một); cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen3-Reranker-0.6B |
|---|---|---|
| T4 | Google Colab | ~10 GiB |
| L4 | Google Colab | ~17 GiB |
| A100 40 GB | Google Colab | ~33 GiB |
| A100 80 GB | Google Colab | ~69 GiB |
| H100 80 GB | Google Colab | ~68 GiB |
| T4 x2 | Kaggle | ~10 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

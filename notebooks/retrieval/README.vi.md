# Model truy hồi: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [mọi notebook](../README.vi.md) · [README chính](../../README.vi.md).

Model chấm điểm hoặc nhúng nội dung để tìm kiếm: xếp hạng lại danh sách tài liệu theo câu hỏi, và biến nội dung thành vector. Vào văn bản (có model nhận cả ảnh và video), ra điểm số hoặc vector; không sinh văn bản. Server không tự tải media: hãy gửi ảnh và video dạng `data:` base64.

## Model

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](qwen3_reranker_0_6b/README.vi.md) | văn bản (câu hỏi và các tài liệu cần xếp hạng) | 1,1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 13/11/2026 | [kaggle_qwen3_reranker_0_6b.ipynb](qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Chưa thử |
| [Qwen3-Reranker-4B](qwen3_reranker_4b/README.vi.md) | văn bản (câu hỏi và các tài liệu cần xếp hạng) | 7,5 GiB | 40 960 | ~13 GB | [colab_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb)<br>Chưa thử | [kaggle_qwen3_reranker_4b.ipynb](qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb)<br>Chưa thử |
| [Qwen3-VL-Reranker-2B](qwen3_vl_reranker_2b/README.vi.md) | văn bản, ảnh, video (câu hỏi và tài liệu) | 4,0 GiB | 32 768 | ~10 GB | [colab_qwen3_vl_reranker_2b.ipynb](qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb)<br>Chưa thử | [kaggle_qwen3_vl_reranker_2b.ipynb](qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb)<br>Chưa thử |

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU (model nhỏ như reranker chỉ dùng một); cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model. Chi tiết và kết quả chạy thử nằm trong README của từng model.

| GPU | Nền tảng | Qwen3-Reranker-0.6B | Qwen3-Reranker-4B | Qwen3-VL-Reranker-2B |
|---|---|---|---|---|
| T4 | Google Colab | ~10 GiB | ~3 GiB | ~6 GiB |
| L4 | Google Colab | ~17 GiB | ~10 GiB | ~13 GiB |
| A100 40 GB | Google Colab | ~33 GiB | ~26 GiB | ~29 GiB |
| A100 80 GB | Google Colab | ~69 GiB | ~62 GiB | ~65 GiB |
| H100 80 GB | Google Colab | ~68 GiB | ~61 GiB | ~64 GiB |
| T4 x2 | Kaggle | ~22 GiB | ~15 GiB | ~18 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

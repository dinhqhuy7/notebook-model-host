# Qwen3-Reranker-0.6B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · văn bản (câu hỏi và các tài liệu cần xếp hạng) · trọng số 1,1 GiB (tải ~2 GB) · context 40 960 token · cần ít nhất ~6 GB tổng bộ nhớ GPU

Reranker: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn (Qwen cho biết kết quả tốt hơn 1% đến 5% khi có instruction riêng cho tác vụ, viết bằng tiếng Anh); không gửi thì mặc định là "Given a web search query, retrieve relevant passages that answer the query". Đa ngôn ngữ. Model nhỏ nên notebook chỉ dùng một GPU, kể cả trên 2 × T4 của Kaggle. Model không viết văn bản (không chat). Giấy phép: Apache 2.0. `float16` trên T4 chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_0_6b.ipynb](colab_qwen3_reranker_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen3_reranker_0_6b.ipynb](kaggle_qwen3_reranker_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb) | Chưa thử |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Xếp hạng tài liệu theo câu hỏi | `POST /rerank`, `/v1/rerank`, `/v2/rerank` | ✓ |
| Chấm điểm từng cặp câu hỏi - tài liệu | `POST /score`, `/v1/score` | ✓ |
| Instruction theo tác vụ trong từng request | `instruction` trong `/rerank` và `/score` | ✓ |
| Cắt câu hỏi và tài liệu dài | `max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`) | ✓ |
| Sinh văn bản / chat | `/v1/chat/completions` | ✗ model không sinh văn bản |
| Embedding | `/v1/embeddings` | ✗ không phải model embedding; Qwen3-Embedding mới là |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-Reranker-0.6B |
|---|---|---|
| T4 | Google Colab | ~10 GiB |
| L4 | Google Colab | ~17 GiB |
| A100 40 GB | Google Colab | ~33 GiB |
| A100 80 GB | Google Colab | ~69 GiB |
| H100 80 GB | Google Colab | ~68 GiB |
| T4 x2 | Kaggle | ~10 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

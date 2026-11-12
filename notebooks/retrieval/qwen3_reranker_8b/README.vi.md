# Qwen3-Reranker-8B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) · văn bản (câu hỏi và các tài liệu cần xếp hạng) · trọng số 15,2 GiB (tải ~17 GB) · context 40 960 token · cần ít nhất ~23 GB tổng bộ nhớ GPU

Reranker, cỡ lớn nhất của họ này: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Bảng điểm của Qwen cho thấy nó chỉ nhỉnh hơn Qwen3-Reranker-4B một chút (CMTEB-R 77,45 so với 75,94, MTEB-R 69,02 so với 69,76) mà tốn gấp đôi bộ nhớ: nên thử bản 4B trước. Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là "Given a web search query, retrieve relevant passages that answer the query". Đa ngôn ngữ. Trọng số 15,25 GiB nên **một T4 không đủ**: trên Colab dùng A100 (L4 chỉ còn khoảng 2 GiB cho KV cache), trên 2 × T4 của Kaggle model được chia ra cả hai GPU (tensor parallel) và đủ khoảng 48 000 token context. KV cache tốn 144 KiB mỗi token. Model không viết văn bản (không chat). Giấy phép: Apache 2.0. Chưa thử: `float16` trên T4, chia hai GPU, và đầu ra embedding riêng (không gắn chung) của model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_8b.ipynb](colab_qwen3_reranker_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_8b/colab_qwen3_reranker_8b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen3_reranker_8b.ipynb](kaggle_qwen3_reranker_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_8b/kaggle_qwen3_reranker_8b.ipynb) | Chưa thử |

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

| GPU | Nền tảng | Qwen3-Reranker-8B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~2 GiB |
| A100 40 GB | Google Colab | ~18 GiB |
| A100 80 GB | Google Colab | ~54 GiB |
| H100 80 GB | Google Colab | ~53 GiB |
| T4 x2 | Kaggle | ~7 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

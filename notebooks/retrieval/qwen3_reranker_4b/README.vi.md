# Qwen3-Reranker-4B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) · văn bản (câu hỏi và các tài liệu cần xếp hạng) · trọng số 7,5 GiB (tải ~8 GB) · context 40 960 token · cần ít nhất ~13 GB tổng bộ nhớ GPU

Reranker, cỡ lớn và chính xác hơn của họ này: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Bảng điểm của Qwen đặt nó trên Qwen3-Reranker-0.6B (ví dụ MTEB-R 69,76 so với 65,80). Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là "Given a web search query, retrieve relevant passages that answer the query". Đa ngôn ngữ. KV cache tốn 144 KiB mỗi token, nên trên một T4 vLLM tự giảm context cho vừa (khoảng 24 000 token; model cho phép 40 960): xem bảng bộ nhớ. Trên 2 × T4 của Kaggle model được chia ra cả hai GPU (tensor parallel), đủ chỗ cho context đầy đủ. Model không viết văn bản (không chat). Giấy phép: Apache 2.0. `float16` trên T4 và việc chia hai GPU chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_4b.ipynb](colab_qwen3_reranker_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen3_reranker_4b.ipynb](kaggle_qwen3_reranker_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb) | Chưa thử |

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

| GPU | Nền tảng | Qwen3-Reranker-4B |
|---|---|---|
| T4 | Google Colab | ~3 GiB |
| L4 | Google Colab | ~10 GiB |
| A100 40 GB | Google Colab | ~26 GiB |
| A100 80 GB | Google Colab | ~62 GiB |
| H100 80 GB | Google Colab | ~61 GiB |
| T4 x2 | Kaggle | ~15 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

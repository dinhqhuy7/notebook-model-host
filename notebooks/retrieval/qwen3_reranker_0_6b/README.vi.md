# Qwen3-Reranker-0.6B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · văn bản (câu hỏi và các tài liệu cần xếp hạng) · trọng số 1,1 GiB (tải ~2 GB) · context 40 960 token · cần ít nhất ~6 GB tổng bộ nhớ GPU

Reranker: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn (Qwen cho biết kết quả tốt hơn 1% đến 5% khi có instruction riêng cho tác vụ, viết bằng tiếng Anh); không gửi thì mặc định là "Given a web search query, retrieve relevant passages that answer the query". Đa ngôn ngữ. Model nhỏ nên notebook chỉ dùng một GPU, kể cả trên 2 × T4 của Kaggle. Model không viết văn bản (không chat). Giấy phép: Apache 2.0. `float16` trên T4 chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_0_6b.ipynb](colab_qwen3_reranker_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb) | ✓ T4, 13/11/2026 |
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

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 11,35 GiB = 106 240 token, 2,59 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 1,12 GiB (1,18 GiB cùng phần non-torch khác), activation đỉnh 0,57 GiB, CUDA graph 0,05 GiB |
| Thời gian | bật server ~2 phút (trọng số 3 giây, torch.compile 30 giây, CUDA graph 11 + 5 giây) |
| Tốc độ | ~45 tài liệu/giây với tài liệu 200 token (~12 000 token/giây); 100 tài liệu trong 2,3 giây; 8 request, mỗi request 16 tài liệu, cùng lúc trong 0,8 giây. Hai tài liệu tổng ~22 000 token mất 30 giây lần đầu (1,6 giây khi lặp lại, nhờ cache prefix) |
| Đã kiểm qua URL công khai | 11/11 phép thử: không key hoặc sai key → 401 ở `/rerank` và `/score`; `/invocations`, `/sleep` và `/v1/chat/completions` → 404; `/rerank` xếp tài liệu đúng lên đầu (điểm 0,999 so với 0,0) và `top_n` cắt danh sách; `/score` cho cùng điểm theo thứ tự gửi; câu hỏi tiếng Việt xếp đúng; `instruction` đi vào prompt (prompt dài ra theo nó, và gửi đúng văn bản mặc định thì điểm giống hệt) nhưng làm điểm của model nhỏ này đổi rất ít (0,9995 → 0,9979 với instruction khác); `max_tokens_per_doc` cắt 1 285 token còn 108; tìm đúng tài liệu giữa ~22 000 token (0,645 so với 0,002). Log server cho thấy chat template đã nạp (không có cảnh báo thiếu template), `float16` chạy được trên T4 và `--max-model-len auto` chạy được với runner pooling |

### Kaggle

Chưa thử.

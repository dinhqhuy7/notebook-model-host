# Qwen3-Reranker-4B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) · văn bản (câu hỏi và các tài liệu cần xếp hạng) · trọng số 7,5 GiB (tải ~8 GB) · context 40 960 token · cần ít nhất ~13 GB tổng bộ nhớ GPU

Reranker, cỡ lớn và chính xác hơn của họ này: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Bảng điểm của Qwen đặt nó trên Qwen3-Reranker-0.6B (ví dụ MTEB-R 69,76 so với 65,80). Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là "Given a web search query, retrieve relevant passages that answer the query". Đa ngôn ngữ. KV cache tốn 144 KiB mỗi token, nên trên một T4 vLLM tự giảm context còn 32 128 token (đã đo; model cho phép 40 960). Chấm điểm chậm hơn bản 0,6B (khoảng 9 tài liệu 200 token mỗi giây trên T4): trên T4 nên giữ mỗi cặp dưới ~10 000 token vì tunnel cắt request sau 100 giây. Trên 2 × T4 của Kaggle model được chia ra cả hai GPU (tensor parallel), đủ chỗ cho context đầy đủ. Model không viết văn bản (không chat). Giấy phép: Apache 2.0. Đã chạy thử trên Colab T4 (`float16`); việc chia hai GPU trên Kaggle chưa chạy thử.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_reranker_4b.ipynb](colab_qwen3_reranker_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb) | ✓ T4, 13/11/2026 |
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

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 32 128 token (tự giảm từ 40 960 cho vừa) |
| KV cache | 4,42 GiB = 32 144 token: một request dài tối đa mỗi lúc |
| Bộ nhớ GPU | trọng số 7,56 GiB (7,58 GiB cùng phần non-torch khác), activation đỉnh 1,12 GiB, CUDA graph 0,37 + 0,08 GiB |
| Thời gian | bật server ~3 phút 10 giây (trọng số 36 giây, torch.compile 37 giây, CUDA graph 15 + 15 giây) |
| Tốc độ | ~9 tài liệu/giây với tài liệu 200 token (~2 600 token/giây); 100 tài liệu ngắn trong 6,8 giây; 8 request, mỗi request 16 tài liệu, cùng lúc trong 1,9 giây. Hai tài liệu tổng ~22 000 token mất 77 giây lần đầu, sát giới hạn 100 giây của tunnel: trên T4 nên giữ mỗi cặp câu hỏi - tài liệu dưới ~10 000 token hoặc chia nhỏ |
| Đã kiểm qua URL công khai | 11/11 phép thử: không key hoặc sai key → 401 ở `/rerank` và `/score`; `/invocations`, `/sleep` và `/v1/chat/completions` → 404; `/rerank` xếp tài liệu đúng lên đầu (điểm 0,998 so với 0,0) và `top_n` cắt danh sách; `/score` cho cùng điểm theo thứ tự gửi; câu hỏi tiếng Việt xếp đúng; `instruction` đi vào prompt (prompt dài ra theo nó, gửi đúng văn bản mặc định thì điểm giống hệt) và làm điểm đổi nhiều hơn bản 0,6B (0,9982 → 0,9587 với instruction khác); `max_tokens_per_doc` cắt 1 285 token còn 108; tìm đúng tài liệu giữa ~22 000 token (0,794 so với 0,001). Log server cho thấy chat template đã nạp (không có cảnh báo thiếu template) và `float16` chạy được trên T4 |

### Kaggle

Chưa thử.

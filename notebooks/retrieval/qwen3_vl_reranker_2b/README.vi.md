# Qwen3-VL-Reranker-2B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · văn bản, ảnh, video (câu hỏi và tài liệu) · trọng số 4,0 GiB (tải ~5 GB) · context 32 768 token · cần ít nhất ~10 GB tổng bộ nhớ GPU

Reranker đa phương thức: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Câu hỏi và từng tài liệu có thể là văn bản, ảnh hoặc ảnh chụp màn hình, video, hoặc kết hợp (`{"content": [...]}`); media gửi dạng URL `data:` base64. Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là "Given a search query, retrieve relevant candidates that answer the query." Qwen ghi context 32K; `auto` của vLLM có thể mở nhiều hơn khi đủ bộ nhớ, bảng bộ nhớ cho biết vừa bao nhiêu. vLLM mặc định đọc tối đa 32 khung hình của một video; `mm_processor_kwargs` được chuyển cho bộ xử lý ảnh và video. Đa ngôn ngữ (hơn 30 ngôn ngữ). Model không viết văn bản (không chat). Trên 2 × T4 của Kaggle model được chia ra cả hai GPU (tensor parallel). Giấy phép: Apache 2.0. Chưa thử: `float16` trên T4, chia hai GPU và video.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen3_vl_reranker_2b.ipynb](kaggle_qwen3_vl_reranker_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb) | Chưa thử |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Xếp hạng tài liệu văn bản theo câu hỏi văn bản | `POST /rerank`, `/v1/rerank`, `/v2/rerank` | ✓ |
| Xếp hạng ảnh và ảnh chụp màn hình | `documents`: `{"content": [{"type": "image_url", ...}]}` trong `/rerank`, `/score` (URL `data:` base64) | ✓ |
| Xếp hạng video | `documents`: `{"content": [{"type": "video_url", ...}]}` trong `/rerank`, `/score` (URL `data:` base64) | ✓ |
| Tài liệu hỗn hợp (văn bản kèm ảnh) và câu hỏi là ảnh hoặc video | `query` hoặc `documents` dạng `{"content": [text, image_url, video_url ...]}` trong `/rerank`, `/score` | ✓ |
| Chấm điểm từng cặp câu hỏi - tài liệu | `POST /score`, `/v1/score` | ✓ |
| Instruction theo tác vụ trong từng request | `instruction` trong `/rerank` và `/score` | ✓ |
| Cắt câu hỏi và tài liệu dài | `max_tokens_per_query`, `max_tokens_per_doc` (`/rerank`, `/score`) | ✓ |
| Thiết lập bộ xử lý ảnh và video | `mm_processor_kwargs` trong `/rerank`, `/score` (chuyển cho bộ xử lý; chưa thử ở đây) | ✓ |
| Ảnh và video qua link http(s) | `image_url` / `video_url` `https://...` / `/rerank` | ✗ chặn có chủ đích: server không tự tải link; hãy gửi base64 |
| Sinh văn bản / chat | `/v1/chat/completions` | ✗ model không sinh văn bản |
| Embedding | `/v1/embeddings` | ✗ không phải model embedding; Qwen3-VL-Embedding mới là |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-VL-Reranker-2B |
|---|---|---|
| T4 | Google Colab | ~6 GiB |
| L4 | Google Colab | ~13 GiB |
| A100 40 GB | Google Colab | ~29 GiB |
| A100 80 GB | Google Colab | ~65 GiB |
| H100 80 GB | Google Colab | ~64 GiB |
| T4 x2 | Kaggle | ~18 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

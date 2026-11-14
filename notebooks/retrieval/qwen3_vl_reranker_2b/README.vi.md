# Qwen3-VL-Reranker-2B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model truy hồi](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · văn bản, ảnh, video (câu hỏi và tài liệu) · trọng số 4,0 GiB (tải ~5 GB) · context 32 768 token · cần ít nhất ~10 GB tổng bộ nhớ GPU

Reranker đa phương thức: gửi một câu hỏi và danh sách tài liệu, nhận điểm liên quan từ 0 đến 1 cho từng tài liệu (`/rerank` trả về đã sắp xếp, `/score` giữ nguyên thứ tự bạn gửi). Câu hỏi và từng tài liệu có thể là văn bản, ảnh hoặc ảnh chụp màn hình, video, hoặc kết hợp (`{"content": [...]}`); media gửi dạng URL `data:` base64. Mỗi cặp được chấm bằng đúng prompt Qwen đã huấn luyện; đặt `instruction` trong request để mô tả tác vụ của bạn, không gửi thì mặc định là "Given a search query, retrieve relevant candidates that answer the query." Qwen ghi context 32K; `auto` của vLLM có thể mở nhiều hơn khi đủ bộ nhớ, bảng bộ nhớ cho biết vừa bao nhiêu. vLLM mặc định đọc tối đa 32 khung hình của một video; request có thể đặt cách xử lý ảnh và video bằng `mm_processor_kwargs` (notebook bật `--trust-request-mm-kwargs` cho server, vLLM cần cờ này). Đa ngôn ngữ (hơn 30 ngôn ngữ). Model không viết văn bản (không chat). Trên 2 × T4 của Kaggle model được chia ra cả hai GPU (tensor parallel). Giấy phép: Apache 2.0. Đã chạy thử trên Colab T4 (`float16`) khi chưa có cờ `--trust-request-mm-kwargs`; chưa thử: cờ đó và chia hai GPU.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b.ipynb) | ✓ T4, 13/11/2026 |
| Kaggle | [kaggle_qwen3_vl_reranker_2b.ipynb](kaggle_qwen3_vl_reranker_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/kaggle_qwen3_vl_reranker_2b.ipynb) | Chưa thử |

## Bản cập nhật chưa chạy thử

Các notebook này mới hơn bản đã chạy thử. Chúng chưa được chạy nên chưa có số đo. Dùng bản đã chạy thử, trừ khi bạn muốn thử bản cập nhật; khi bản cập nhật được chạy thử, nó thay cho bản đã chạy thử.

| Bản cập nhật (chưa chạy thử) | Thay cho |
|---|---|
| ⚠ [colab_qwen3_vl_reranker_2b_update.ipynb](colab_qwen3_vl_reranker_2b_update.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/retrieval/qwen3_vl_reranker_2b/colab_qwen3_vl_reranker_2b_update.ipynb) | [colab_qwen3_vl_reranker_2b.ipynb](colab_qwen3_vl_reranker_2b.ipynb) |

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
| Thiết lập bộ xử lý ảnh và video | `mm_processor_kwargs` trong `/rerank`, `/score` (cần `--trust-request-mm-kwargs`: đã bật ở bản `_update` Colab và bản Kaggle, chưa có ở bản Colab đã chạy thử) | ✓ |
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

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 68 160 token (tự giảm từ 262 144 của config cho vừa; Qwen ghi 32K) |
| KV cache | 7,28 GiB = 68 176 token: một request dài tối đa mỗi lúc |
| Bộ nhớ GPU | trọng số 4,08 GiB (4,19 GiB cùng phần non-torch khác), activation đỉnh 1,63 GiB, CUDA graph 0,29 + 0,06 GiB |
| Thời gian | bật server ~2 phút 41 giây (trọng số 3 giây, torch.compile 42 giây, khởi động engine 109 giây) |
| Tốc độ | ~17 tài liệu văn bản 200 token/giây (~4 700 token/giây); 100 tài liệu ngắn trong 3,2 giây; 8 request × 16 tài liệu trong 0,9 giây; 8 request × 4 ảnh trong 1,0 giây; một video nhỏ tốn 1 206 token; hai tài liệu văn bản tổng ~22 000 token mất 31,7 giây |
| Đã kiểm qua URL công khai | 20/20 phép thử: key và đường dẫn bị chặn như các reranker text; xếp hạng văn bản, tiếng Việt, `top_n`, `/score` bằng `/rerank`, `instruction` vào prompt, `max_tokens_per_doc`, 100 tài liệu, song song; ảnh (tài liệu ảnh base64 xếp đầu, câu hỏi là ảnh với tài liệu văn bản, tài liệu hỗn hợp văn bản + ảnh, ảnh chụp chữ), tài liệu video base64 xếp đầu (0,86 so với 0,38). Link `https://` bị từ chối HTTP 400 và link `file://` HTTP 500, không tải gì cả. Bản notebook này chưa có `--trust-request-mm-kwargs` nên vLLM từ chối `mm_processor_kwargs` trong request (HTTP 400): bản cập nhật của notebook này bật nó. Log cho thấy chat template đã nạp và `float16` chạy được với bộ mã hoá ảnh; bộ xử lý video cảnh báo một số video tốn nhiều token hơn mã tham chiếu của Qwen |

### Kaggle

Chưa thử.

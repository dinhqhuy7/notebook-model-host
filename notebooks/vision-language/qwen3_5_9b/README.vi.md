# Qwen3.5-9B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ảnh và video](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · văn bản, ảnh, video · trọng số 18,0 GiB (tải ~20 GB) · context 262 144 token · cần ít nhất ~28 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_5_9b.ipynb](colab_qwen3_5_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/colab_qwen3_5_9b.ipynb) | ✓ A100 40 GB 10/11/2026, A100 80 GB 11/11/2026 |
| Kaggle | [kaggle_qwen3_5_9b.ipynb](kaggle_qwen3_5_9b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb) | ✓ T4 x2, 11/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3.5-9B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ✗ thiếu bộ nhớ |
| A100 40 GB | Google Colab | ~13 GiB |
| A100 80 GB | Google Colab | ~49 GiB |
| H100 80 GB | Google Colab | ~49 GiB |
| T4 x2 | Kaggle | ~2 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

| Mục | Kết quả |
|---|---|
| Lần chạy A100 80 GB | 1 × NVIDIA A100-SXM4-80GB, `--max-num-seqs 128`: context 262 144, KV cache 49,25 GiB = 1 592 668 token (6,08 request dài tối đa cùng lúc), bật server ~5 phút 50 giây, ~88 token/giây cho một request, ~505 tổng khi 8 request; qua 13/13 phép thử, prompt 24 000 token trong 3,2 giây. Các dòng dưới là lần chạy A100 40 GB |
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 13,17 GiB = 425 788 token, 1,62 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 17,57 GiB, non-torch khác ~2,5 GiB (bộ mã hoá ảnh, trạng thái linear attention), activation đỉnh 2,27 GiB, CUDA graph 0,27 GiB |
| Thời gian | bật server 6 phút (trọng số 10 giây, torch.compile 28 giây, FlashInfer JIT ~1 phút 50 giây, ~1 phút 40 giây khởi động không ghi log) |
| Tốc độ | ~73 token/giây cho một request; ~500 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort="none"`); gọi hàm qua `qwen3_coder`; JSON schema; mô tả đúng ảnh base64; link ảnh `http(s)` bị từ chối (HTTP 400); Anthropic `/v1/messages`; stream; tìm đúng mã giấu giữa prompt 57 242 token (6 giây) |
| 11 ảnh Wikimedia | đều chạy (10 ảnh song song 8,7 giây; 3 ảnh trong một request, 2 125 token, 3,9 giây). Nhận đúng cảnh vật, món ăn, địa danh và công thức viết tay trên bảng đen; hoá đơn bị gấp: số tiền đúng nhưng tên cửa hàng và ngày sai; đọc sai biển cấm vượt của Việt Nam |
| 13 tác vụ văn bản (chấm tự động) | đúng: toán nhiều bước và câu đố logic (có suy nghĩ), tóm tắt, sửa chính tả, trích JSON, nhớ hội thoại nhiều lượt, gọi hàm 2 bước. Sai: dịch thành ngữ theo nghĩa đen kèm lỗi chính tả, "sông dài nhất chảy qua Việt Nam" trả lời sông Hồng, thơ lục bát sai luật 6-8. Viết code khi tắt suy nghĩ: qua 11/14 test trong 8 giây; khi bật suy nghĩ model lặp 30 000 token (7 phút) mà không ra code, nên tắt suy nghĩ khi viết code |

### Kaggle

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), --max-num-seqs 128 |
| Context (`auto`) | 109 824 token (tự giảm từ 262 144 cho vừa) |
| KV cache | 3,5 GiB (1,74 GiB mỗi GPU) = 110 337 token: một request dài tối đa mỗi lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 8,83 GiB, non-torch khác ~0,9 GiB, activation đỉnh 1,63 GiB, CUDA graph 0,28 + 0,14 GiB |
| Thời gian | bật server ~12 phút (trọng số 76 giây, torch.compile 32 giây, ~7 phút khởi động không ghi log, CUDA graph 62 giây) |
| Tốc độ | ~17–19 token/giây cho một request; ~167 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; mô tả đúng ảnh base64; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 8 000 token (10 giây). Một câu trả lời ngắn bị sai ("Toàn" cho thủ đô Nhật Bản với `reasoning_effort="none"`). Request đó không gửi tham số sinh; trên Qwen3.5-0.8B cùng câu hỏi đúng 9/20 lần khi không gửi và 20/20 khi gửi tham số Qwen khuyên, nên nguyên nhân là tham số sinh mặc định của vLLM, không phải float16 |

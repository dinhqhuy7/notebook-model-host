# Host model mở trên Kaggle: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [README chính](../../README.vi.md).

## Notebook

| Notebook | Model | Trọng số | Dung lượng tải | Tổng bộ nhớ GPU tối thiểu | Chạy thử |
|---|---|---|---|---|---|
| [kaggle_qwen3_0_6b.ipynb](kaggle_qwen3_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_0_6b.ipynb) | `Qwen/Qwen3-0.6B` | 1,4 GiB | ~2 GB | ~6 GB | ✓ T4 x2, 10/11/2026 |
| [kaggle_qwen3_1_7b.ipynb](kaggle_qwen3_1_7b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_1_7b.ipynb) | `Qwen/Qwen3-1.7B` | 3,8 GiB | ~4 GB | ~9 GB | ✓ T4 x2, 10/11/2026 |
| [kaggle_qwen3_4b.ipynb](kaggle_qwen3_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_4b.ipynb) | `Qwen/Qwen3-4B` | 7,5 GiB | ~8 GB | ~13 GB | ✓ T4 x2, 10/11/2026 |
| [kaggle_qwen3_8b.ipynb](kaggle_qwen3_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_8b.ipynb) | `Qwen/Qwen3-8B` | 15,3 GiB | ~16 GB | ~21 GB | ✓ T4 x2, 10/11/2026 |
| [kaggle_qwen3_5_9b.ipynb](kaggle_qwen3_5_9b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18,0 GiB | ~20 GB | ~28 GB | ✓ T4 x2, 11/11/2026 |

**Qwen3.5-9B:** Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4, khởi động mất ~12 phút.

**Qwen3-14B:** không có notebook cho Kaggle: cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB.

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3.5-9B |
|---|---|---|---|---|---|
| T4 x2 | ~22 GiB | ~19 GiB | ~16 GiB | ~8 GiB | ~2 GiB |
| P100 | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

### kaggle_qwen3_0_6b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 23,5 GiB (11,75 GiB mỗi GPU) = 220 080 token, 5,37 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 0,57 GiB, activation đỉnh 0,57 GiB, CUDA graph 0,18 GiB |
| Thời gian | khởi động engine 80 giây (torch.compile 30 giây, CUDA graph 19 giây) |
| Tốc độ | ~104 token/giây cho một request; ~834 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 18 000 token mất 28,5 giây, bằng nửa thời gian trên 1 × T4 |

### kaggle_qwen3_1_7b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 21,4 GiB (10,72 GiB mỗi GPU) = 200 752 token, 4,90 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 1,61 GiB, activation đỉnh 0,56 GiB, CUDA graph 0,18 GiB |
| Thời gian | khởi động engine 76 giây (torch.compile 27 giây, CUDA graph 19 giây) |
| Tốc độ | ~73 token/giây cho một request; ~506 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 18 000 token mất 29,6 giây, khoảng nửa thời gian trên 1 × T4 |

### kaggle_qwen3_4b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 16,8 GiB (8,41 GiB mỗi GPU) = 122 416 token, 2,99 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 3,82 GiB, activation đỉnh 0,67 GiB, CUDA graph 0,2 GiB |
| Thời gian | khởi động engine 85 giây (trọng số 10 giây, torch.compile 32 giây, CUDA graph 21 giây) |
| Tốc độ | ~42 token/giây cho một request; ~290 token/giây tổng khi 8 request cùng lúc (khoảng 1,7 lần Colab 1 × T4) |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Prompt 27 000 token bị lỗi 524 khi không stream, trả lời đúng khi stream (token đầu sau 15 giây): prompt dài nên dùng stream |

### kaggle_qwen3_8b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 9,2 GiB (4,58 GiB mỗi GPU) = 66 704 token, 1,63 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 7,64 GiB, CUDA graph 0,42 + 0,23 GiB |
| Thời gian | nạp trọng số 52 giây, khởi động engine 90 giây (torch.compile 33 giây, CUDA graph 26 giây) |
| Tốc độ | một request ~8–26 token/giây, không đều giữa các lần; ~170 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Suy nghĩ 1 172 token mất 89 giây, prompt 18 000 token mất 76 giây, sát giới hạn 100 giây: nên dùng stream |

### kaggle_qwen3_5_9b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), --max-num-seqs 128 |
| Context (`auto`) | 109 824 token (tự giảm từ 262 144 cho vừa) |
| KV cache | 3,5 GiB (1,74 GiB mỗi GPU) = 110 337 token: một request dài tối đa mỗi lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 8,83 GiB, non-torch khác ~0,9 GiB, activation đỉnh 1,63 GiB, CUDA graph 0,28 + 0,14 GiB |
| Thời gian | bật server ~12 phút (trọng số 76 giây, torch.compile 32 giây, ~7 phút khởi động không ghi log, CUDA graph 62 giây) |
| Tốc độ | ~17–19 token/giây cho một request; ~167 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; mô tả đúng ảnh base64; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 8 000 token (10 giây). Một câu trả lời ngắn bị sai ("Toàn" cho thủ đô Nhật Bản với `reasoning_effort="none"`); chưa tách được nguyên nhân (float16 hay tham số sinh) |

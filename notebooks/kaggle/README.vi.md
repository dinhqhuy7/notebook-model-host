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
| [kaggle_qwen3_5_0_8b.ipynb](kaggle_qwen3_5_0_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_0_8b.ipynb) | `Qwen/Qwen3.5-0.8B` | 1,6 GiB | ~2 GB | ~10 GB | ✓ T4 x2, 11/11/2026 |
| [kaggle_qwen3_5_2b.ipynb](kaggle_qwen3_5_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_2b.ipynb) | `Qwen/Qwen3.5-2B` | 4,2 GiB | ~5 GB | ~12 GB | ✓ T4 x2, 11/11/2026 |
| [kaggle_qwen3_5_4b.ipynb](kaggle_qwen3_5_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_4b.ipynb) | `Qwen/Qwen3.5-4B` | 8,7 GiB | ~9 GB | ~17 GB | ✓ T4 x2, 11/11/2026 |
| [kaggle_qwen3_5_9b.ipynb](kaggle_qwen3_5_9b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/kaggle/kaggle_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18,0 GiB | ~20 GB | ~28 GB | ✓ T4 x2, 11/11/2026 |

**Qwen3.5-0.8B:** Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-2B:** Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-4B:** Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-9B:** Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3-14B:** không có notebook cho Kaggle: cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB.

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|---|---|---|---|---|
| T4 x2 | ~22 GiB | ~19 GiB | ~16 GiB | ~8 GiB | ~18 GiB | ~16 GiB | ~11 GiB | ~2 GiB |
| P100 | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

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

### kaggle_qwen3_5_0_8b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 20,4 GiB (10,19 GiB mỗi GPU) = 1 758 728 token, 6,71 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 0,84 GiB, non-torch khác ~0,9 GiB, CUDA graph 0,21 + 0,12 GiB |
| Thời gian | bật server ~11,5 phút (trọng số 7 giây, torch.compile 30 giây, ~7,7 phút biên dịch kernel không ghi log, CUDA graph 33 giây) |
| Tốc độ | ~120 token/giây cho một request (có suy nghĩ); ~289 token/giây tổng khi 8 request, chậm hơn Colab 1 × T4 (~640): model nhỏ thế này thì chia 2 GPU tốn hơn lợi |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; suy nghĩ khi gửi `enable_thinking: true`; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (7,9 giây, nhanh hơn 1 × T4). Ảnh base64: đúng chữ và hình chữ nhật đỏ, hình tròn xanh dương bị gọi là xanh lá (giống trên Colab) |

### kaggle_qwen3_5_2b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 17,3 GiB (8,64 GiB mỗi GPU) = 1 490 675 token, 5,69 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 2,12 GiB, non-torch khác ~0,9 GiB, CUDA graph 0,22 + 0,12 GiB |
| Thời gian | bật server ~10 phút (trọng số 7 giây, torch.compile 25 giây, ~6,6 phút biên dịch kernel không ghi log, CUDA graph 32 giây) |
| Tốc độ | ~86 token/giây cho một request (có suy nghĩ); ~272 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; suy nghĩ khi gửi `enable_thinking: true`; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (9,6 giây). Ảnh base64: đúng chữ, hình chữ nhật bị gọi là hình vuông |
| Tham số sinh | một câu hỏi ngắn, 20 lần mỗi cách: temperature 0 đúng 20/20; tham số Qwen chung (0.7/0.8, top_k 20, presence 1.5) 17/20; tham số Qwen cho văn bản của 0.8B/2B (1.0/1.0, top_k 20, presence 2.0) 9/20. Câu trả lời ngắn cần chính xác thì dùng temperature thấp |

### kaggle_qwen3_5_4b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 11,9 GiB (5,96 GiB mỗi GPU) = 385 659 token, 1,47 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 4,38 GiB, CUDA graph 0,58 + 0,22 GiB |
| Thời gian | bật server ~11 phút (trọng số 25 giây, torch.compile 31 giây, ~7 phút biên dịch kernel không ghi log, CUDA graph 55 giây) |
| Tốc độ | ~44 token/giây cho một request (có suy nghĩ); ~258 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định có suy nghĩ (791 token, 18 giây); gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (~20 giây). Ảnh base64: đúng chữ và cả hai màu, hình chữ nhật bị gọi là hình vuông |

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
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; mô tả đúng ảnh base64; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 8 000 token (10 giây). Một câu trả lời ngắn bị sai ("Toàn" cho thủ đô Nhật Bản với `reasoning_effort="none"`). Request đó không gửi tham số sinh; trên Qwen3.5-0.8B cùng câu hỏi đúng 9/20 lần khi không gửi và 20/20 khi gửi tham số Qwen khuyên, nên nguyên nhân là tham số sinh mặc định của vLLM, không phải float16 |

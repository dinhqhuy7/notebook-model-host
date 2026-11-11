# Host model mở trên Google Colab: notebook dựng sẵn

[English](README.md) · **Tiếng Việt**

Quay lại [README chính](../../README.vi.md).

## Notebook

| Notebook | Model | Trọng số | Dung lượng tải | Tổng bộ nhớ GPU tối thiểu | Chạy thử |
|---|---|---|---|---|---|
| [colab_qwen3_0_6b.ipynb](colab_qwen3_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_0_6b.ipynb) | `Qwen/Qwen3-0.6B` | 1,4 GiB | ~2 GB | ~6 GB | ✓ T4, 10/11/2026 |
| [colab_qwen3_1_7b.ipynb](colab_qwen3_1_7b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_1_7b.ipynb) | `Qwen/Qwen3-1.7B` | 3,8 GiB | ~4 GB | ~9 GB | ✓ T4, 10/11/2026 |
| [colab_qwen3_4b.ipynb](colab_qwen3_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_4b.ipynb) | `Qwen/Qwen3-4B` | 7,5 GiB | ~8 GB | ~13 GB | ✓ T4, 10/11/2026 |
| [colab_qwen3_8b.ipynb](colab_qwen3_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_8b.ipynb) | `Qwen/Qwen3-8B` | 15,3 GiB | ~16 GB | ~21 GB | ✓ A100 40 GB, 10/11/2026 |
| [colab_qwen3_14b.ipynb](colab_qwen3_14b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_14b.ipynb) | `Qwen/Qwen3-14B` | 27,5 GiB | ~30 GB | ~35 GB | ✓ A100 40 GB, 10/11/2026 |
| [colab_qwen3_5_0_8b.ipynb](colab_qwen3_5_0_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_0_8b.ipynb) | `Qwen/Qwen3.5-0.8B` | 1,6 GiB | ~2 GB | ~10 GB | ✓ T4, 11/11/2026 |
| [colab_qwen3_5_2b.ipynb](colab_qwen3_5_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_2b.ipynb) | `Qwen/Qwen3.5-2B` | 4,2 GiB | ~5 GB | ~12 GB | ✓ T4, 11/11/2026 |
| [colab_qwen3_5_4b.ipynb](colab_qwen3_5_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_4b.ipynb) | `Qwen/Qwen3.5-4B` | 8,7 GiB | ~9 GB | ~17 GB | Chưa thử |
| [colab_qwen3_5_9b.ipynb](colab_qwen3_5_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_5_9b.ipynb) | `Qwen/Qwen3.5-9B` | 18,0 GiB | ~20 GB | ~28 GB | ✓ A100 40 GB 10/11/2026, A100 80 GB 11/11/2026 |
| [colab_qwen3_vl_2b_instruct.ipynb](colab_qwen3_vl_2b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_vl_2b_instruct.ipynb) | `Qwen/Qwen3-VL-2B-Instruct` | 4,0 GiB | ~5 GB | ~10 GB | Chưa thử |
| [colab_qwen3_vl_2b_thinking.ipynb](colab_qwen3_vl_2b_thinking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/colab/colab_qwen3_vl_2b_thinking.ipynb) | `Qwen/Qwen3-VL-2B-Thinking` | 4,0 GiB | ~5 GB | ~10 GB | Chưa thử |

**Qwen3.5-0.8B:** Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-2B:** Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-4B:** Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3.5-9B:** Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

**Qwen3-VL-2B-Instruct:** Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm context cho vừa (xem bảng bộ nhớ). **Không bao giờ suy nghĩ** (không có reasoning).

**Qwen3-VL-2B-Thinking:** Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm context cho vừa (xem bảng bộ nhớ). **Luôn suy nghĩ**: chat template không có công tắc nên `enable_thinking` không đổi gì; hãy để `max_tokens` đủ chỗ cho phần suy nghĩ.

## Thông số chung

vLLM 0.33.0, cloudflared 2026.10.7. `--max-model-len auto`, `--gpu-memory-utilization 0.9`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. Server không đặt tham số sinh, không ghi prompt hay câu trả lời.

## Bộ nhớ ước tính còn cho context

| GPU | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T4 | ~10 GiB | ~8 GiB | ~4 GiB | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ | ~7 GiB | ~4 GiB | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ | ~7 GiB | ~7 GiB |
| L4 | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ thiếu bộ nhớ | ~14 GiB | ~11 GiB | ~7 GiB | ✗ thiếu bộ nhớ | ~13 GiB | ~13 GiB |
| A100 40 GB | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB | ~6 GiB | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB | ~29 GiB | ~29 GiB |
| A100 80 GB | ~69 GiB | ~66 GiB | ~63 GiB | ~55 GiB | ~42 GiB | ~65 GiB | ~63 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB |
| H100 80 GB | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB | ~65 GiB | ~65 GiB |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### colab_qwen3_0_6b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 11,14 GiB = 104 256 token, 2,55 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 1,12 GiB, activation đỉnh 0,59 GiB, CUDA graph 0,17 GiB |
| Thời gian | bật server 2 phút 25 giây (trọng số 11 giây, torch.compile 28 giây, CUDA graph 18 giây) |
| Tốc độ | ~77 token/giây cho một request; ~570 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Prompt dài chậm trên T4 bất kể cỡ model: ~17 000–18 000 token mất ~50–55 giây (Qwen3-1.7B cũng vậy), nên prompt trên ~30 000 token cần stream |

### colab_qwen3_1_7b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 8,72 GiB = 81 632 token, 1,99 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 3,22 GiB, activation đỉnh 0,92 GiB, CUDA graph 0,17 GiB |
| Thời gian | bật server 2 phút 52 giây (trọng số 26 giây, torch.compile 29 giây, CUDA graph 20 giây) |
| Tốc độ | ~50 token/giây cho một request; ~330 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort="none"`); văn bản tiếng Việt ở float16 bình thường; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Suy nghĩ với câu đơn giản: 1 231 token trong 22 giây. Chất lượng có giới hạn: một câu trả lời lẫn vài chữ tiếng Thái |

### colab_qwen3_4b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1 |
| Context (`auto`) | 30 592 token (tự giảm từ 40 960 cho vừa) |
| KV cache | 4,2 GiB = 30 608 token: một request dài tối đa, hoặc nhiều request ngắn |
| Bộ nhớ GPU | trọng số 7,56 GiB, activation đỉnh 1,15 GiB, CUDA graph 0,22 GiB |
| Tốc độ | ~25 token/giây cho một request; ~170 token/giây tổng khi 8 request cùng lúc |
| Thời gian | bật server 3 phút 30 giây (trọng số 42 giây, torch.compile 37 giây, CUDA graph 30 giây) |
| Log trông như lỗi nhưng bình thường | `ERROR ... Cannot use FA version 2` (T4 chuyển sang Triton attention), FlashInfer sampler không dùng được, và nhiều cảnh báo nạp trọng số `bfloat16 into float16` |
| Đã kiểm qua URL công khai | không key → 401; `/invocations` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort="none"`); văn bản tiếng Việt ở float16 bình thường; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream. Suy nghĩ chậm trên T4 (câu đơn giản tốn 2 000 token, 81 giây): nên tắt hoặc dùng stream |

### colab_qwen3_8b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 40 960 token (tối đa của model) |
| KV cache | 18,47 GiB = 134 448 token, 3,28 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 15,27 GiB, activation đỉnh 1,64 GiB, CUDA graph 0,23 GiB |
| Tốc độ | ~75 token/giây cho một request |
| Thời gian | cài 2 phút, tải 43 giây, bật server 3 phút 35 giây (FlashInfer JIT ~1 phút 45 giây, torch.compile 31 giây); ~7 phút từ Run all tới khi có URL công khai |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ (`enable_thinking`, `reasoning_effort="none"`, `/no_think`); gọi hàm; JSON schema; completions; `/tokenize`; Anthropic `/v1/messages`; `cache_salt`; stream 124 giây / 9 000 token không bị cắt |

### colab_qwen3_14b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / bfloat16 / 1 |
| Context (`auto`) | 38 304 token (tự giảm từ 40 960 cho vừa) |
| KV cache | 5,85 GiB = 38 320 token: một request dài tối đa mỗi lúc, nhiều hơn nếu ngắn hơn |
| Bộ nhớ GPU | trọng số 27,52 GiB, activation đỉnh 2,0 GiB, CUDA graph 0,28 GiB |
| Thời gian | bật server 3 phút 47 giây (trọng số 12 giây, torch.compile 35 giây, FlashInfer JIT ~1 phút 50 giây, CUDA graph 16 giây) |
| Tốc độ | ~44 token/giây cho một request; ~287 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; bật/tắt suy nghĩ; gọi hàm; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; tìm đúng mã trong prompt 27 044 token (6 giây). Muốn đủ context 40 960 và nhiều request dài cùng lúc, dùng GPU 80 GB |

### colab_qwen3_5_0_8b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 9,11 GiB = 785 357 token, 3,00 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số 1,63 GiB, non-torch khác ~1 GiB, CUDA graph 0,20 + 0,12 GiB |
| Thời gian | bật server ~11 phút (trọng số 15 giây, torch.compile 27 giây, ~7,5 phút biên dịch kernel không ghi log, CUDA graph 47 giây) |
| Tốc độ | ~100 token/giây cho một request (có suy nghĩ); ~640 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định không suy nghĩ, suy nghĩ khi gửi `enable_thinking: true`; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (13,5 giây). Ảnh base64: đọc đúng chữ và hình chữ nhật đỏ, nhưng nói hình tròn xanh dương là xanh lá |
| Tham số sinh | một câu hỏi ngắn đúng 9/20 lần khi request không gửi tham số sinh (mặc định vLLM) và 20/20 khi gửi tham số Qwen khuyên: luôn gửi tham số |

### colab_qwen3_5_2b.ipynb

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 6,15 GiB = 530 734 token, 2,02 request dài tối đa cùng lúc (giống nhau ở cả hai lần chạy) |
| Bộ nhớ GPU | trọng số 4,18 GiB, CUDA graph 0,20 + 0,12 GiB |
| Thời gian | bật server 12 phút 53 giây và 11 phút 50 giây (hai lần cách nhau 8,5 tiếng), trong đó ~8 phút là biên dịch kernel không ghi log sau torch.compile |
| Tốc độ | ~300–332 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 12/13 phép thử, hai lần. không key/sai key → 401; `/invocations`, `/sleep` → 404; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (~16 giây). Phép thử suy nghĩ trượt chỉ vì model này đôi khi nghĩ quá ngân sách 4 000 token của phép thử: qua stream với 12 000 token thì trả lời đúng 3/3 lần (1 300–2 100 token). Ảnh base64: đúng chữ, hình chữ nhật bị gọi là hình vuông |

### colab_qwen3_5_4b.ipynb

Chưa thử.

### colab_qwen3_5_9b.ipynb

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

### colab_qwen3_vl_2b_instruct.ipynb

Chưa thử.

### colab_qwen3_vl_2b_thinking.ipynb

Chưa thử.

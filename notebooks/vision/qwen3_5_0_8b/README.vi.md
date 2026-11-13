# Qwen3.5-0.8B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model thị giác](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · văn bản, ảnh, video · trọng số 1,6 GiB (tải ~2 GB) · context 262 144 token · cần ít nhất ~10 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_5_0_8b.ipynb](colab_qwen3_5_0_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb) | ✓ T4, 11/11/2026 |
| Kaggle | [kaggle_qwen3_5_0_8b.ipynb](kaggle_qwen3_5_0_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb) | ✓ T4 x2, 11/11/2026 |

## Bản cập nhật chưa chạy thử

Các notebook này mới hơn bản đã chạy thử. Chúng chưa được chạy nên chưa có số đo. Dùng bản đã chạy thử, trừ khi bạn muốn thử bản cập nhật; khi bản cập nhật được chạy thử, nó thay cho bản đã chạy thử.

| Bản cập nhật (chưa chạy thử) | Thay cho |
|---|---|
| ⚠ [colab_qwen3_5_0_8b_update.ipynb](colab_qwen3_5_0_8b_update.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/colab_qwen3_5_0_8b_update.ipynb) | [colab_qwen3_5_0_8b.ipynb](colab_qwen3_5_0_8b.ipynb) |
| ⚠ [kaggle_qwen3_5_0_8b_update.ipynb](kaggle_qwen3_5_0_8b_update.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_0_8b/kaggle_qwen3_5_0_8b_update.ipynb) | [kaggle_qwen3_5_0_8b.ipynb](kaggle_qwen3_5_0_8b.ipynb) |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Ảnh đầu vào | `image_url` với URL `data:` base64 trong `/v1/chat/completions` | ✓ |
| Video đầu vào | `video_url` với URL `data:` base64 trong `/v1/chat/completions` | ✓ |
| Ảnh và video qua link http(s) | `image_url` / `video_url` `https://...` / `/v1/chat/completions` | ✗ chặn có chủ đích: server không tự tải link; hãy gửi base64 |
| Chế độ suy nghĩ, bật hoặc tắt theo từng request | `chat_template_kwargs.enable_thinking` trong `/v1/chat/completions`; phần suy nghĩ trả về trong `reasoning` | ✓ |
| Gọi hàm | `tools` trong `/v1/chat/completions`; lời gọi trả về trong `tool_calls` | ✓ |
| Dữ liệu có cấu trúc (JSON schema) | `response_format` trong `/v1/chat/completions` | ✓ |
| Thiết lập bộ xử lý ảnh và video theo từng request | `mm_processor_kwargs` in `/v1/chat/completions` (needs `--trust-request-mm-kwargs`) | ✗ chưa có trong notebook đã chạy thử: bản `_update` của nó bật cờ cho server, chưa chạy thử; vLLM 0.33 từ chối trường này nếu thiếu cờ |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3.5-0.8B |
|---|---|---|
| T4 | Google Colab | ~7 GiB |
| L4 | Google Colab | ~14 GiB |
| A100 40 GB | Google Colab | ~29 GiB |
| A100 80 GB | Google Colab | ~65 GiB |
| H100 80 GB | Google Colab | ~65 GiB |
| T4 x2 | Kaggle | ~18 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

### Kaggle

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

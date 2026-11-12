# Qwen3-4B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ngôn ngữ](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · văn bản · trọng số 7,5 GiB (tải ~8 GB) · context 40 960 token · cần ít nhất ~13 GB tổng bộ nhớ GPU

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_4b.ipynb](colab_qwen3_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_4b/colab_qwen3_4b.ipynb) | ✓ T4, 10/11/2026 |
| Kaggle | [kaggle_qwen3_4b.ipynb](kaggle_qwen3_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_4b/kaggle_qwen3_4b.ipynb) | ✓ T4 x2, 10/11/2026 |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Chế độ suy nghĩ, bật hoặc tắt theo từng request | `chat_template_kwargs.enable_thinking` trong `/v1/chat/completions`; phần suy nghĩ trả về trong `reasoning` | ✓ |
| Gọi hàm | `tools` trong `/v1/chat/completions`; lời gọi trả về trong `tool_calls` | ✓ |
| Dữ liệu có cấu trúc (JSON schema) | `response_format` trong `/v1/chat/completions` | ✓ |
| Hoàn thành văn bản | `prompt` / `/v1/completions` | ✓ |
| Context dài hơn bằng YaRN | rope scaling trong cấu hình model | ✗ chưa mở: notebook phục vụ context gốc (chưa có tuỳ chọn rope scaling) |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-4B |
|---|---|---|
| T4 | Google Colab | ~4 GiB |
| L4 | Google Colab | ~11 GiB |
| A100 40 GB | Google Colab | ~27 GiB |
| A100 80 GB | Google Colab | ~63 GiB |
| H100 80 GB | Google Colab | ~62 GiB |
| T4 x2 | Kaggle | ~16 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

### Kaggle

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

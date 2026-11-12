# Qwen3-8B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ngôn ngữ](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · văn bản · trọng số 15,3 GiB (tải ~16 GB) · context 40 960 token · cần ít nhất ~21 GB tổng bộ nhớ GPU

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_8b.ipynb](colab_qwen3_8b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_8b/colab_qwen3_8b.ipynb) | ✓ A100 40 GB, 10/11/2026 |
| Kaggle | [kaggle_qwen3_8b.ipynb](kaggle_qwen3_8b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_8b/kaggle_qwen3_8b.ipynb) | ✓ T4 x2, 10/11/2026 |

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

| GPU | Nền tảng | Qwen3-8B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~3 GiB |
| A100 40 GB | Google Colab | ~19 GiB |
| A100 80 GB | Google Colab | ~55 GiB |
| H100 80 GB | Google Colab | ~54 GiB |
| T4 x2 | Kaggle | ~8 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

### Kaggle

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

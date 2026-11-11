# Qwen3-0.6B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model văn bản](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · văn bản · trọng số 1,4 GiB (tải ~2 GB) · context 40 960 token · cần ít nhất ~6 GB tổng bộ nhớ GPU

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_0_6b.ipynb](colab_qwen3_0_6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/colab_qwen3_0_6b.ipynb) | ✓ T4, 10/11/2026 |
| Kaggle | [kaggle_qwen3_0_6b.ipynb](kaggle_qwen3_0_6b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb) | ✓ T4 x2, 10/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-0.6B |
|---|---|---|
| T4 | Google Colab | ~10 GiB |
| L4 | Google Colab | ~17 GiB |
| A100 40 GB | Google Colab | ~33 GiB |
| A100 80 GB | Google Colab | ~69 GiB |
| H100 80 GB | Google Colab | ~68 GiB |
| T4 x2 | Kaggle | ~22 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

### Kaggle

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

# Qwen3-1.7B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model văn bản](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · văn bản · trọng số 3,8 GiB (tải ~4 GB) · context 40 960 token · cần ít nhất ~9 GB tổng bộ nhớ GPU

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_1_7b.ipynb](colab_qwen3_1_7b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/colab_qwen3_1_7b.ipynb) | ✓ T4, 10/11/2026 |
| Kaggle | [kaggle_qwen3_1_7b.ipynb](kaggle_qwen3_1_7b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb) | ✓ T4 x2, 10/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-1.7B |
|---|---|---|
| T4 | Google Colab | ~8 GiB |
| L4 | Google Colab | ~14 GiB |
| A100 40 GB | Google Colab | ~30 GiB |
| A100 80 GB | Google Colab | ~66 GiB |
| H100 80 GB | Google Colab | ~66 GiB |
| T4 x2 | Kaggle | ~19 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

### Kaggle

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

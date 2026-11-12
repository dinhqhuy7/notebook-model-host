# Qwen3-14B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ngôn ngữ](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) · văn bản · trọng số 27,5 GiB (tải ~30 GB) · context 40 960 token · cần ít nhất ~35 GB tổng bộ nhớ GPU

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_14b.ipynb](colab_qwen3_14b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/language/qwen3_14b/colab_qwen3_14b.ipynb) | ✓ A100 40 GB, 10/11/2026 |
| Kaggle | — | cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-14B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ✗ thiếu bộ nhớ |
| A100 40 GB | Google Colab | ~6 GiB |
| A100 80 GB | Google Colab | ~42 GiB |
| H100 80 GB | Google Colab | ~42 GiB |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

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

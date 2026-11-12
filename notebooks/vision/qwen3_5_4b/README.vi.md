# Qwen3.5-4B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model thị giác](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · văn bản, ảnh, video · trọng số 8,7 GiB (tải ~9 GB) · context 262 144 token · cần ít nhất ~17 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video). Attention lai (24 lớp linear + 8 lớp full): KV cache chỉ ~32 KiB mỗi token, nên context gốc 262 144 token vừa GPU 40 GB. Trên T4 model chạy `float16` thay vì `bfloat16` gốc: đã thử trên Kaggle 2 × T4. Trên T4 lần khởi động đầu mất ~11–12 phút (biên dịch kernel): bước 5 không bị treo.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_5_4b.ipynb](colab_qwen3_5_4b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_4b/colab_qwen3_5_4b.ipynb) | ✓ L4, 11/11/2026 |
| Kaggle | [kaggle_qwen3_5_4b.ipynb](kaggle_qwen3_5_4b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb) | ✓ T4 x2, 11/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3.5-4B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~7 GiB |
| A100 40 GB | Google Colab | ~22 GiB |
| A100 80 GB | Google Colab | ~58 GiB |
| H100 80 GB | Google Colab | ~58 GiB |
| T4 x2 | Kaggle | ~11 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

| Mục | Kết quả |
|---|---|
| GPU | 1 × NVIDIA L4 (22,5 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / bfloat16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 9,19 GiB = 297 061 token, 1,13 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số + non-torch 8,73 GiB, activation đỉnh 1,91 GiB, CUDA graph 0,27 + 0,14 GiB |
| Thời gian | bật server 5 phút 53 giây (trọng số 7 giây, torch.compile 32 giây, profiling/warmup ban đầu 94 giây, FlashInfer JIT ~1 phút 50 giây, CUDA graph 24 giây) |
| Tốc độ | ~29 token/giây cho một request (có suy nghĩ); ~194 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 13/13 phép thử. không key/sai key → 401; `/invocations`, `/sleep` → 404; mặc định có suy nghĩ (911 token, 32 giây); tắt suy nghĩ bằng cả hai cách; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; 8 request song song; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (3,7 giây). Ảnh base64: đúng chữ và cả hai màu, hình chữ nhật bị gọi là hình vuông |

### Kaggle

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

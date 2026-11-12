# Qwen3-VL-2B-Instruct: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model thị giác](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · văn bản, ảnh, video · trọng số 4,0 GiB (tải ~5 GB) · context 262 144 token · cần ít nhất ~10 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm context cho vừa (xem bảng bộ nhớ). **Không bao giờ suy nghĩ** (không có reasoning).

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_vl_2b_instruct.ipynb](colab_qwen3_vl_2b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb) | ✓ T4 & A100 40 GB, 11/11/2026 |
| Kaggle | [kaggle_qwen3_vl_2b_instruct.ipynb](kaggle_qwen3_vl_2b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb) | ✓ T4 x2, 11/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-VL-2B-Instruct |
|---|---|---|
| T4 | Google Colab | ~7 GiB |
| L4 | Google Colab | ~13 GiB |
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
| GPU | 1 × Tesla T4 (15 GB) và 1 × NVIDIA A100-SXM4-40GB, driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 (T4), bfloat16 (A100) / 1 |
| Context (`auto`) | T4: 61 808 token; A100: 262 144 token (tối đa của model) |
| KV cache | T4: 6,6 GiB = 61 823 token, 1,00×; A100: 29,03 GiB = 271 808 token, 1,04× |
| Bộ nhớ GPU | T4: trọng số + non-torch 4,38 GiB, activation đỉnh 2,12 GiB, CUDA graph 0,30 + 0,18 GiB; A100: 4,37 GiB, 2,15 GiB, 0,30 + 0,20 GiB |
| Thời gian | T4: bật server 4 phút 5 giây; A100: 3 phút 47 giây (trọng số 1,3 giây, warmup đa phương thức 14,5 giây, torch.compile 40,9 giây, FlashInfer JIT lần đầu ~1 phút 46 giây, CUDA graph ~11 giây) |
| Tốc độ | T4: ~44 token/s một request, ~318–335 token/s với 8; A100: ~216 một request, ~1 165–1 365 với 8 |
| Đã kiểm qua URL công khai | 26/26 trên cả hai GPU: không có reasoning; gọi hàm qua `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; 8 request song song; ảnh base64 và nhiều ảnh; OCR tài liệu/bảng sang JSON; grounding 2D; đếm và suy luận không gian; hiểu biểu đồ; visual coding; đa lượt có ảnh; tool call từ nội dung ảnh; Responses API với ảnh; OCR/thứ tự/timestamp gần đúng và stream video; trộn ảnh + video; link media `http(s)` bị từ chối; tìm đúng mã trong prompt 18 044 token (T4: 56,6 giây khi chưa cache; A100: 0,9 giây ở lần thử đầu). Trên A100, lần xác nhận cuối định vị đúng mốc video giây 2 và trộn ảnh + video qua hai lần; trên T4, hai phép chất lượng này thay đổi giữa các lần |

### Kaggle

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel) |
| Context (`auto`) | 170 256 token (giảm từ tối đa 262 144 của model) |
| KV cache | 18,2 GiB tổng (9,09 GiB mỗi GPU) = 170 272 token, 1,00 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số + non-torch 2,29 GiB, activation đỉnh 1,72 GiB, CUDA graph 0,30 + 0,18 GiB |
| Thời gian | bật server 3 phút 39 giây (trọng số 8 giây, warmup đa phương thức 16 giây, torch.compile 45 giây, CUDA graph ~23 giây) |
| Tốc độ | ~71 token/giây cho một request; ~498 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 12/12 phép thử chuẩn: không có reasoning; gọi hàm qua `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; ảnh base64; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 18 000 token (29,5 giây). Phép thử mở rộng qua: nhiều ảnh, OCR tài liệu sang JSON, grounding 2D, không gian, visual coding, đa lượt có ảnh, tool call từ nội dung ảnh, Responses API với văn bản và ảnh, OCR/thứ tự/timestamp video, trộn ảnh + video và stream video; link video `http(s)` bị từ chối. Câu hỏi video quá ngắn nhầm màu cuối, prompt đủ ba đoạn trả đúng |

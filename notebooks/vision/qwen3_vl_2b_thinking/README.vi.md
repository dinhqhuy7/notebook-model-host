# Qwen3-VL-2B-Thinking: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model thị giác](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) · văn bản, ảnh, video · trọng số 4,0 GiB (tải ~5 GB) · context 262 144 token · cần ít nhất ~10 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 112 KiB mỗi token, nên đủ context 262 144 token cần ~28 GiB KV cache và chỉ vừa GPU cỡ A100; GPU nhỏ hơn thì vLLM tự giảm context cho vừa (xem bảng bộ nhớ). **Luôn suy nghĩ**: chat template không có công tắc nên `enable_thinking` không đổi gì; hãy để `max_tokens` đủ chỗ cho phần suy nghĩ.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_vl_2b_thinking.ipynb](colab_qwen3_vl_2b_thinking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb) | ✓ T4, 11/11/2026 |
| Kaggle | [kaggle_qwen3_vl_2b_thinking.ipynb](kaggle_qwen3_vl_2b_thinking.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb) | ✓ T4 x2, 11/11/2026 |

## Chức năng

Mọi thứ model làm được, và notebook có host hay không. Đã chạy thử hay chưa nằm ở kết quả chạy thử bên dưới.

| Chức năng | Cách gọi | Host |
|---|---|---|
| Ảnh đầu vào | `image_url` với URL `data:` base64 trong `/v1/chat/completions` | ✓ |
| Video đầu vào | `video_url` với URL `data:` base64 trong `/v1/chat/completions` | ✓ |
| Ảnh và video qua link http(s) | `image_url` / `video_url` `https://...` / `/v1/chat/completions` | ✗ chặn có chủ đích: server không tự tải link; hãy gửi base64 |
| Suy nghĩ (model luôn suy nghĩ) | `/v1/chat/completions`; phần suy nghĩ trả về trong `reasoning` | ✓ |
| Gọi hàm | `tools` trong `/v1/chat/completions`; lời gọi trả về trong `tool_calls` | ✓ |
| Dữ liệu có cấu trúc (JSON schema) | `response_format` trong `/v1/chat/completions` | ✓ |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3-VL-2B-Thinking |
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
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1, reasoning parser `qwen3` |
| Context (`auto`) | 61 808 token (giảm từ tối đa 262 144 của model) |
| KV cache | 6,6 GiB = 61 823 token, 1,00 request dài tối đa cùng lúc |
| Bộ nhớ GPU | trọng số + non-torch 4,38 GiB, activation đỉnh 2,12 GiB, CUDA graph 0,30 + 0,18 GiB |
| Thời gian | bật server 4 phút 2 giây (trọng số 17 giây, warmup đa phương thức 28 giây, torch.compile 45 giây, CUDA graph ~23 giây) |
| Tốc độ | ~42 token/giây cho một request; ~185–357 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 24/26 phép thử qua. Luôn suy nghĩ và reasoning được tách vào `message.reasoning`; gọi hàm qua `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; ảnh base64 và nhiều ảnh; OCR tài liệu/bảng sang JSON; đếm và suy luận không gian; hiểu biểu đồ; visual coding; tool call từ nội dung ảnh; Responses API với ảnh; OCR/thứ tự/timestamp và stream video; trộn ảnh + video; link media `http(s)` bị từ chối; tìm đúng mã trong prompt 18 046 token (17–75 giây tuỳ độ dài reasoning). Giới hạn: grounding 2D trả tọa độ pixel hoặc hết ngân sách reasoning trước bbox; multi-turn có ảnh đúng 2/4 lần, các lần khác nhầm mã nhìn thấy thành HTML hoặc lặp tới giới hạn token. Model cũng có lúc chép nguyên placeholder câu trả lời. Phép thử chức năng dùng sampling xác định |

### Kaggle

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), reasoning parser `qwen3` |
| Context (`auto`) | 170 256 token (giảm từ tối đa 262 144 của model) |
| KV cache | 18,2 GiB tổng (9,09 GiB mỗi GPU) = 170 272 token, 1,00 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số + non-torch 2,29 GiB, activation đỉnh 1,72 GiB, CUDA graph 0,30 + 0,18 GiB |
| Thời gian | bật server ~3 phút 35 giây, hai lần chạy như nhau (trọng số 7 giây, warmup đa phương thức 15 giây, torch.compile 41 giây, CUDA graph ~21 giây) |
| Tốc độ | ~71–74 token/giây cho một request; ~452 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 24/26 phép thử qua. Luôn suy nghĩ và reasoning được tách vào `message.reasoning`; gọi hàm qua `hermes`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; ảnh base64 và nhiều ảnh; OCR tài liệu/bảng sang JSON; đếm và suy luận không gian; hiểu biểu đồ; visual coding; tool call từ nội dung ảnh; Responses API với ảnh; OCR/thứ tự/timestamp và stream video; trộn ảnh + video; link media `http(s)` bị từ chối; tìm đúng mã trong prompt 18 046 token (38,9 giây). Giới hạn: grounding 2D trả sai tọa độ (`0,0,200,200` thay vì khoảng `100,200,350,533`); multi-turn có ảnh trả sai; model cũng có lúc chép nguyên placeholder câu trả lời. Phép thử chức năng dùng sampling xác định |

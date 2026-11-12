# Qwen2.5-VL-7B-Instruct: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model thị giác](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · văn bản, ảnh, video · trọng số 15,4 GiB (tải ~17 GB) · context 128 000 token · cần ít nhất ~23 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video), attention thuần: KV cache tốn 56 KiB mỗi token, nên đủ context 128 000 token chỉ cần ~7 GiB KV cache. **Không bao giờ suy nghĩ và không có function calling**: chat template không hỗ trợ tool nên server không bật tool parser (dùng JSON schema để lấy dữ liệu có cấu trúc). Không gửi tham số sinh thì model gần như greedy (`generation_config.json`). Tọa độ grounding dự kiến là pixel tuyệt đối (theo tài liệu Qwen; chưa thử ở đây). Trên T4 model chạy `float16` thay vì `bfloat16` gốc; chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen2_5_vl_7b_instruct.ipynb](colab_qwen2_5_vl_7b_instruct.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen2_5_vl_7b_instruct.ipynb](kaggle_qwen2_5_vl_7b_instruct.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision/qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb) | Chưa thử |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen2.5-VL-7B-Instruct |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~2 GiB |
| A100 40 GB | Google Colab | ~18 GiB |
| A100 80 GB | Google Colab | ~54 GiB |
| H100 80 GB | Google Colab | ~53 GiB |
| T4 x2 | Kaggle | ~7 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

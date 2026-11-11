# Qwen2.5-Omni-3B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model đa phương thức đầy đủ](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) · văn bản, ảnh, âm thanh, video (trả lời bằng văn bản và giọng nói) · trọng số 11,2 GiB (tải ~12 GB) · context 32 768 token · cần ít nhất ~20 GB tổng bộ nhớ GPU

Nghe và nói được: vào là văn bản, ảnh, âm thanh, video; ra là văn bản và giọng nói. Chạy trên vLLM-Omni 0.32.0 cùng vLLM 0.32.0 (các notebook khác dùng vLLM 0.33.0), qua ba giai đoạn đều được nạp: 11,15 GiB trọng số, nên **một T4 đơn không đủ** (một L4 hoặc A100, hoặc 2 × T4 của Kaggle). Muốn nhận giọng nói: `"modalities": ["text", "audio"]` (WAV base64 ở lựa chọn thứ hai), chọn giọng bằng `"speaker"` (`Chelsie` hoặc `Ethan`); câu trả lời có âm thanh không stream được nên giữ ngắn (tunnel cắt request sau 100 giây). Mỗi giai đoạn mặc định chỉ nhận một request mỗi lần. **Không có function calling và không có chế độ suy nghĩ.** Giấy phép: Qwen Research License, không phải Apache 2.0. `float16` trên T4 (Kaggle) chưa thử với model này.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen2_5_omni_3b.ipynb](colab_qwen2_5_omni_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/omni/qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb) | Chưa thử |
| Kaggle | [kaggle_qwen2_5_omni_3b.ipynb](kaggle_qwen2_5_omni_3b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/omni/qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb) | Chưa thử |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen2.5-Omni-3B |
|---|---|---|
| T4 | Google Colab | ✗ thiếu bộ nhớ |
| L4 | Google Colab | ~4 GiB |
| A100 40 GB | Google Colab | ~20 GiB |
| A100 80 GB | Google Colab | ~56 GiB |
| H100 80 GB | Google Colab | ~56 GiB |
| T4 x2 | Kaggle | ~9 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

Chưa thử.

### Kaggle

Chưa thử.

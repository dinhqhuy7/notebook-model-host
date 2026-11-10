# Notebook

[English](README.md) · **Tiếng Việt**

Quay lại [README chính](../README.vi.md).

Mỗi notebook host một model trên một nền tảng. Chọn model, rồi chọn nền tảng mà tài khoản của bạn có GPU đủ bộ nhớ. Bảng GPU và kết quả chạy thử theo nền tảng: [Google Colab](colab/README.vi.md), [Kaggle](kaggle/README.vi.md). Chọn GPU trước khi Run all; GPU cần ít nhất "Tổng bộ nhớ GPU tối thiểu" của model.

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | văn bản | 1,4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](colab/colab_qwen3_0_6b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_0_6b.ipynb](kaggle/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| `Qwen/Qwen3-1.7B` | văn bản | 3,8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](colab/colab_qwen3_1_7b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_1_7b.ipynb](kaggle/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| `Qwen/Qwen3-4B` | văn bản | 7,5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](colab/colab_qwen3_4b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_4b.ipynb](kaggle/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| `Qwen/Qwen3-8B` | văn bản | 15,3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](colab/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | [kaggle_qwen3_8b.ipynb](kaggle/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| `Qwen/Qwen3-14B` | văn bản | 27,5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](colab/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | — (cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB) |
| `Qwen/Qwen3.5-0.8B` | văn bản, ảnh, video | 1,6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](colab/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_5_0_8b.ipynb](kaggle/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| `Qwen/Qwen3.5-2B` | văn bản, ảnh, video | 4,2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](colab/colab_qwen3_5_2b.ipynb)<br>Chưa thử | [kaggle_qwen3_5_2b.ipynb](kaggle/kaggle_qwen3_5_2b.ipynb)<br>Chưa thử |
| `Qwen/Qwen3.5-9B` | văn bản, ảnh, video | 18,0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](colab/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 10/11/2026, A100 80 GB 11/11/2026 | [kaggle_qwen3_5_9b.ipynb](kaggle/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 11/11/2026 |

# Notebook

[English](README.md) · **Tiếng Việt**

Quay lại [README chính](../README.vi.md).

Mỗi notebook host một model trên một nền tảng. Notebook gom theo loại model, mỗi model một thư mục kèm notebook, bảng GPU và kết quả chạy thử. Chọn model, rồi chọn nền tảng mà tài khoản của bạn có GPU đủ bộ nhớ; chọn GPU trước khi Run all (ít nhất "Tổng bộ nhớ GPU tối thiểu" của model). Tốc độ trong kết quả chạy thử là của GPU dùng lúc đó, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

## [Model ngôn ngữ](language/README.vi.md)

Model đọc và viết văn bản: trò chuyện, có thể suy nghĩ, gọi hàm, viết và hoàn thành code. Vào văn bản, ra văn bản.

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B](language/qwen3_0_6b/README.vi.md) | văn bản | 1,4 GiB | 40 960 | ~6 GB | [colab_qwen3_0_6b.ipynb](language/qwen3_0_6b/colab_qwen3_0_6b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_0_6b.ipynb](language/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-1.7B](language/qwen3_1_7b/README.vi.md) | văn bản | 3,8 GiB | 40 960 | ~9 GB | [colab_qwen3_1_7b.ipynb](language/qwen3_1_7b/colab_qwen3_1_7b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_1_7b.ipynb](language/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-4B](language/qwen3_4b/README.vi.md) | văn bản | 7,5 GiB | 40 960 | ~13 GB | [colab_qwen3_4b.ipynb](language/qwen3_4b/colab_qwen3_4b.ipynb)<br>✓ T4, 10/11/2026 | [kaggle_qwen3_4b.ipynb](language/qwen3_4b/kaggle_qwen3_4b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-8B](language/qwen3_8b/README.vi.md) | văn bản | 15,3 GiB | 40 960 | ~21 GB | [colab_qwen3_8b.ipynb](language/qwen3_8b/colab_qwen3_8b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | [kaggle_qwen3_8b.ipynb](language/qwen3_8b/kaggle_qwen3_8b.ipynb)<br>✓ T4 x2, 10/11/2026 |
| [Qwen3-14B](language/qwen3_14b/README.vi.md) | văn bản | 27,5 GiB | 40 960 | ~35 GB | [colab_qwen3_14b.ipynb](language/qwen3_14b/colab_qwen3_14b.ipynb)<br>✓ A100 40 GB, 10/11/2026 | — (cần ~35 GB bộ nhớ GPU; Kaggle 2 × T4 chỉ có tổng 30 GB) |
| [Qwen2.5-Coder-3B-Instruct](language/qwen2_5_coder_3b_instruct/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b_instruct.ipynb](language/qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b_instruct.ipynb](language/qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb)<br>Chưa thử |
| [Qwen2.5-Coder-3B](language/qwen2_5_coder_3b/README.vi.md) | văn bản | 5,8 GiB | 32 768 | ~11 GB | [colab_qwen2_5_coder_3b.ipynb](language/qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb)<br>Chưa thử | [kaggle_qwen2_5_coder_3b.ipynb](language/qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb)<br>Chưa thử |

## [Model truy hồi](retrieval/README.vi.md)

Model chấm điểm hoặc nhúng văn bản để tìm kiếm: xếp hạng lại danh sách tài liệu theo câu hỏi, và biến văn bản thành vector. Vào văn bản, ra điểm số hoặc vector; không sinh văn bản.

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3-Reranker-0.6B](retrieval/qwen3_reranker_0_6b/README.vi.md) | văn bản (câu hỏi và các tài liệu cần xếp hạng) | 1,1 GiB | 40 960 | ~6 GB | [colab_qwen3_reranker_0_6b.ipynb](retrieval/qwen3_reranker_0_6b/colab_qwen3_reranker_0_6b.ipynb)<br>✓ T4, 13/11/2026 | [kaggle_qwen3_reranker_0_6b.ipynb](retrieval/qwen3_reranker_0_6b/kaggle_qwen3_reranker_0_6b.ipynb)<br>Chưa thử |
| [Qwen3-Reranker-4B](retrieval/qwen3_reranker_4b/README.vi.md) | văn bản (câu hỏi và các tài liệu cần xếp hạng) | 7,5 GiB | 40 960 | ~13 GB | [colab_qwen3_reranker_4b.ipynb](retrieval/qwen3_reranker_4b/colab_qwen3_reranker_4b.ipynb)<br>Chưa thử | [kaggle_qwen3_reranker_4b.ipynb](retrieval/qwen3_reranker_4b/kaggle_qwen3_reranker_4b.ipynb)<br>Chưa thử |

## [Model thị giác](vision/README.vi.md)

Model nhìn được: đọc ảnh và video cùng với văn bản. Server không tự tải media: hãy gửi ảnh và video dạng `data:` base64.

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen3.5-0.8B](vision/qwen3_5_0_8b/README.vi.md) | văn bản, ảnh, video | 1,6 GiB | 262 144 | ~10 GB | [colab_qwen3_5_0_8b.ipynb](vision/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_5_0_8b.ipynb](vision/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-2B](vision/qwen3_5_2b/README.vi.md) | văn bản, ảnh, video | 4,2 GiB | 262 144 | ~12 GB | [colab_qwen3_5_2b.ipynb](vision/qwen3_5_2b/colab_qwen3_5_2b.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_5_2b.ipynb](vision/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-4B](vision/qwen3_5_4b/README.vi.md) | văn bản, ảnh, video | 8,7 GiB | 262 144 | ~17 GB | [colab_qwen3_5_4b.ipynb](vision/qwen3_5_4b/colab_qwen3_5_4b.ipynb)<br>✓ L4, 11/11/2026 | [kaggle_qwen3_5_4b.ipynb](vision/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3.5-9B](vision/qwen3_5_9b/README.vi.md) | văn bản, ảnh, video | 18,0 GiB | 262 144 | ~28 GB | [colab_qwen3_5_9b.ipynb](vision/qwen3_5_9b/colab_qwen3_5_9b.ipynb)<br>✓ A100 40 GB 10/11/2026, A100 80 GB 11/11/2026 | [kaggle_qwen3_5_9b.ipynb](vision/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3-VL-2B-Instruct](vision/qwen3_vl_2b_instruct/README.vi.md) | văn bản, ảnh, video | 4,0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_instruct.ipynb](vision/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 & A100 40 GB, 11/11/2026 | [kaggle_qwen3_vl_2b_instruct.ipynb](vision/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen3-VL-2B-Thinking](vision/qwen3_vl_2b_thinking/README.vi.md) | văn bản, ảnh, video | 4,0 GiB | 262 144 | ~10 GB | [colab_qwen3_vl_2b_thinking.ipynb](vision/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb)<br>✓ T4, 11/11/2026 | [kaggle_qwen3_vl_2b_thinking.ipynb](vision/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb)<br>✓ T4 x2, 11/11/2026 |
| [Qwen2.5-VL-7B-Instruct](vision/qwen2_5_vl_7b_instruct/README.vi.md) | văn bản, ảnh, video | 15,4 GiB | 128 000 | ~23 GB | [colab_qwen2_5_vl_7b_instruct.ipynb](vision/qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb)<br>Chưa thử | [kaggle_qwen2_5_vl_7b_instruct.ipynb](vision/qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb)<br>Chưa thử |

## [Model đa phương thức](multimodal/README.vi.md)

Model nhận nhiều loại đầu vào và trả lời bằng nhiều dạng: vào là văn bản, ảnh, âm thanh, video; ra là văn bản và giọng nói. Chạy trên vLLM-Omni (phần mở rộng của vLLM) qua nhiều giai đoạn nên cần nhiều bộ nhớ GPU hơn kích thước model. Server không tự tải media: hãy gửi ảnh, âm thanh, video dạng `data:` base64.

| Model | Đầu vào | Trọng số | Context | Tổng bộ nhớ GPU tối thiểu | Google Colab | Kaggle |
|---|---|---|---|---|---|---|
| [Qwen2.5-Omni-3B](multimodal/qwen2_5_omni_3b/README.vi.md) | văn bản, ảnh, âm thanh, video (trả lời bằng văn bản và giọng nói) | 11,2 GiB | 32 768 | ~20 GB | [colab_qwen2_5_omni_3b.ipynb](multimodal/qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb)<br>Chưa thử | [kaggle_qwen2_5_omni_3b.ipynb](multimodal/qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb)<br>Chưa thử |

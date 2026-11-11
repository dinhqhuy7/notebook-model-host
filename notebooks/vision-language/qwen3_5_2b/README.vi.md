# Qwen3.5-2B: host trên Colab hoặc Kaggle

[English](README.md) · **Tiếng Việt**

Quay lại [Model ảnh và video](../README.vi.md) · [mọi notebook](../../README.vi.md) · [README chính](../../../README.vi.md).

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) · văn bản, ảnh, video · trọng số 4,2 GiB (tải ~5 GB) · context 262 144 token · cần ít nhất ~12 GB tổng bộ nhớ GPU

Đa phương thức (văn bản, ảnh, video), attention lai: KV cache chỉ ~12 KiB mỗi token, nên đủ context 262 144 token trên mọi GPU trong bảng. **Mặc định không suy nghĩ**: gửi `chat_template_kwargs: {"enable_thinking": true}` để bật. Trên T4 model chạy `float16` thay vì `bfloat16` gốc, và lần khởi động đầu mất ~11 phút (biên dịch kernel): bước 5 không bị treo.

## Notebook

| Nền tảng | Notebook | Chạy thử |
|---|---|---|
| Google Colab | [colab_qwen3_5_2b.ipynb](colab_qwen3_5_2b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/colab_qwen3_5_2b.ipynb) | ✓ T4, 11/11/2026 |
| Kaggle | [kaggle_qwen3_5_2b.ipynb](kaggle_qwen3_5_2b.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb) | ✓ T4 x2, 11/11/2026 |

## Bộ nhớ ước tính còn cho context

| GPU | Nền tảng | Qwen3.5-2B |
|---|---|---|
| T4 | Google Colab | ~4 GiB |
| L4 | Google Colab | ~11 GiB |
| A100 40 GB | Google Colab | ~27 GiB |
| A100 80 GB | Google Colab | ~63 GiB |
| H100 80 GB | Google Colab | ~62 GiB |
| T4 x2 | Kaggle | ~16 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ |

## Kết quả chạy thử

Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.

### Google Colab

| Mục | Kết quả |
|---|---|
| GPU | 1 × Tesla T4 (15 GB), driver 580.82.07 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 1, `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 6,15 GiB = 530 734 token, 2,02 request dài tối đa cùng lúc (giống nhau ở cả hai lần chạy) |
| Bộ nhớ GPU | trọng số 4,18 GiB, CUDA graph 0,20 + 0,12 GiB |
| Thời gian | bật server 12 phút 53 giây và 11 phút 50 giây (hai lần cách nhau 8,5 tiếng), trong đó ~8 phút là biên dịch kernel không ghi log sau torch.compile |
| Tốc độ | ~300–332 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | 12/13 phép thử, hai lần. không key/sai key → 401; `/invocations`, `/sleep` → 404; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (~16 giây). Phép thử suy nghĩ trượt chỉ vì model này đôi khi nghĩ quá ngân sách 4 000 token của phép thử: qua stream với 12 000 token thì trả lời đúng 3/3 lần (1 300–2 100 token). Ảnh base64: đúng chữ, hình chữ nhật bị gọi là hình vuông |

### Kaggle

| Mục | Kết quả |
|---|---|
| GPU | 2 × Tesla T4 (15 GB), driver 580.178.04, Python 3.13 |
| vLLM / dtype / số GPU | 0.33.0 / float16 / 2 (tensor parallel), `--max-num-seqs 128` |
| Context (`auto`) | 262 144 token (tối đa của model) |
| KV cache | 17,3 GiB (8,64 GiB mỗi GPU) = 1 490 675 token, 5,69 request dài tối đa cùng lúc |
| Bộ nhớ GPU (mỗi GPU) | trọng số 2,12 GiB, non-torch khác ~0,9 GiB, CUDA graph 0,22 + 0,12 GiB |
| Thời gian | bật server ~10 phút (trọng số 7 giây, torch.compile 25 giây, ~6,6 phút biên dịch kernel không ghi log, CUDA graph 32 giây) |
| Tốc độ | ~86 token/giây cho một request (có suy nghĩ); ~272 token/giây tổng khi 8 request cùng lúc |
| Đã kiểm qua URL công khai | không key/sai key → 401; `/invocations`, `/sleep` → 404; suy nghĩ khi gửi `enable_thinking: true`; gọi hàm qua `qwen3_coder`; JSON schema; Anthropic `/v1/messages`; `/tokenize`; stream; link ảnh `http(s)` bị từ chối; tìm đúng mã trong prompt 16 000 token (9,6 giây). Ảnh base64: đúng chữ, hình chữ nhật bị gọi là hình vuông |
| Tham số sinh | một câu hỏi ngắn, 20 lần mỗi cách: temperature 0 đúng 20/20; tham số Qwen chung (0.7/0.8, top_k 20, presence 1.5) 17/20; tham số Qwen cho văn bản của 0.8B/2B (1.0/1.0, top_k 20, presence 2.0) 9/20. Câu trả lời ngắn cần chính xác thì dùng temperature thấp |

# Notebook dựng sẵn để host model mở trên Google Colab và Kaggle

[English](README.md) · **Tiếng Việt**

Notebook dựng sẵn chạy model mở bằng [vLLM](https://github.com/vllm-project/vllm) trên GPU Colab hoặc
Kaggle và mở ra Internet thành **API tương thích OpenAI có khoá**. Không cần tự viết code host: mở một
notebook bên dưới và chạy. Notebook chỉ host model: prompt, tham số và câu trả lời đều ở phía người
gọi.

```
người dùng ──HTTPS──▶ xxxx.trycloudflare.com ──▶ proxy (bắt key) ──▶ vLLM ──▶ GPU
                      (Cloudflare Tunnel)        └────── trong máy notebook ──────┘
```

## Model có sẵn

### Model văn bản

| Model | Trọng số | Colab | Kaggle |
|---|---|---|---|
| [Qwen3-0.6B](notebooks/text/qwen3_0_6b/README.vi.md) | 1,4 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/colab_qwen3_0_6b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb) |
| [Qwen3-1.7B](notebooks/text/qwen3_1_7b/README.vi.md) | 3,8 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/colab_qwen3_1_7b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb) |
| [Qwen3-4B](notebooks/text/qwen3_4b/README.vi.md) | 7,5 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_4b/colab_qwen3_4b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_4b/kaggle_qwen3_4b.ipynb) |
| [Qwen3-8B](notebooks/text/qwen3_8b/README.vi.md) | 15,3 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/colab_qwen3_8b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/kaggle_qwen3_8b.ipynb) |
| [Qwen3-14B](notebooks/text/qwen3_14b/README.vi.md) | 27,5 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_14b/colab_qwen3_14b.ipynb) | — (cần ~35 GB bộ nhớ GPU) |

### Model ảnh và video

| Model | Trọng số | Colab | Kaggle |
|---|---|---|---|
| [Qwen3.5-0.8B](notebooks/vision-language/qwen3_5_0_8b/README.vi.md) (ảnh, video) | 1,6 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb) |
| [Qwen3.5-2B](notebooks/vision-language/qwen3_5_2b/README.vi.md) (ảnh, video) | 4,2 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/colab_qwen3_5_2b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb) |
| [Qwen3.5-4B](notebooks/vision-language/qwen3_5_4b/README.vi.md) (ảnh, video) | 8,7 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_4b/colab_qwen3_5_4b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb) |
| [Qwen3.5-9B](notebooks/vision-language/qwen3_5_9b/README.vi.md) (ảnh, video) | 18,0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/colab_qwen3_5_9b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb) |
| [Qwen3-VL-2B-Instruct](notebooks/vision-language/qwen3_vl_2b_instruct/README.vi.md) (ảnh, video) | 4,0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb) |
| [Qwen3-VL-2B-Thinking](notebooks/vision-language/qwen3_vl_2b_thinking/README.vi.md) (ảnh, video) | 4,0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb) |

Chọn GPU trước khi bấm **Run all** cho phù hợp nhu cầu; GPU phải đủ bộ nhớ cho model (✗ trong bảng
dưới là không đủ). Repo mới đo trên các GPU ghi "đo thật". Mỗi model được đo trên GPU tại thời điểm chạy
thử, thường là loại khiêm tốn (T4, 2 × T4), nên tốc độ ghi trong kết quả chưa phải mức tốt nhất: GPU mạnh
hơn sẽ trả lời nhanh hơn, và một số model chưa được đo trên GPU tốt nhất hiện có. Bộ nhớ ước tính còn cho
context:

**Model văn bản**

| GPU | Nền tảng | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|---|
| T4 (15 GB) | Colab | ~10 GiB (đo thật: 11,1 GiB) | ~8 GiB (đo thật: 8,7 GiB) | ~4 GiB (đo thật: 4,2 GiB) | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ |
| T4 x2 | Kaggle | ~22 GiB (đo thật: 23,5 GiB) | ~19 GiB (đo thật: 21,4 GiB) | ~16 GiB (đo thật: 16,8 GiB) | ~8 GiB (đo thật: 9,2 GiB) | ✗ thiếu bộ nhớ |
| L4 (22,5 GB) | Colab | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ thiếu bộ nhớ |
| A100 40 GB | Colab | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB (đo thật: 18,5 GiB) | ~6 GiB (đo thật: 5,85 GiB) |
| A100 / H100 80 GB | Colab | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

**Model ảnh và video**

| GPU | Nền tảng | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking |
|---|---|---|---|---|---|---|---|
| T4 (15 GB) | Colab | ~7 GiB (đo thật: 9,1 GiB) | ~4 GiB (đo thật: 6,2 GiB) | ✗ thiếu bộ nhớ | ✗ thiếu bộ nhớ | ~7 GiB (đo thật: 6,6 GiB) | ~7 GiB (đo thật: 6,6 GiB) |
| T4 x2 | Kaggle | ~18 GiB (đo thật: 20,4 GiB) | ~16 GiB (đo thật: 17,3 GiB) | ~11 GiB (đo thật: 11,9 GiB) | ~2 GiB (đo thật: 3,5 GiB) | ~18 GiB (đo thật: 18,2 GiB) | ~18 GiB (đo thật: 18,2 GiB) |
| L4 (22,5 GB) | Colab | ~14 GiB | ~11 GiB | ~7 GiB (đo thật: 9,2 GiB) | ✗ thiếu bộ nhớ | ~13 GiB | ~13 GiB |
| A100 40 GB | Colab | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB (đo thật: 13,2 GiB) | ~29 GiB (đo thật: 29,03 GiB) | ~29 GiB |
| A100 / H100 80 GB | Colab | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB (đo thật A100 80 GB: 49,3 GiB) | ~65 GiB | ~65 GiB |
| P100 | Kaggle | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ | ✗ vLLM không hỗ trợ |

Số ghi "đo thật" lấy từ lần chạy thật; còn lại là ước tính. GPU khác: bước 1 của notebook tự kiểm.
Qwen3.5-9B tốn KV cache ít hơn ~4,5 lần mỗi token (attention lai), nên 13 GiB đủ cho context gốc
262 144 token. Mọi notebook kèm trạng thái chạy thử: [notebooks/README.vi.md](notebooks/README.vi.md)
(theo loại: [văn bản](notebooks/text/README.vi.md), [ảnh và video](notebooks/vision-language/README.vi.md)).
Mỗi model có thư mục riêng chứa notebook, bảng GPU và kết quả chạy thử đầy đủ: mở từ tên model ở đó.

## Cách host model trên Google Colab hoặc Kaggle

**Colab:** mở notebook → Runtime → Change runtime type nếu muốn GPU khác → (nên làm) thêm secret
`LLM_API_KEY` ở 🔑 Secrets, bật Notebook access → **Run all**.

**Kaggle:** mở notebook → Settings: chọn **Accelerator**, bật **Internet** (tài khoản cần xác minh
số điện thoại) → (nên làm) Add-ons → Secrets: thêm `LLM_API_KEY`, tick gắn vào notebook →
**Run All**.

Lần đầu mất khoảng 7–15 phút. Ô bước 8 in **Base URL** và **API key**. Để tab mở trong lúc dùng;
xong thì chạy ô **Tắt** và trả GPU. Không đặt secret thì key đổi mỗi lần chạy.

## Gọi model đã host (API tương thích OpenAI)

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="<API key>")
reply = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "Xin chào"}],
    max_tokens=1024,
)
print(reply.choices[0].message.content)
```

Server mở `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (kiểu
Anthropic), `/v1/models`, `/tokenize`, `/detokenize`; hỗ trợ suy nghĩ (thinking), gọi hàm, JSON
schema và (Qwen3.5) ảnh, video. Ví dụ đầy đủ: [docs/api.vi.md](docs/api.vi.md).

## Bảo mật và dữ liệu

- **Chỉ một cửa vào:** vLLM và proxy nghe ở `127.0.0.1`; từ Internet chỉ vào được qua URL tunnel.
  Proxy bắt key ở mọi đường dẫn và chặn API quản trị, vì `--api-key` của vLLM không bảo vệ
  `/invocations`, `/sleep`...
- **Một key chung:** đổi `LLM_API_KEY` rồi chạy lại để thu hồi. Key nằm trong output notebook: xoá
  output trước khi chia sẻ; trên Kaggle giữ notebook Private.
- **Không lưu nội dung:** server không thêm system prompt, không đặt tham số sinh (request không
  gửi thì dùng mặc định của chính model nếu model có, xem
  [Khác nhau giữa các model](docs/api.vi.md#khác-nhau-giữa-các-model)), không ghi prompt hay câu trả lời vào log.
- **Server không tự tải gì:** Qwen3.5 chỉ nhận ảnh và video dạng URL `data:` base64 do người gọi gửi;
  link `http(s)` bị từ chối thay vì được server tải về.
- **Nội dung vẫn đi qua** Cloudflare (HTTPS được giải mã ở đó) và bộ nhớ máy notebook. Chỉ dùng
  server của người bạn tin.
- **Prefix cache dùng chung:** khi gửi dữ liệu nhạy cảm, thêm `cache_salt` để không chia cache với
  người khác ([chi tiết](docs/api.vi.md#tách-prefix-cache)).

## Giới hạn

- URL đổi mỗi lần chạy; server chỉ sống cùng phiên notebook.
- Request không stream bị cắt sau 100 giây (lỗi 524). Câu trả lời dài hoặc có suy nghĩ thì dùng
  `stream=true`, không bị cắt (đã thử: 124 giây).
- Prompt dài chậm trên T4 (~350 token/giây khi đọc prompt, bất kể cỡ model): trên 1 × T4 nên stream
  khi prompt trên ~30 000 token; Kaggle 2 × T4 đọc nhanh gấp đôi.
- Hợp cho cá nhân hoặc nhóm nhỏ, không phải dịch vụ công cộng.

## Điều khoản nền tảng

Colab và Kaggle giới hạn việc chạy dịch vụ web hay proxy tuỳ gói tài khoản (ví dụ
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Kaggle Terms](https://www.kaggle.com/terms)). Tự đọc điều khoản của gói bạn dùng trước khi mở API
ra Internet.

## Phát triển

```
notebooks/   gom theo loại model (text/, vision-language/), mỗi model một thư mục chứa notebook và
             README; sinh tự động, đừng sửa tay
notebooks.lock.json   notebook đã có kết quả chạy thử bị khoá: bộ sinh không bao giờ ghi đè
proxy/       proxy bắt key (được nhúng vào mọi notebook)
tools/       build_notebooks.py: khuôn notebook, danh sách model và nền tảng
tests/       test proxy và test khoá notebook, không cần GPU
docs/        hướng dẫn gọi API (api.md, api.vi.md)
```

```bash
pip install aiohttp
python -m unittest discover -s tests
python tools/build_notebooks.py
```

Thêm model: thêm một dòng vào `MODELS` trong `tools/build_notebooks.py` (`kind` quyết định thư mục loại,
xem `TYPES`), sinh lại, chạy thử trên GPU thật, ghi kết quả vào `TESTED` rồi khoá bằng
`python tools/build_notebooks.py --freeze <notebook>`. Thay đổi vào notebook đã khoá được ghi ra file riêng
`<notebook>_update.ipynb`, đánh dấu chưa chạy thử, đặt cạnh nó; `--promote` thay bản cũ sau khi chạy thử. Model không vừa một nền tảng thì đặt `platforms` và `unavailable`
(lý do). Model không phải Qwen3 có thể cần `--reasoning-parser` / `--tool-call-parser` khác.

## Giấy phép

Mã: [MIT](LICENSE). Model theo giấy phép riêng (Qwen3, Qwen3.5: Apache 2.0).

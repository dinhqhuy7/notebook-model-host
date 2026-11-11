# Gọi API

[English](api.md) · **Tiếng Việt**

Notebook in ra **Base URL** (`https://xxxx.trycloudflare.com/v1`), **API key** và tên **model**
(`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`,
`Qwen3.5-9B`, `Qwen3-VL-2B-Instruct`, `Qwen3-VL-2B-Thinking`, `Qwen2.5-VL-7B-Instruct`). Các ví
dụ dùng thư
viện `openai` (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="<API key>")
MODEL = "Qwen3-4B"
```

## Khác nhau giữa các model

| | Qwen3-0.6B, 1.7B, 4B, 8B, 14B | Qwen3.5-0.8B, 2B | Qwen3.5-4B, 9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct |
|---|---|---|---|---|---|---|
| Đầu vào | văn bản | văn bản, ảnh, video | văn bản, ảnh, video | văn bản, ảnh, video | văn bản, ảnh, video | văn bản, ảnh, video |
| Context | 40 960 token | tới 262 144 token | tới 262 144 token | tới 262 144 token (cần ~28 GiB KV cache: GPU nhỏ hơn được ít hơn) | như Instruct | tối đa 128 000 token (~7 GiB KV cache) |
| Suy nghĩ mặc định | bật | **tắt**: gửi `enable_thinking: true` để bật | bật | **không bao giờ** | **luôn luôn**, không có công tắc | **không bao giờ** |
| Mặc định khi request không gửi tham số sinh | `generation_config.json` của model (temperature 0.6, top_p 0.95, top_k 20) | mặc định của vLLM (temperature 1.0, không top_p/top_k): câu trả lời kém tin cậy (thử: đúng 9/20, so với 20/20 khi gửi tham số khuyên dùng), **luôn gửi tham số** | mặc định của vLLM (temperature 1.0, không top_p/top_k): câu trả lời kém tin cậy (thử: đúng 9/20, so với 20/20 khi gửi tham số khuyên dùng), **luôn gửi tham số** | `generation_config.json` của model (temperature 0.7, top_p 0.8, top_k 20) | `generation_config.json` của model (temperature 1.0, top_p 0.95, top_k 20) | `generation_config.json` của model (temperature 0.000001, repetition_penalty 1.05): gần như greedy |
| Khuyên dùng khi có suy nghĩ | `temperature=0.6, top_p=0.95, top_k=20` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | không có | `temperature=1.0, top_p=0.95, top_k=20` | không áp dụng |
| Khuyên dùng khi không suy nghĩ | `temperature=0.7, top_p=0.8, top_k=20` | văn bản: `temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0`; ảnh: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | ảnh: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`; văn bản: `temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0` | không có | Qwen không công bố; mặc định ở trên hợp với câu trả lời thực tế |
| `/think`, `/no_think` trong tin nhắn | có | không hỗ trợ: dùng `enable_thinking` | không hỗ trợ: dùng `enable_thinking` | không có | không có | không áp dụng |
| Gọi hàm | hỗ trợ (`hermes`) | hỗ trợ (`qwen3_coder`) | hỗ trợ (`qwen3_coder`) | hỗ trợ (`hermes`) | hỗ trợ (`hermes`) | **không hỗ trợ**: chat template không có tool, dùng JSON schema |
| Viết code | chưa thử | chưa thử | 9B: tắt suy nghĩ (khi bật, model lặp 30 000 token mà không viết ra code); 4B: chưa thử | chưa thử | chưa thử | chưa thử |

## Chat

```python
reply = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Trả lời ngắn gọn bằng tiếng Việt."},
        {"role": "user", "content": "Vì sao bầu trời màu xanh?"},
    ],
    max_tokens=1024,
)
print(reply.choices[0].message.content)
```

```bash
curl https://xxxx.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer <API key>" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "messages": [{"role": "user", "content": "Xin chào"}], "max_tokens": 1024}'
```

## Suy nghĩ và tham số sinh

Qwen3, Qwen3.5-4B và 9B mặc định suy nghĩ trước khi trả lời (Qwen3.5-0.8B và 2B thì không; Qwen3-VL-2B-Instruct và Qwen2.5-VL-7B-Instruct không bao giờ suy nghĩ, Qwen3-VL-2B-Thinking luôn suy nghĩ và không có công tắc). Phần suy nghĩ nằm ở `message.reasoning`, `content` chỉ
chứa câu trả lời. Server không đặt tham số nào: nên gửi tường minh (giá trị cho từng model ở mục
[Khác nhau giữa các model](#khác-nhau-giữa-các-model)). Ví dụ với Qwen3:

```python
# Có suy nghĩ: câu khó (toán, code, suy luận nhiều bước)
thinking = dict(temperature=0.6, top_p=0.95, extra_body={"top_k": 20})
# Không suy nghĩ: câu đơn giản, nhanh hơn nhiều
direct = dict(temperature=0.7, top_p=0.8,
              extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})

reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=2048, **thinking)
print(getattr(reply.choices[0].message, "reasoning", None))  # phần suy nghĩ
print(reply.choices[0].message.content)                      # câu trả lời
```

- **Luôn đặt `max_tokens`.** Không stream thì giữ dưới mức notebook gợi ý (giới hạn 100 giây);
  câu trả lời dài thì dùng stream (xem bên dưới).
- `content` rỗng: hết `max_tokens` khi model còn đang suy nghĩ. Tăng `max_tokens` hoặc tắt suy nghĩ.
- Không đặt `temperature=0` khi có suy nghĩ: model dễ lặp vô hạn.
- Model nhỏ (Qwen3.5-0.8B, 2B): câu trả lời ngắn cần chính xác thì dùng temperature thấp. Khi thử, temperature 0
  đúng 20/20 lần, còn bộ tham số khuyên cho văn bản chỉ đúng 9/20.
- Cách khác để tắt suy nghĩ: `reasoning_effort="none"`; riêng Qwen3 còn có `/no_think` ở cuối tin
  nhắn (`/think` bật lại cho lượt đó).
- Tham số khác của vLLM (`min_p`, `repetition_penalty`, `seed`, `stop`, `logprobs`...) gửi qua
  `extra_body`.

## Stream

Nên dùng cho câu trả lời dài và khi có suy nghĩ: token về dần trong lúc sinh, và request không bị
cắt ở 100 giây.

```python
stream = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=8000, stream=True)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Ảnh và video (Qwen3.5, Qwen3-VL và Qwen2.5-VL)

Server không bao giờ tự tải URL: gửi ảnh từ máy bạn dưới dạng URL `data:` base64 (link ảnh
`http(s)://` bị từ chối).

```python
import base64

with open("photo.jpg", "rb") as f:
    image = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

reply = client.chat.completions.create(
    model="Qwen3.5-9B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "Trong ảnh có gì?"},
    ]}],
    max_tokens=2048, temperature=0.7, top_p=0.8, presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.content)
```

Video làm tương tự với `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`.
Mỗi request tối đa 32 MiB, nên giữ ảnh và video nhỏ.

## Gọi hàm

Không dùng được với Qwen2.5-VL-7B-Instruct: chat template của model không hỗ trợ tool nên server không bật tool parser. Hãy dùng [JSON theo schema](#json-theo-schema).

```python
import json

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Lấy thời tiết hiện tại của một thành phố",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
messages = [{"role": "user", "content": "Đà Nẵng hôm nay thế nào?"}]

reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=1024)
call = reply.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)  # {"city": "Đà Nẵng"}

# Tự chạy hàm ở máy bạn, rồi gửi kết quả lại cho model
messages += [reply.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 29})}]
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=1024)
print(final.choices[0].message.content)
```

## JSON theo schema

```python
schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["tích cực", "tiêu cực", "trung tính"]}},
    "required": ["sentiment"],
}
reply = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Đánh giá: 'Giao hàng nhanh, đóng gói cẩn thận.'"}],
    max_tokens=256,
    response_format={"type": "json_schema", "json_schema": {"name": "review", "schema": schema}},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(json.loads(reply.choices[0].message.content))
```

## Completion và đếm token

```python
reply = client.completions.create(model=MODEL, prompt="Ngày xửa ngày xưa,", max_tokens=200)
print(reply.choices[0].text)
```

```bash
curl https://xxxx.trycloudflare.com/tokenize \
  -H "Authorization: Bearer <API key>" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "messages": [{"role": "user", "content": "Xin chào"}], "add_generation_prompt": true}'
# trả về {"count": <số token>, "max_model_len": <context tối đa>, "tokens": [...]}
```

## Anthropic SDK

Base URL **không có** `/v1`:

```python
import anthropic

client = anthropic.Anthropic(base_url="https://xxxx.trycloudflare.com", api_key="<API key>")
reply = client.messages.create(model="Qwen3-4B", max_tokens=1024,
                               messages=[{"role": "user", "content": "Xin chào"}])
print(reply.content[-1].text)
```

## Giao diện chat (Open WebUI, LibreChat...)

Thêm kết nối "OpenAI-compatible" với URL `https://xxxx.trycloudflare.com/v1` và API key; model tự
hiện ra. URL đổi mỗi lần chạy lại notebook.

## Tách prefix cache

Server dùng lại KV cache của các prompt có cùng phần đầu để chạy nhanh hơn. Khi nhiều người dùng
chung server, về lý thuyết có thể đoán phần đầu prompt của người khác qua thời gian phản hồi. Với dữ
liệu nhạy cảm, gửi kèm một chuỗi bí mật của riêng bạn; cache chỉ dùng chung giữa các request có cùng
chuỗi:

```python
reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024,
                                       extra_body={"cache_salt": "<chuỗi bí mật của bạn>"})
```

## Mã lỗi

| Mã | Nghĩa | Xử lý |
|---|---|---|
| 401 | Thiếu hoặc sai key | Gửi `Authorization: Bearer <key>` (hoặc `x-api-key`) |
| 404 | Đường dẫn không được mở | Xem danh sách API ở [README](../README.vi.md) |
| 400 | Request sai, thường là vượt context | Rút prompt hoặc giảm `max_tokens`; kiểm bằng `/tokenize` |
| 502 | vLLM đã tắt hoặc đang khởi động lại | Xem ô Theo dõi trong notebook |
| 524 | Request không stream chạy quá 100 giây | Dùng `stream=true`, hoặc giảm `max_tokens` |
| 530 | Tunnel đã tắt | Chạy lại notebook, dùng URL mới |

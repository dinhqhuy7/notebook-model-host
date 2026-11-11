# Calling the API

**English** · [Tiếng Việt](api.vi.md)

The notebook prints the **Base URL** (`https://xxxx.trycloudflare.com/v1`), the **API key** and the
**model** name (`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`,
`Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3-VL-2B-Instruct`, `Qwen3-VL-2B-Thinking`, `Qwen2.5-VL-7B-Instruct`, `Qwen2.5-Omni-3B`).
The examples use the `openai` library (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="<API key>")
MODEL = "Qwen3-4B"
```

## Model differences

| | Qwen3-0.6B, 1.7B, 4B, 8B, 14B | Qwen3.5-0.8B, 2B | Qwen3.5-4B, 9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct | Qwen2.5-Omni-3B |
|---|---|---|---|---|---|---|---|
| Input | text | text, images, video | text, images, video | text, images, video | text, images, video | text, images, video | text, images, audio, video; output text **and speech** |
| Context | 40 960 tokens | up to 262 144 tokens | up to 262 144 tokens | up to 262 144 tokens (needs ~28 GiB of KV cache: smaller GPUs get less) | same as Instruct | up to 128 000 tokens (~7 GiB of KV cache) | 32 768 tokens |
| Thinking by default | on | **off**: send `enable_thinking: true` to turn it on | on | **never** | **always**, no switch | **never** | **never** |
| Defaults when a request omits sampling parameters | the model's `generation_config.json` (temperature 0.6, top_p 0.95, top_k 20) | vLLM's defaults (temperature 1.0, no top_p/top_k): answers are unreliable (a test question was right 9/20 times vs 20/20 with the recommended values), **always send parameters** | vLLM's defaults (temperature 1.0, no top_p/top_k): answers are unreliable (a test question was right 9/20 times vs 20/20 with the recommended values), **always send parameters** | the model's `generation_config.json` (temperature 0.7, top_p 0.8, top_k 20) | the model's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) | the model's `generation_config.json` (temperature 0.000001, repetition_penalty 1.05): almost greedy | the defaults of its vLLM-Omni config (Thinker: temperature 0, repetition_penalty 1.1) |
| Recommended with thinking | `temperature=0.6, top_p=0.95, top_k=20` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | n/a | `temperature=1.0, top_p=0.95, top_k=20` | n/a | n/a |
| Recommended without thinking | `temperature=0.7, top_p=0.8, top_k=20` | text: `temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0`; images: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | images: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`; text: `temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0` | n/a | none published by Qwen; the defaults above suit factual answers | none published by Qwen; the defaults above apply |
| `/think`, `/no_think` in messages | supported | not supported: use `enable_thinking` | not supported: use `enable_thinking` | n/a | n/a | n/a | n/a |
| Function calling | supported (`hermes`) | supported (`qwen3_coder`) | supported (`qwen3_coder`) | supported (`hermes`) | supported (`hermes`) | **not supported**: its chat template has no tools, use a JSON schema | **not supported**: its chat template has no tools, use a JSON schema |
| Writing code | not tested | not tested | 9B: disable thinking (with it the model looped for 30 000 tokens without writing code); 4B: not tested | not tested | not tested | not tested | not tested |

## Chat

```python
reply = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    max_tokens=1024,
)
print(reply.choices[0].message.content)
```

```bash
curl https://xxxx.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer <API key>" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1024}'
```

## Thinking and sampling parameters

Qwen3, Qwen3.5-4B and 9B think before answering by default (Qwen3.5-0.8B and 2B do not; Qwen3-VL-2B-Instruct, Qwen2.5-VL-7B-Instruct and Qwen2.5-Omni-3B never think, Qwen3-VL-2B-Thinking always does and has no switch). The thinking is in `message.reasoning`; `content` holds
only the answer. The server sets no parameters: send them explicitly (values for each model in
[Model differences](#model-differences)). Qwen3 example:

```python
# With thinking: hard questions (math, code, multi-step reasoning)
thinking = dict(temperature=0.6, top_p=0.95, extra_body={"top_k": 20})
# Without thinking: simple questions, much faster
direct = dict(temperature=0.7, top_p=0.8,
              extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})

reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=2048, **thinking)
print(getattr(reply.choices[0].message, "reasoning", None))  # thinking
print(reply.choices[0].message.content)                      # answer
```

- **Always set `max_tokens`.** Without streaming, keep it below the value the notebook suggests
  (100-second limit); use streaming for long answers (see below).
- Empty `content`: `max_tokens` ran out while the model was still thinking. Raise it or disable thinking.
- Do not use `temperature=0` with thinking: the model tends to loop forever.
- Small models (Qwen3.5-0.8B, 2B): for short exact answers use a low temperature. In a test, temperature 0 was
  right 20/20 times, the recommended text values only 9/20.
- Other ways to disable thinking: `reasoning_effort="none"`; on Qwen3 also `/no_think` at the end of
  a message (`/think` turns it back on for that turn).
- Other vLLM parameters (`min_p`, `repetition_penalty`, `seed`, `stop`, `logprobs`...) go in `extra_body`.

## Streaming

Recommended for long answers and thinking: tokens arrive as they are generated, and the request is
not cut at 100 seconds.

```python
stream = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=8000, stream=True)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Images and video (Qwen3.5, Qwen3-VL and Qwen2.5-VL)

The server never downloads URLs itself: send images from your machine as base64 `data:` URLs
(`http(s)://` image links are rejected).

```python
import base64

with open("photo.jpg", "rb") as f:
    image = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

reply = client.chat.completions.create(
    model="Qwen3.5-9B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "What is in this picture?"},
    ]}],
    max_tokens=2048, temperature=0.7, top_p=0.8, presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.content)
```

Videos work the same way with `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`.
A request body is limited to 32 MiB, so keep images and videos small.

## Audio and speech (Qwen2.5-Omni)

Qwen2.5-Omni-3B also hears audio and can answer with speech. Send audio as a base64 `data:` URL (the server never
downloads URLs), and ask for speech with `modalities`. The speech comes back as a WAV in base64, in the second choice.

```python
import base64

with open("question.wav", "rb") as f:
    audio = "data:audio/wav;base64," + base64.b64encode(f.read()).decode()

reply = client.chat.completions.create(
    model="Qwen2.5-Omni-3B",
    messages=[{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": audio}},
        {"type": "text", "text": "What does the speaker say? Answer in one sentence."},
    ]}],
    modalities=["text", "audio"],          # ["text"] for the text only (faster)
    extra_body={"speaker": "Chelsie"},     # or "Ethan"
    max_tokens=256, temperature=0.2,
)
print(reply.choices[0].message.content)    # the answer as text
with open("answer.wav", "wb") as f:        # the answer as speech
    f.write(base64.b64decode(reply.choices[1].message.audio.data))
```

- **Streaming** works for the text only. An answer with speech arrives whole, after the text, speech codes and waveform
  are generated: keep such answers short, because a request without streaming is cut after 100 seconds.
- **A video with its own sound:** send only the video and add `extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}}`;
  do not send the audio separately.
- Each stage handles one request at a time by default, so parallel requests queue.

This section follows vLLM-Omni's documentation; the notebook has not been tested yet.

## Function calling

Not available with Qwen2.5-VL-7B-Instruct and Qwen2.5-Omni-3B: their chat templates have no tool support, so the server enables no tool parser. Use a [JSON schema](#json-schema) instead.

```python
import json

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
messages = [{"role": "user", "content": "What's the weather in Hanoi today?"}]

reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=1024)
call = reply.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)  # {"city": "Hanoi"}

# Run the function on your machine, then send the result back
messages += [reply.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 29})}]
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=1024)
print(final.choices[0].message.content)
```

## JSON schema

```python
schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}},
    "required": ["sentiment"],
}
reply = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Review: 'Fast delivery, careful packaging.'"}],
    max_tokens=256,
    response_format={"type": "json_schema", "json_schema": {"name": "review", "schema": schema}},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(json.loads(reply.choices[0].message.content))
```

## Completions and token counting

```python
reply = client.completions.create(model=MODEL, prompt="Once upon a time,", max_tokens=200)
print(reply.choices[0].text)
```

```bash
curl https://xxxx.trycloudflare.com/tokenize \
  -H "Authorization: Bearer <API key>" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B", "messages": [{"role": "user", "content": "Hello"}], "add_generation_prompt": true}'
# returns {"count": <tokens>, "max_model_len": <max context>, "tokens": [...]}
```

## Anthropic SDK

The base URL has **no** `/v1`:

```python
import anthropic

client = anthropic.Anthropic(base_url="https://xxxx.trycloudflare.com", api_key="<API key>")
reply = client.messages.create(model="Qwen3-4B", max_tokens=1024,
                               messages=[{"role": "user", "content": "Hello"}])
print(reply.content[-1].text)
```

## Chat UIs (Open WebUI, LibreChat...)

Add an "OpenAI-compatible" connection with URL `https://xxxx.trycloudflare.com/v1` and the API key;
the model shows up automatically. The URL changes every time the notebook is rerun.

## Isolate the prefix cache

The server reuses the KV cache of prompts that share a prefix to run faster. When several people
share a server, someone could in theory infer the start of another person's prompt from response
times. For sensitive data, send a secret string of your own; the cache is only shared between
requests with the same string:

```python
reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024,
                                       extra_body={"cache_salt": "<your secret string>"})
```

## Error codes

| Code | Meaning | What to do |
|---|---|---|
| 401 | Missing or wrong key | Send `Authorization: Bearer <key>` (or `x-api-key`) |
| 404 | Path not exposed | See the API list in the [README](../README.md) |
| 400 | Bad request, usually over the context limit | Shorten the prompt or lower `max_tokens`; check with `/tokenize` |
| 502 | vLLM is down or restarting | Check the Monitor cell in the notebook |
| 524 | Non-streaming request took over 100 seconds | Use `stream=true`, or lower `max_tokens` |
| 530 | Tunnel is down | Rerun the notebook and use the new URL |

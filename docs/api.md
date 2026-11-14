# Calling the API

**English** · [Tiếng Việt](api.vi.md)

The notebook prints the **Base URL** (`https://xxxx.trycloudflare.com/v1`), the **API key** and the
**model** name (`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`,
`Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3-VL-2B-Instruct`, `Qwen3-VL-2B-Thinking`, `Qwen2.5-VL-7B-Instruct`, `Qwen2.5-Omni-3B`, `Qwen2.5-Coder-3B-Instruct`, `Qwen2.5-Coder-3B`, `Qwen3-Reranker-0.6B`, `Qwen3-Reranker-4B`, `Qwen3-Reranker-8B`, `Qwen3-VL-Reranker-2B`).
The examples use the `openai` library (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="<API key>")
MODEL = "Qwen3-4B"
```

## Model differences

| | Qwen3-0.6B, 1.7B, 4B, 8B, 14B | Qwen3.5-0.8B, 2B | Qwen3.5-4B, 9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct | Qwen2.5-Omni-3B | Qwen2.5-Coder-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|---|---|---|---|---|---|
| Input | text | text, images, video | text, images, video | text, images, video | text, images, video | text, images, video | text, images, audio, video; output text **and speech** | text (code) | text (code) |
| Context | 40 960 tokens | up to 262 144 tokens | up to 262 144 tokens | up to 262 144 tokens (needs ~28 GiB of KV cache: smaller GPUs get less) | same as Instruct | up to 128 000 tokens (~7 GiB of KV cache) | 32 768 tokens | 32 768 tokens | 32 768 tokens |
| Thinking by default | on | **off**: send `enable_thinking: true` to turn it on | on | **never** | **always**, no switch | **never** | **never** | **never** | **never** |
| Defaults when a request omits sampling parameters | the model's `generation_config.json` (temperature 0.6, top_p 0.95, top_k 20) | vLLM's defaults (temperature 1.0, no top_p/top_k): answers are unreliable (a test question was right 9/20 times vs 20/20 with the recommended values), **always send parameters** | vLLM's defaults (temperature 1.0, no top_p/top_k): answers are unreliable (a test question was right 9/20 times vs 20/20 with the recommended values), **always send parameters** | the model's `generation_config.json` (temperature 0.7, top_p 0.8, top_k 20) | the model's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) | the model's `generation_config.json` (temperature 0.000001, repetition_penalty 1.05): almost greedy | the defaults of its vLLM-Omni config (Thinker: temperature 0, repetition_penalty 1.1) | the model's `generation_config.json` (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) | vLLM's defaults (temperature 1.0): **always send `temperature`** |
| Recommended with thinking | `temperature=0.6, top_p=0.95, top_k=20` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | n/a | `temperature=1.0, top_p=0.95, top_k=20` | n/a | n/a | n/a | n/a |
| Recommended without thinking | `temperature=0.7, top_p=0.8, top_k=20` | text: `temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0`; images: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` | images: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`; text: `temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0` | n/a | none published by Qwen; the defaults above suit factual answers | none published by Qwen; the defaults above apply | none published by Qwen; the defaults above are a good start | none published by Qwen; low `temperature` (0–0.2) for completion |
| `/think`, `/no_think` in messages | supported | not supported: use `enable_thinking` | not supported: use `enable_thinking` | n/a | n/a | n/a | n/a | n/a | n/a |
| Function calling | supported (`hermes`) | supported (`qwen3_coder`) | supported (`qwen3_coder`) | supported (`hermes`) | supported (`hermes`) | **not supported**: its chat template has no tools, use a JSON schema | **not supported**: its chat template has no tools, use a JSON schema | supported (`hermes`) | not meant for it: base model, use `/v1/completions` |
| Writing code | not tested | not tested | 9B: disable thinking (with it the model looped for 30 000 tokens without writing code); 4B: not tested | not tested | not tested | not tested | not tested | not tested | not tested |

The Qwen3 rerankers (0.6B, 4B, 8B, VL-2B) are not in this table because they write no text: see [Rerank and score](#rerank-and-score-qwen3-reranker).

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

Qwen3, Qwen3.5-4B and 9B think before answering by default (Qwen3.5-0.8B and 2B do not; Qwen3-VL-2B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-Omni-3B and Qwen2.5-Coder-3B never think, Qwen3-VL-2B-Thinking always does and has no switch). The thinking is in `message.reasoning`; `content` holds
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

- `extra_body={"mm_processor_kwargs": ...}` sets how images and video are processed (for example the frame rate of a video).
  vLLM 0.33 refuses this field unless the server runs with `--trust-request-mm-kwargs`, which the `Qwen2.5-VL-7B-Instruct`
  notebooks and the `_update` notebooks of `Qwen3.5` and `Qwen3-VL` set (not tested yet); the `Qwen3.5` and `Qwen3-VL`
  notebooks tested earlier do not set it and answer HTTP 400.

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

## Code completion (Qwen2.5-Coder)

Qwen2.5-Coder-3B (the base model) completes code and fills in the middle of it through `/v1/completions`. Put the code
before the gap after `<|fim_prefix|>`, the code after it after `<|fim_suffix|>`, and end the prompt with `<|fim_middle|>`;
the model writes what belongs in the gap.

```python
prefix = "def is_prime(n):\n    "
suffix = "\n\nprint(is_prime(7))\n"
reply = client.completions.create(
    model="Qwen2.5-Coder-3B",
    prompt=f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
    max_tokens=128, temperature=0.2,
    stop=["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>"],
)
print(prefix + reply.choices[0].text + suffix)   # the file with the gap filled
```

- Always send `temperature`: without it vLLM samples at 1.0. Qwen2.5-Coder-3B-Instruct is the model for chat, tools and
  explanations; it adds the system message "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when
  you send none, so send your own system message to replace it.
- The FIM format follows Qwen's documentation; this section has not been tested here yet.

## Rerank and score (Qwen3-Reranker)

Qwen3-Reranker (0.6B, 4B, 8B and VL-2B) writes no text, so it has no chat: it takes a query and documents and returns a relevance score
from 0 to 1 for each one (the probability of "yes" against "no" in the prompt Qwen trained it with). It has its own
paths, without `/v1`: `/rerank` (sorted, with `top_n`) and `/score` (in the order you sent). `/v1/chat/completions`
and `/v1/completions` are not available. Use plain HTTP; the base URL is the tunnel URL without `/v1`.

```python
import requests

BASE = "https://xxxx.trycloudflare.com"          # the URL without /v1
HEADERS = {"Authorization": "Bearer <API key>"}  # or x-api-key
docs = ["Gravity is a force that attracts two bodies towards each other.",
        "The capital of China is Beijing.",
        "Paris is the capital of France."]

reply = requests.post(f"{BASE}/rerank", headers=HEADERS, timeout=90, json={
    "model": "Qwen3-Reranker-0.6B",
    "query": "What is the capital of China?",
    "documents": docs,
    "top_n": 2,                                    # optional: keep the best 2
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",  # optional
}).json()
for hit in reply["results"]:                       # sorted, best first
    print(hit["index"], round(hit["relevance_score"], 3), hit["document"]["text"])

scores = requests.post(f"{BASE}/score", headers=HEADERS, timeout=90, json={
    "model": "Qwen3-Reranker-0.6B", "queries": "What is the capital of China?", "documents": docs,
}).json()
print([item["score"] for item in scores["data"]])  # one score per document, in the order sent
```

- **`instruction`** describes your task and goes into the prompt of every pair (`Given a web search query, retrieve
  relevant passages that answer the query` when omitted). Qwen reports 1% to 5% better results with an instruction
  written for your task, in English, even for other languages. This small model moves its scores very little when
  the instruction changes (in a test a different instruction changed the best score from 0.9995 to 0.9979).
- **Long inputs:** `max_tokens_per_query` and `max_tokens_per_doc` cut each query or document to that many tokens
  before scoring; the whole pair must fit in the context (40 960 tokens).
- **Time:** a request is cut after 100 seconds (error 524). The notebook prints how many documents per second it
  scores; split very large lists into several requests.
- **Other paths** (`/v1/rerank`, `/v2/rerank`, `/v1/score`) are the same APIs under other names.

The 0.6B and 4B models were tested on Colab T4 (all the calls above, see the [model's README](../notebooks/retrieval/qwen3_reranker_0_6b/README.md)); Kaggle and the 8B model have not been tested yet; Qwen3-VL-Reranker-2B was tested on Colab T4 (text, images and video). The examples use `Qwen3-Reranker-0.6B`; the 4B model takes the same calls.

### Images and video (Qwen3-VL-Reranker)

Qwen3-VL-Reranker-2B scores pairs in which the query and each document can be text, an image or a screenshot, a video, or a
mix. Wrap such a query or document as `{"content": [...]}` with the same parts as a chat message, and send media as base64
`data:` URLs (the server never downloads links: an `https://` or `file://` link is rejected with HTTP 400). A plain string
still works as text, and both kinds can be mixed in one list.

```python
import base64

def data_url(path, mime="image/png"):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

reply = requests.post(f"{BASE}/rerank", headers=HEADERS, timeout=90, json={
    "model": "Qwen3-VL-Reranker-2B",
    "query": "Which page shows the quarterly revenue chart?",
    "documents": [
        {"content": [{"type": "image_url", "image_url": {"url": data_url("page1.png")}}]},
        {"content": [{"type": "text", "text": "Page 2:"},
                     {"type": "image_url", "image_url": {"url": data_url("page2.png")}}]},   # text and image together
        "Plain text documents work too.",
    ],
}).json()
```

- A video goes in as `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`. vLLM reads at most 32
  frames of a video by default. The query can be an image or a video too: `"query": {"content": [...]}`.
- `mm_processor_kwargs` (a field of both APIs) sets how images and video are processed. vLLM refuses it unless the server runs
  with `--trust-request-mm-kwargs`, which `colab_qwen3_vl_reranker_2b_update.ipynb` and the Kaggle notebook set (not tested yet); the
  Colab notebook tested so far does not set it and answers HTTP 400.
- Keep the request body under 32 MiB, and the whole request under 100 seconds.

## Function calling

Not available with Qwen2.5-VL-7B-Instruct and Qwen2.5-Omni-3B (their chat templates have no tool support, so the server enables no tool parser) nor with the base Qwen2.5-Coder-3B. Use a [JSON schema](#json-schema) instead.

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
| 404 | Path not exposed | Each model type has its own APIs (chat models: `/v1/...`; Qwen3-Reranker: `/rerank`, `/score`), see the [README](../README.md) |
| 400 | Bad request, usually over the context limit | Shorten the prompt or lower `max_tokens`; check with `/tokenize` |
| 502 | vLLM is down or restarting | Check the Monitor cell in the notebook |
| 524 | Non-streaming request took over 100 seconds | Use `stream=true`, or lower `max_tokens` |
| 530 | Tunnel is down | Rerun the notebook and use the new URL |

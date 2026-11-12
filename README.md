# Ready-made notebooks to host open models on Google Colab and Kaggle

**English** · [Tiếng Việt](README.vi.md)

Pre-built notebooks that serve an open model with [vLLM](https://github.com/vllm-project/vllm) on a
Colab or Kaggle GPU and expose it to the Internet as a **key-protected, OpenAI-compatible API**. No
hosting code to write: open a notebook below and run it. The notebook only hosts the model: prompts,
parameters and answers stay on the caller's side.

```
caller ──HTTPS──▶ xxxx.trycloudflare.com ──▶ proxy (checks key) ──▶ vLLM ──▶ GPU
                  (Cloudflare Tunnel)        └─────── inside the notebook ───────┘
```

## Available models

### Text models

| Model | Weights | Colab | Kaggle |
|---|---|---|---|
| [Qwen3-0.6B](notebooks/text/qwen3_0_6b/README.md) | 1.4 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/colab_qwen3_0_6b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_0_6b/kaggle_qwen3_0_6b.ipynb) |
| [Qwen3-1.7B](notebooks/text/qwen3_1_7b/README.md) | 3.8 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/colab_qwen3_1_7b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_1_7b/kaggle_qwen3_1_7b.ipynb) |
| [Qwen3-4B](notebooks/text/qwen3_4b/README.md) | 7.5 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_4b/colab_qwen3_4b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_4b/kaggle_qwen3_4b.ipynb) |
| [Qwen3-8B](notebooks/text/qwen3_8b/README.md) | 15.3 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/colab_qwen3_8b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_8b/kaggle_qwen3_8b.ipynb) |
| [Qwen3-14B](notebooks/text/qwen3_14b/README.md) | 27.5 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/text/qwen3_14b/colab_qwen3_14b.ipynb) | — (needs ~35 GB of GPU memory) |

### Code models

| Model | Weights | Colab | Kaggle |
|---|---|---|---|
| [Qwen2.5-Coder-3B-Instruct](notebooks/code/qwen2_5_coder_3b_instruct/README.md) | 5.8 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/colab_qwen2_5_coder_3b_instruct.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b_instruct/kaggle_qwen2_5_coder_3b_instruct.ipynb) |
| [Qwen2.5-Coder-3B](notebooks/code/qwen2_5_coder_3b/README.md) | 5.8 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b/colab_qwen2_5_coder_3b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/code/qwen2_5_coder_3b/kaggle_qwen2_5_coder_3b.ipynb) |

### Vision-language models

| Model | Weights | Colab | Kaggle |
|---|---|---|---|
| [Qwen3.5-0.8B](notebooks/vision-language/qwen3_5_0_8b/README.md) (images, video) | 1.6 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_0_8b/colab_qwen3_5_0_8b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_0_8b/kaggle_qwen3_5_0_8b.ipynb) |
| [Qwen3.5-2B](notebooks/vision-language/qwen3_5_2b/README.md) (images, video) | 4.2 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/colab_qwen3_5_2b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_2b/kaggle_qwen3_5_2b.ipynb) |
| [Qwen3.5-4B](notebooks/vision-language/qwen3_5_4b/README.md) (images, video) | 8.7 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_4b/colab_qwen3_5_4b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_4b/kaggle_qwen3_5_4b.ipynb) |
| [Qwen3.5-9B](notebooks/vision-language/qwen3_5_9b/README.md) (images, video) | 18.0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/colab_qwen3_5_9b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_5_9b/kaggle_qwen3_5_9b.ipynb) |
| [Qwen3-VL-2B-Instruct](notebooks/vision-language/qwen3_vl_2b_instruct/README.md) (images, video) | 4.0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_instruct/colab_qwen3_vl_2b_instruct.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_instruct/kaggle_qwen3_vl_2b_instruct.ipynb) |
| [Qwen3-VL-2B-Thinking](notebooks/vision-language/qwen3_vl_2b_thinking/README.md) (images, video) | 4.0 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_thinking/colab_qwen3_vl_2b_thinking.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen3_vl_2b_thinking/kaggle_qwen3_vl_2b_thinking.ipynb) |
| [Qwen2.5-VL-7B-Instruct](notebooks/vision-language/qwen2_5_vl_7b_instruct/README.md) (images, video) | 15.4 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen2_5_vl_7b_instruct/colab_qwen2_5_vl_7b_instruct.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/vision-language/qwen2_5_vl_7b_instruct/kaggle_qwen2_5_vl_7b_instruct.ipynb) |

### Omni models

| Model | Weights | Colab | Kaggle |
|---|---|---|---|
| [Qwen2.5-Omni-3B](notebooks/omni/qwen2_5_omni_3b/README.md) (images, audio, video; speech out) | 11.2 GiB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinhqhuy7/notebook-model-host/blob/main/notebooks/omni/qwen2_5_omni_3b/colab_qwen2_5_omni_3b.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/dinhqhuy7/notebook-model-host/blob/main/notebooks/omni/qwen2_5_omni_3b/kaggle_qwen2_5_omni_3b.ipynb) |

Choose the GPU before **Run all** to fit your needs; it must have enough memory for the model (✗ in
the table below means not enough). Only GPUs marked "measured" have been tested. Each model was tested on
the GPU that happened to be used, often a modest one (T4, 2 × T4), so the speeds in the test results are
not the best you can get: a stronger GPU answers faster, and some models were not tested on the best GPU
available. Estimated memory left for context:

**Text models**

| GPU | Platform | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|---|
| T4 (15 GB) | Colab | ~10 GiB (measured: 11.1 GiB) | ~8 GiB (measured: 8.7 GiB) | ~4 GiB (measured: 4.2 GiB) | ✗ not enough memory | ✗ not enough memory |
| T4 x2 | Kaggle | ~22 GiB (measured: 23.5 GiB) | ~19 GiB (measured: 21.4 GiB) | ~16 GiB (measured: 16.8 GiB) | ~8 GiB (measured: 9.2 GiB) | ✗ not enough memory |
| L4 (22.5 GB) | Colab | ~17 GiB | ~14 GiB | ~11 GiB | ~3 GiB | ✗ not enough memory |
| A100 40 GB | Colab | ~33 GiB | ~30 GiB | ~27 GiB | ~19 GiB (measured: 18.5 GiB) | ~6 GiB (measured: 5.85 GiB) |
| A100 / H100 80 GB | Colab | ~68 GiB | ~66 GiB | ~62 GiB | ~54 GiB | ~42 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM |

**Code models**

| GPU | Platform | Qwen2.5-Coder-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|
| T4 (15 GB) | Colab | ~6 GiB | ~6 GiB |
| T4 x2 | Kaggle | ~17 GiB | ~17 GiB |
| L4 (22.5 GB) | Colab | ~12 GiB | ~12 GiB |
| A100 40 GB | Colab | ~28 GiB | ~28 GiB |
| A100 / H100 80 GB | Colab | ~64 GiB | ~64 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM |

**Vision-language models**

| GPU | Platform | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Thinking | Qwen2.5-VL-7B-Instruct |
|---|---|---|---|---|---|---|---|---|
| T4 (15 GB) | Colab | ~7 GiB (measured: 9.1 GiB) | ~4 GiB (measured: 6.2 GiB) | ✗ not enough memory | ✗ not enough memory | ~7 GiB (measured: 6.6 GiB) | ~7 GiB (measured: 6.6 GiB) | ✗ not enough memory |
| T4 x2 | Kaggle | ~18 GiB (measured: 20.4 GiB) | ~16 GiB (measured: 17.3 GiB) | ~11 GiB (measured: 11.9 GiB) | ~2 GiB (measured: 3.5 GiB) | ~18 GiB (measured: 18.2 GiB) | ~18 GiB (measured: 18.2 GiB) | ~7 GiB |
| L4 (22.5 GB) | Colab | ~14 GiB | ~11 GiB | ~7 GiB (measured: 9.2 GiB) | ✗ not enough memory | ~13 GiB | ~13 GiB | ~2 GiB |
| A100 40 GB | Colab | ~29 GiB | ~27 GiB | ~22 GiB | ~13 GiB (measured: 13.2 GiB) | ~29 GiB (measured: 29.03 GiB) | ~29 GiB | ~18 GiB |
| A100 / H100 80 GB | Colab | ~65 GiB | ~62 GiB | ~58 GiB | ~49 GiB (measured A100 80 GB: 49.3 GiB) | ~65 GiB | ~65 GiB | ~54 GiB |
| P100 | Kaggle | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM | ✗ not supported by vLLM |

**Omni models**

| GPU | Platform | Qwen2.5-Omni-3B |
|---|---|---|
| T4 (15 GB) | Colab | ✗ not enough memory |
| T4 x2 | Kaggle | ~9 GiB |
| L4 (22.5 GB) | Colab | ~4 GiB |
| A100 40 GB | Colab | ~20 GiB |
| A100 / H100 80 GB | Colab | ~56 GiB |
| P100 | Kaggle | ✗ not supported by vLLM |

"measured" values come from real runs; the rest are estimates. Other GPUs: step 1 of the notebook
checks them. Qwen3.5-9B needs ~4.5× less KV cache per token (hybrid attention), so 13 GiB fits its
native 262 144-token context. Every notebook with its test status: [notebooks/README.md](notebooks/README.md)
(by type: [text](notebooks/text/README.md), [vision-language](notebooks/vision-language/README.md)).
Each model has its own folder with its notebooks, GPU table and full test results: open it from the model name there.

## How to host a model on Google Colab or Kaggle

**Colab:** open the notebook → Runtime → Change runtime type for another GPU → (recommended) add a
`LLM_API_KEY` secret in 🔑 Secrets and enable Notebook access → **Run all**.

**Kaggle:** open the notebook → Settings: pick an **Accelerator**, turn **Internet** on (requires a
phone-verified account) → (recommended) Add-ons → Secrets: add `LLM_API_KEY` and attach it →
**Run All**.

The first run takes about 7–15 minutes. Step 8 prints the **Base URL** and **API key**. Keep the tab
open while in use; when done, run the **Shut down** cell and release the GPU. Without the secret,
the key changes on every run.

## Call the hosted model (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="<API key>")
reply = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
)
print(reply.choices[0].message.content)
```

The server exposes `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`
(Anthropic-style), `/v1/models`, `/tokenize`, `/detokenize`, and supports thinking, function calling,
JSON schemas, (Qwen3.5, Qwen2.5-VL) images and video, and (Qwen2.5-Omni) audio in and speech out. Full examples: [docs/api.md](docs/api.md).

## Security and data

- **One way in:** vLLM and the proxy listen on `127.0.0.1`; the Internet can only reach them through
  the tunnel URL. The proxy checks the key on every path and blocks admin APIs, because vLLM's
  `--api-key` does not protect `/invocations`, `/sleep`...
- **One shared key:** change `LLM_API_KEY` and rerun to revoke access. The key appears in the
  notebook output: clear outputs before sharing; on Kaggle keep the notebook Private.
- **No content stored:** the server adds no system prompt, sets no sampling parameters (requests
  that omit them get the model's own defaults when it ships them, see
  [Model differences](docs/api.md#model-differences)), and logs no prompts or answers.
- **No server-side downloads:** Qwen3.5 accepts images and video only as base64 `data:` URLs sent by
  the caller; the server rejects `http(s)` links instead of fetching them.
- **Content still passes through** Cloudflare (HTTPS is decrypted there) and the notebook machine's
  memory. Only use servers run by people you trust.
- **Shared prefix cache:** for sensitive data, add a `cache_salt` so you do not share cache with
  others ([details](docs/api.md#isolate-the-prefix-cache)).

## Limits

- The URL changes on every run; the server only lives as long as the notebook session.
- Requests without streaming are cut after 100 seconds (error 524). For long answers or thinking use
  `stream=true`, which is not cut (tested: 124 seconds).
- Long prompts are slow on T4 (~350 tokens/second to read, whatever the model size): on a single T4,
  stream prompts over ~30 000 tokens; Kaggle's 2 × T4 read them twice as fast.
- Suited to personal use or a small group, not a public service.

## Platform terms

Colab and Kaggle restrict running web services or proxies, depending on the account plan (see the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[Kaggle Terms](https://www.kaggle.com/terms)). Read the terms for your plan before exposing an API
to the Internet.

## Generate your own notebook

The notebooks in `notebooks/` are ready-made, tested samples. `tools/build_notebooks.py` is a separate script for
generating notebooks of your own, for any model and platform in its list, into a folder you choose. It never writes
into `notebooks/`.

```bash
python tools/build_notebooks.py --list                                   # models, types, minimum GPU memory
python tools/build_notebooks.py --model qwen3_8b --platform colab        # -> generated/colab_qwen3_8b.ipynb
python tools/build_notebooks.py --model qwen3_4b qwen3_8b --out my_notebooks
```

Upload the file to Colab or Kaggle and run all cells, as with the samples. To add a model, add an entry to `MODELS`
in `tools/nbgen/kinds/<type>.py` (its `kind` picks the type, see `TYPES` in `tools/nbgen/config.py`) and generate.
A model that does not fit a platform sets `platforms` and `unavailable` (the reason). Models other than Qwen3 may need
a different `--reasoning-parser` / `--tool-call-parser`.

## Development

```
notebooks/   sample notebooks grouped by model type (text/, vision-language/), one folder per model with its
             notebooks and a README with the GPU table and test results
proxy/       key-checking proxy (embedded in every notebook)
tools/       build_notebooks.py and nbgen/: notebook template, platforms and models
tests/       proxy and generator tests, no GPU needed
docs/        API guide (api.md, api.vi.md)
```

```bash
pip install aiohttp
python -m unittest discover -s tests
```

## License

Code: [MIT](LICENSE). Models have their own licenses (Qwen3, Qwen3.5, Qwen2.5-VL-7B-Instruct: Apache 2.0; Qwen2.5-Omni-3B and Qwen2.5-Coder-3B: Qwen Research License, read their terms before use).

"""The notebook template and the function that builds one notebook."""

from __future__ import annotations

import json
import re
from .catalog import gpu_table, min_gpu_mib, notebook_name, runnable
from .config import BANNER_ID, CLOUDFLARED_VERSION, GPU_MEMORY, PLATFORMS, REPO, UPDATE_SUFFIX, VLLM_VERSION


# Notebook template: ("md" | "code", text). {{NAME}} is substituted at build time.
CELLS = [
    ("md", """\
# Host {{SERVED_NAME}} on {{PLATFORM_NAME}}

{{BADGE}}

Runs [{{MODEL_ID}}](https://huggingface.co/{{MODEL_ID}}) with vLLM and exposes it to the Internet
as a key-protected, OpenAI-compatible API. The notebook only hosts the model: prompts, parameters
and answers stay on the caller's side, and the server records no content.

```
caller ──HTTPS──▶ Cloudflare Tunnel ──▶ proxy (checks key) ──▶ vLLM ──▶ GPU
```

{{MODEL_NOTES}}**GPU** (weights {{WEIGHTS_GIB}} GiB; other GPUs are checked in step 1):

{{GPU_TABLE}}

**How to run:**

{{SETUP}}

Step 8 prints the URL and API key. How to call the API:
[docs/api.md](https://github.com/{{REPO}}/blob/main/docs/api.md)."""),

    ("md", """\
## 1. Check the GPU and Internet

The notebook downloads vLLM, the model and cloudflared, so it needs Internet. vLLM {{VLLM_VERSION}} needs NVIDIA driver ≥ 580 (CUDA 13) and GPU compute capability ≥ 7.0. The
notebook picks `bfloat16` (Ampere and newer) or `float16` (T4), and splits the model across all GPUs
when there are several."""),

    ("code", """\
import socket, subprocess

for host in ("pypi.org", "huggingface.co", "github.com"):
    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        raise RuntimeError(f"No Internet (cannot resolve {host}): {{NO_INTERNET}}") from None

out = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=True,
).stdout.strip()
rows = [[s.strip() for s in line.split(",")] for line in out.splitlines()]
GPU_NAME, DRIVER = rows[0][0], rows[0][2]
GPU_MIB = min(int(r[1]) for r in rows)
CAP = min(float(r[3]) for r in rows)
TP = len(rows) if len(rows) in (1, 2, 4, 8) else 1
DTYPE = "bfloat16" if CAP >= 8.0 else "float16"
assert int(DRIVER.split(".")[0]) >= 580, f"Driver {DRIVER} is older than 580: try another GPU type"
assert CAP >= 7.0, f"vLLM does not support {GPU_NAME}: {{GPU_FAIL}}"
assert GPU_MIB * TP >= {{MIN_GPU_MIB}}, f"{GPU_MIB * TP} MiB is not enough: {{GPU_FAIL}}"
print(f"{len(rows)} × {GPU_NAME}, {GPU_MIB} MiB, driver {DRIVER}, compute {CAP} → {DTYPE}, {TP} GPU")"""),

    ("md", """\
## 2. Settings

- `MAX_MODEL_LEN`: max tokens per request. `auto` = the longest that fits on the GPU.
- `GPU_MEMORY`: fraction of GPU memory for vLLM; lower to 0.85 on out-of-memory errors.
- `ENFORCE_EAGER`: disables CUDA graphs; loads faster with less memory but generates slower.

The API key comes from the `LLM_API_KEY` secret; without it a random key is generated on every run.
{{SHARE_WARNING}}"""),

    ("code", """\
import os, secrets

MODEL_ID = "{{MODEL_ID}}"
SERVED_NAME = "{{SERVED_NAME}}"
MODEL_DIR = "{{BASE}}/models/{{SERVED_NAME}}"
WORK = "{{BASE}}/llm_host"
PORT, PROXY_PORT = 8000, 8001

MAX_MODEL_LEN = "auto"  #@param {type:"string"}
GPU_MEMORY = {{GPU_MEMORY}}  #@param {type:"number"}
ENFORCE_EAGER = False  #@param {type:"boolean"}

{{SECRET_CODE}}
print("API key:", "from the LLM_API_KEY secret" if API_KEY else "randomly generated")
API_KEY = API_KEY or secrets.token_urlsafe(32)
os.makedirs(WORK, exist_ok=True)"""),

    ("md", """\
## 3. Install vLLM

Takes a few minutes. pip then prints red `ERROR: pip's dependency resolver...` lines about
preinstalled packages (google-colab, gradio...): this is expected and harmless, since vLLM runs in its
own process. The install succeeded if the last line shows the vLLM version."""),

    ("code", """\
!pip install -q "vllm=={{VLLM_VERSION}}"
!vllm --version"""),

    ("md", """\
## 4. Download the model (~{{DOWNLOAD_GB}} GB)"""),

    ("code", """\
import shutil
from huggingface_hub import snapshot_download

os.makedirs(MODEL_DIR, exist_ok=True)
free_gb = shutil.disk_usage(MODEL_DIR).free / 1e9
assert free_gb > {{DOWNLOAD_GB}} * 1.1, f"Only {free_gb:.0f} GB of disk left, need ~{{DOWNLOAD_GB}} GB"
snapshot_download(MODEL_ID, local_dir=MODEL_DIR)"""),

    ("md", """\
## 5. Start the server

The server adds no system prompt, sets no sampling parameters ({{SAMPLING}}), and logs no prompts
or answers. The flags only enable features that requests can use:

{{FLAG_NOTES}}

The server only listens on `127.0.0.1`. The first start takes a few minutes."""),

    ("code", """\
import json, time, urllib.request

LOG = f"{WORK}/server.log"
cmd = [
    "vllm", "serve", MODEL_DIR, "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1", "--port", str(PORT), "--api-key", API_KEY,
    "--dtype", DTYPE, "--tensor-parallel-size", str(TP),
    "--max-model-len", str(MAX_MODEL_LEN), "--gpu-memory-utilization", str(GPU_MEMORY),
    "--enable-prefix-caching", "--disable-uvicorn-access-log",
    {{SERVE_FLAGS}},
]
if ENFORCE_EAGER:
    cmd.append("--enforce-eager")
server = subprocess.Popen(cmd, stdout=open(LOG, "ab"), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    with urllib.request.urlopen(urllib.request.Request(base + path, data, headers), timeout=300) as r:
        return json.loads(r.read())


end = time.time() + 30 * 60
while time.time() < end and server.poll() is None and not healthy():
    time.sleep(5)
if not healthy():
    log = open(LOG, errors="replace").read()
    print(log[-3000:])
    errors = [l for l in log.splitlines() if "Error" in l and "Traceback" not in l]
    if errors:  # the root cause is often far above the last lines
        print("\\nError lines in the log:\\n" + "\\n".join(errors[-15:]))
    raise RuntimeError("The server did not start; see the log above")

LOCAL = f"http://127.0.0.1:{PORT}"
CONTEXT = call(LOCAL, "/v1/models")["data"][0]["max_model_len"]
print(f"Server ready, max context {CONTEXT} tokens")"""),

    ("md", """\
## 6. Measure speed

One request with no content generates 256 tokens (`ignore_eos`) to measure speed. Nothing is printed
from the answer."""),

    ("code", """\
started = time.time()
reply = call(LOCAL, "/v1/chat/completions", {
    "model": SERVED_NAME, "messages": [{"role": "user", "content": "."}],
    "max_tokens": 256, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False},
})
TOKENS_PER_SECOND = reply["usage"]["completion_tokens"] / (time.time() - started)
print(f"~{TOKENS_PER_SECOND:.0f} tokens/second")"""),

    ("md", """\
## 7. Key-checking proxy

vLLM's `--api-key` does not protect every path (`/invocations`, `/sleep`... stay open). The proxy
is the only way in: it checks the key on every path and only lets the model APIs through. Source and
tests: [proxy/proxy.py](https://github.com/{{REPO}}/blob/main/proxy/proxy.py)."""),

    ("code", """\
import sys, urllib.error

PROXY_FILE = f"{WORK}/proxy.py"
with open(PROXY_FILE, "w") as f:
    f.write(r'''{{PROXY_SRC}}''')
proxy = subprocess.Popen([sys.executable, PROXY_FILE, str(PROXY_PORT), str(PORT), API_KEY],
                         stdout=open(f"{WORK}/proxy.log", "ab"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
PROXY = f"http://127.0.0.1:{PROXY_PORT}"
for _ in range(30):
    try:
        call(PROXY, "/v1/models")
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError(f"The proxy did not start; see {WORK}/proxy.log")

for path, expected in [("/v1/chat/completions", 401), ("/invocations", 404)]:  # POST without key
    try:
        urllib.request.urlopen(urllib.request.Request(PROXY + path, b"{}"))
        raise AssertionError(f"{path} is not blocked")
    except urllib.error.HTTPError as error:
        assert error.code == expected, (path, error.code)
print("Proxy running: no key → 401, admin APIs → 404")"""),

    ("md", """\
## 8. Expose to the Internet

Cloudflare quick tunnel: no account needed, the URL changes on every run. Requests without
streaming are cut after 100 seconds (error 524); with `stream: true` the answer arrives as it is
generated and is not cut (tested: 124 seconds, 9000 tokens)."""),

    ("code", """\
import re

CLOUDFLARED = f"{WORK}/cloudflared"
subprocess.run(["wget", "-q", "-O", CLOUDFLARED, "https://github.com/cloudflare/cloudflared/releases/"
                "download/{{CLOUDFLARED_VERSION}}/cloudflared-linux-amd64"], check=True)
os.chmod(CLOUDFLARED, 0o755)
TUNNEL_LOG = f"{WORK}/tunnel.log"
tunnel = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", PROXY],
                          stdout=open(TUNNEL_LOG, "ab"), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)
PUBLIC = None
for _ in range(60):
    found = re.findall(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open(TUNNEL_LOG).read())
    if found:
        PUBLIC = found[-1]
        break
    time.sleep(2)
assert PUBLIC, f"Could not get the tunnel URL; see {TUNNEL_LOG}"
for _ in range(30):  # DNS for a new URL takes a few seconds
    try:
        call(PUBLIC, "/v1/models")
        break
    except OSError:
        time.sleep(3)
else:
    raise RuntimeError(f"Could not reach the server through the tunnel; see {TUNNEL_LOG}")

print(f'''
Base URL : {PUBLIC}/v1
API key  : {API_KEY}
Model    : {SERVED_NAME}
Context  : {CONTEXT} tokens
Speed    : ~{TOKENS_PER_SECOND:.0f} tokens/second; without streaming keep max_tokens below ~{int(TOKENS_PER_SECOND * 90)} (100-second limit)

curl {PUBLIC}/v1/chat/completions \\\\
  -H "Authorization: Bearer {API_KEY}" -H "Content-Type: application/json" \\\\
  -d '{{"model": "{SERVED_NAME}", "messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 512}}'
''')"""),

    ("md", """\
## 9. Monitor

Prints status and throughput stats every minute. Stop this cell (■) before running Shut down: the
notebook runs one cell at a time. Stopping it does not stop the server."""),

    ("code", """\
PARTS = {"vLLM": server, "proxy": proxy, "tunnel": tunnel}
try:
    while True:
        dead = [name for name, process in PARTS.items() if process.poll() is not None]
        stats = [l for l in open(LOG, errors="replace").read().splitlines() if "throughput" in l]
        state = f"ERROR: {', '.join(dead)} stopped" if dead else ("OK" if healthy() else "vLLM /health not OK")
        print(time.strftime("%H:%M:%S"), state, "|", stats[-1][max(stats[-1].find("Avg prompt"), 0):] if stats else "")
        if dead:
            break
        time.sleep(60)
except KeyboardInterrupt:
    print("Monitor stopped; the server is still running")"""),

    ("md", """\
## 10. Shut down

Stop the Monitor cell first, then run the cell below to stop the tunnel, proxy and vLLM. {{SHUTDOWN}}"""),

    ("code", """\
import signal

for process in (globals().get(n) for n in ("tunnel", "proxy", "server")):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
print("Tunnel, proxy and vLLM stopped")"""),
]


def render(text: str, values: dict) -> str:
    for key, value in values.items():  # in order: earlier values may contain later {{...}}
        text = text.replace("{{" + key + "}}", str(value))
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    assert not left, f"not substituted: {left}"
    return text


def badge_for(key: str, model: dict, update: bool = False) -> str:
    """The 'Open in Colab/Kaggle' badge; for an update it opens the update notebook, not the tested one."""
    badge = PLATFORMS[key]["badge"]
    if update:
        assert "{{SLUG}}.ipynb" in badge
        badge = badge.replace("{{SLUG}}.ipynb", "{{SLUG}}" + UPDATE_SUFFIX + ".ipynb")
    return render(badge, {"REPO": REPO, "KIND": model["kind"], "SLUG": model["slug"]})


def build(key: str, platform: dict, model: dict, proxy_src: str, update: bool = False) -> dict:
    default_gpu = next(gpu[0] for gpu in platform["gpus"] if runnable(model, gpu))
    values = {
        "BADGE": badge_for(key, model, update), "SETUP": platform["setup"], "GPU_TABLE": gpu_table(model, platform),
        "SECRET_CODE": platform["secret_code"], "SHARE_WARNING": platform["share_warning"],
        "SHUTDOWN": platform["shutdown"], "GPU_FAIL": platform["gpu_fail"], "NO_INTERNET": platform["no_internet"],
        "PLATFORM_NAME": platform["name"], "BASE": platform["base"], "DEFAULT_GPU": default_gpu,
        "REPO": REPO, "VLLM_VERSION": VLLM_VERSION, "CLOUDFLARED_VERSION": CLOUDFLARED_VERSION,
        "SLUG": model["slug"], "MODEL_ID": model["model_id"], "SERVED_NAME": model["served_name"],
        "WEIGHTS_GIB": f"{model['weights_gib']:.1f}", "FLAG_NOTES": model["flag_notes"],
        "SAMPLING": model["sampling"], "SERVE_FLAGS": ", ".join(json.dumps(f) for f in model["flags"]),
        "MODEL_NOTES": model["notes"] + "\n\n" if model["notes"] else "",
        "DOWNLOAD_GB": model["download_gb"], "MIN_GPU_MIB": min_gpu_mib(model), "GPU_MEMORY": GPU_MEMORY,
        "PROXY_SRC": proxy_src,  # last: nothing inside the proxy source gets substituted
    }
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        source = render(text, values).split("\n")
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i:02d}", "metadata": {},
                "source": [line + "\n" for line in source[:-1]] + source[-1:]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    if update:
        official = notebook_name(key, model)
        text = (f"> ⚠ **Updated version, not tested yet.** It sits next to `{official}`, the version that was run "
                f"and measured. Use `{official}` unless you want to try this one. Once this version is tested it "
                "replaces the official notebook.")
        cells.insert(0, {"cell_type": "markdown", "id": BANNER_ID, "metadata": {}, "source": [text]})
    metadata = platform["metadata"](default_gpu)
    metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}

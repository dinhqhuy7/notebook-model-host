"""Constants, model types and platforms."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = ROOT / "notebooks.lock.json"  # {"notebooks": {"colab_qwen3_8b": "sha256:..."}}
UPDATE_SUFFIX = "_update"  # colab_qwen3_8b_update.ipynb: an updated notebook that is not tested yet
BANNER_ID = "update-banner"

REPO = "dinhqhuy7/notebook-model-host"  # GitHub repo, used by the "Open in Colab/Kaggle" badges
VLLM_VERSION = "0.33.0"
CLOUDFLARED_VERSION = "2026.10.7"

GPU_MEMORY = 0.90   # fraction of GPU memory given to vLLM
OVERHEAD_GIB = 2.0  # per GPU: activations, CUDA graphs... (measured ~1.9 GiB on A100)
MIN_KV_GIB = 1.5    # minimum KV cache to be usable

# Model types: one folder per type under notebooks/, one folder per model inside it.
# A new kind of model (embedding, speech...) is a new entry here plus MODELS entries with that "kind".
TYPES = {
    "text": {
        "name": ("Text models", "Model văn bản"),
        "intro": ("Language models that read and write text, with optional thinking and function calling.",
                  "Model ngôn ngữ đọc và viết văn bản, có thể suy nghĩ và gọi hàm."),
    },
    "vision-language": {
        "name": ("Vision-language models", "Model ảnh và video"),
        "intro": ("Multimodal models that also read images and video. The server never downloads media itself: "
                  "send images and video as base64 `data:` URLs.",
                  "Model đa phương thức, đọc được cả ảnh và video. Server không tự tải media: hãy gửi ảnh và video "
                  "dạng `data:` base64."),
    },
}

# gpus: (name, GPU count, MiB per GPU, compute capability) commonly offered on the platform.
PLATFORMS = {
    "colab": {
        "name": "Google Colab",
        "badge": "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 "(https://colab.research.google.com/github/{{REPO}}/blob/main/notebooks/{{KIND}}/{{SLUG}}/colab_{{SLUG}}.ipynb)",
        "gpus": [("T4", 1, 15360, 7.5), ("L4", 1, 23034, 8.9), ("A100 40 GB", 1, 40960, 8.0),
                 ("A100 80 GB", 1, 81920, 8.0), ("H100 80 GB", 1, 81559, 9.0)],
        "setup": """\
1. **Runtime → Change runtime type:** the notebook opens with {{DEFAULT_GPU}}; pick a stronger GPU if you have one.
2. (Recommended) 🔑 **Secrets** in the left sidebar: add `LLM_API_KEY`, enable Notebook access.
3. **Runtime → Run all.** Keep the tab open while in use.""",
        "gpu_fail": "Runtime → Change runtime type → pick a GPU with enough memory (table at the top)",
        "no_internet": "check the runtime's network, then Runtime → Disconnect and delete runtime and run again",
        "base": "/content",
        "secret_code": """\
try:
    from google.colab import userdata
    API_KEY = userdata.get("LLM_API_KEY")
except Exception:
    API_KEY = None""",
        "share_warning": "Clear outputs (Edit → Clear all outputs) before sharing the notebook: the key is in them.",
        "shutdown": "Then Runtime → Disconnect and delete runtime to release the GPU.",
        "metadata": lambda gpu: {"accelerator": "GPU", "colab": {"gpuType": gpu.split()[0]}},
    },
    "kaggle": {
        "name": "Kaggle",
        "badge": "[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
                 "(https://kaggle.com/kernels/welcome?src=https://github.com/{{REPO}}/blob/main/notebooks/{{KIND}}/{{SLUG}}/kaggle_{{SLUG}}.ipynb)",
        "gpus": [("T4 x2", 2, 15360, 7.5), ("P100", 1, 16384, 6.0)],
        "setup": """\
1. **Settings → Accelerator:** a GPU with enough memory per the table above (not P100).
2. **Settings → Internet → On** (requires a phone-verified account).
3. (Recommended) **Add-ons → Secrets:** add `LLM_API_KEY` and attach it to the notebook.
4. **Run All.** Keep the tab open while in use.""",
        "gpu_fail": "Settings → Accelerator → pick a GPU with enough memory (table at the top)",
        "no_internet": "Settings → Internet → On (requires a phone-verified account), then run all cells again",
        "base": "/tmp",
        "secret_code": """\
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("LLM_API_KEY")
except Exception:
    API_KEY = None""",
        "share_warning": "Keep the notebook Private and clear outputs before Save Version: the key is in them.",
        "shutdown": "Then click Stop session to stop using GPU quota.",
        "metadata": lambda gpu: {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                            "isInternetEnabled": True}},
    },
}

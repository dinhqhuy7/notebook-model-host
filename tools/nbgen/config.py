"""Constants, model types and platforms."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPO = "dinhqhuy7/notebook-model-host"  # GitHub repo: the notebooks link to its docs/api.md and proxy/proxy.py
VLLM_VERSION = "0.33.0"
CLOUDFLARED_VERSION = "2026.10.7"

GPU_MEMORY = 0.90   # fraction of GPU memory given to vLLM
OVERHEAD_GIB = 2.0  # per GPU: activations, CUDA graphs... (measured ~1.9 GiB on A100)
MIN_KV_GIB = 1.5    # minimum KV cache to be usable

# Model types: one folder per type under notebooks/, one folder per model inside it. A type is what the model takes in
# and gives out, not its field: a code model is a language model.
# A new kind of model is a new entry here, a module in kinds/ with that name, and MODELS entries with that "kind".
# "profile" is the set of APIs the proxy lets through for the type (proxy/proxy.py PROFILES); a model can override it.
# The host must give a model ALL its functions: list them in the model's "functions" and check that its profile has
# every API they need.
TYPES = {
    "language": {
        "profile": "chat",
        "name": ("Language models", "Model ngôn ngữ"),
        "intro": ("Models that read and write text: chat, optional thinking, function calling, and writing or completing "
                  "code. Text in, text out.",
                  "Model đọc và viết văn bản: trò chuyện, có thể suy nghĩ, gọi hàm, viết và hoàn thành code. Vào văn "
                  "bản, ra văn bản."),
    },
    "retrieval": {
        "profile": "retrieval",
        "name": ("Retrieval models", "Model truy hồi"),
        "intro": ("Models that score or embed text for search: reranking a list of documents against a query, and "
                  "turning text into vectors. Text in, scores or vectors out; they do not generate text.",
                  "Model chấm điểm hoặc nhúng văn bản để tìm kiếm: xếp hạng lại danh sách tài liệu theo câu hỏi, và "
                  "biến văn bản thành vector. Vào văn bản, ra điểm số hoặc vector; không sinh văn bản."),
    },
    "vision": {
        "profile": "chat",
        "name": ("Vision models", "Model thị giác"),
        "intro": ("Models that see: they read images and video as well as text. The server never downloads media "
                  "itself: send images and video as base64 `data:` URLs.",
                  "Model nhìn được: đọc ảnh và video cùng với văn bản. Server không tự tải media: hãy gửi ảnh và "
                  "video dạng `data:` base64."),
    },
    "audio": {
        "profile": "audio",
        "name": ("Audio models", "Model âm thanh"),
        "intro": ("Models that hear or speak: speech recognition and speech generation.",
                  "Model nghe hoặc nói: nhận dạng giọng nói và sinh giọng nói."),
    },
    "video": {
        "profile": "chat",  # placeholder until a video model needs other APIs
        "name": ("Video models", "Model video"),
        "intro": ("Models that work with video.", "Model làm việc với video."),
    },
    "multimodal": {
        "profile": "chat",
        "name": ("Multimodal models", "Model đa phương thức"),
        "intro": ("Models that take several kinds of input and answer in more than one: text, images, audio and video "
                  "in, text and speech out. They run on vLLM-Omni (a vLLM extension) in several stages, so they need "
                  "more GPU memory than their size suggests. The server never downloads media itself: send images, "
                  "audio and video as base64 `data:` URLs.",
                  "Model nhận nhiều loại đầu vào và trả lời bằng nhiều dạng: vào là văn bản, ảnh, âm thanh, video; ra "
                  "là văn bản và giọng nói. Chạy trên vLLM-Omni (phần mở rộng của vLLM) qua nhiều giai đoạn nên cần "
                  "nhiều bộ nhớ GPU hơn kích thước model. Server không tự tải media: hãy gửi ảnh, âm thanh, video dạng "
                  "`data:` base64."),
    },
}


def profile_of(model: dict) -> str:
    """The proxy profile (which APIs the proxy lets through) of a model: its own, or the one of its type."""
    return model.get("profile") or TYPES[model["kind"]]["profile"]


# gpus: (name, GPU count, MiB per GPU, compute capability) commonly offered on the platform.
PLATFORMS = {
    "colab": {
        "name": "Google Colab",
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

"""Omni models: model families and models.

Add a model: a line in MODELS (its "kind" must be "omni"), then `python tools/build_notebooks.py --list` shows it.
"""

# Qwen2.5-Omni runs on vLLM-Omni, which needs the matching vLLM release (0.32.0 with vLLM-Omni 0.32.0, the stable
# pair; the other notebooks use the newer vLLM). It loads all three stages: Thinker 8.76 GiB, Talker 0.72 GiB and
# Token2Wav 1.67 GiB, so the weights below are the whole model, and a single T4 is not enough.
QWEN2_5_OMNI = {
    "kind": "omni",
    "engine": "vllm-omni",
    "vllm_version": "0.32.0",
    "omni_version": "0.32.0",
    "input": ("text, images, audio, video (answers in text and speech)",
              "văn bản, ảnh, âm thanh, video (trả lời bằng văn bản và giọng nói)"),
    "context": 32768,
    "flags": ["--allowed-media-domains", "none.invalid"],
    "flag_notes": """\
- `--omni`: vLLM-Omni runs the model's three stages (Thinker, Talker, Code2Wav). `--stage-overrides`, built in the cell
  below, gives each stage its GPU and share of memory; every other stage setting is the model's bundled deploy config.
- `--allowed-media-domains none.invalid`: the server never downloads image/audio/video URLs itself;
  callers send media from their machine as base64 `data:` URLs.
- No tool parser and no reasoning parser: the model's chat template has no tool support and the model never thinks.""",
    "sampling": "requests that omit sampling parameters get the defaults shipped in the model's vLLM-Omni config "
                "(Thinker: temperature 0 and repetition_penalty 1.1)",
    "notes": "Hears and speaks: text, images, audio and video in; text and speech out. It runs on vLLM-Omni 0.32.0 "
             "with vLLM 0.32.0 (the other notebooks use vLLM 0.33.0), in three stages that are all loaded: "
             "11.15 GiB of weights, so **a single T4 is not enough** (one L4 or A100, or Kaggle's 2 × T4). Ask for "
             "speech with `\"modalities\": [\"text\", \"audio\"]` (a WAV in base64, in the second choice) and a voice "
             "with `\"speaker\"` (`Chelsie` or `Ethan`); an answer with audio cannot stream, so keep it short (the "
             "tunnel cuts requests after 100 seconds). Each stage takes one request at a time by default. **No "
             "function calling and no thinking mode.** License: Qwen Research License, not Apache 2.0. `float16` on "
             "T4 (Kaggle) has not been tested with this model.",
    "notes_vi": "Nghe và nói được: vào là văn bản, ảnh, âm thanh, video; ra là văn bản và giọng nói. Chạy trên "
                "vLLM-Omni 0.32.0 cùng vLLM 0.32.0 (các notebook khác dùng vLLM 0.33.0), qua ba giai đoạn đều được "
                "nạp: 11,15 GiB trọng số, nên **một T4 đơn không đủ** (một L4 hoặc A100, hoặc 2 × T4 của Kaggle). "
                "Muốn nhận giọng nói: `\"modalities\": [\"text\", \"audio\"]` (WAV base64 ở lựa chọn thứ hai), chọn "
                "giọng bằng `\"speaker\"` (`Chelsie` hoặc `Ethan`); câu trả lời có âm thanh không stream được nên giữ "
                "ngắn (tunnel cắt request sau 100 giây). Mỗi giai đoạn mặc định chỉ nhận một request mỗi lần. "
                "**Không có function calling và không có chế độ suy nghĩ.** Giấy phép: Qwen Research License, không "
                "phải Apache 2.0. `float16` trên T4 (Kaggle) chưa thử với model này.",
    "extra_gib": 3.0,  # estimate, not measured yet: three stage processes, the Talker's KV cache and the DiT activations
}

MODELS = [
    {"slug": "qwen2_5_omni_3b", "model_id": "Qwen/Qwen2.5-Omni-3B", "served_name": "Qwen2.5-Omni-3B",
     "weights_gib": 11.15, "download_gb": 12, **QWEN2_5_OMNI},
]

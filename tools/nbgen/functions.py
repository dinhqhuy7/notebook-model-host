"""What a hosted model can do, and whether the host gives it.

Every host must give a model ALL its functions: each entry of a model's "functions" is
(function, function in Vietnamese, how to call it, True or (reason it is not given, reason in Vietnamese)),
where "how to call it" is a text, or an (English, Vietnamese) pair when it has words in it.
A function that is given must use only APIs that the proxy profile of the model lets through (tests check it);
a function that is not given says why. When you add a model, list every function of its model card here.
"""

CHAT = "`/v1/chat/completions`"

THINKING_SWITCH = ("Thinking mode, on or off per request", "Chế độ suy nghĩ, bật hoặc tắt theo từng request",
                   (f"`chat_template_kwargs.enable_thinking` in {CHAT}; the thinking comes back in `reasoning`",
                    f"`chat_template_kwargs.enable_thinking` trong {CHAT}; phần suy nghĩ trả về trong `reasoning`"), True)
THINKING_ALWAYS = ("Thinking (the model always thinks)", "Suy nghĩ (model luôn suy nghĩ)",
                   (f"{CHAT}; the thinking comes back in `reasoning`", f"{CHAT}; phần suy nghĩ trả về trong `reasoning`"), True)
TOOLS = ("Function calling", "Gọi hàm",
         (f"`tools` in {CHAT}; calls come back in `tool_calls`", f"`tools` trong {CHAT}; lời gọi trả về trong `tool_calls`"), True)
NO_TOOLS = ("Function calling", "Gọi hàm", f"`tools` / {CHAT}",
            ("the model's chat template has no tool support, so the server enables no tool parser",
             "chat template của model không hỗ trợ tool nên server không bật tool parser"))
JSON_SCHEMA = ("Structured output (JSON schema)", "Dữ liệu có cấu trúc (JSON schema)",
               (f"`response_format` in {CHAT}", f"`response_format` trong {CHAT}"), True)
IMAGES = ("Image input", "Ảnh đầu vào",
          (f"`image_url` with a base64 `data:` URL in {CHAT}", f"`image_url` với URL `data:` base64 trong {CHAT}"), True)
VIDEO = ("Video input", "Video đầu vào",
         (f"`video_url` with a base64 `data:` URL in {CHAT}", f"`video_url` với URL `data:` base64 trong {CHAT}"), True)
MEDIA_URL = ("Image and video by http(s) link", "Ảnh và video qua link http(s)", f"`image_url` / `video_url` `https://...` / {CHAT}",
             ("blocked on purpose: the server never downloads links; send base64",
              "chặn có chủ đích: server không tự tải link; hãy gửi base64"))
YARN = ("Longer context with YaRN", "Context dài hơn bằng YaRN", ("rope scaling in the model config", "rope scaling trong cấu hình model"),
        ("not given: the notebook serves the native context (a rope-scaling setting is not offered yet)",
         "chưa mở: notebook phục vụ context gốc (chưa có tuỳ chọn rope scaling)"))
COMPLETION = ("Text completion", "Hoàn thành văn bản", "`prompt` / `/v1/completions`", True)
CODE_COMPLETION = ("Code completion and fill-in-the-middle", "Hoàn thành code và điền vào giữa",
                   ("`prompt` in `/v1/completions`; fill-in-the-middle with `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`",
                    "`prompt` trong `/v1/completions`; điền vào giữa bằng `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`"), True)

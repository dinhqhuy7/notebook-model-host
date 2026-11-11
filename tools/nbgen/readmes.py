"""README generators: per model, per type and the overview."""

from __future__ import annotations

import math
from .catalog import available, kv_gib, min_gpu_mib, notebook_name, pending_update, runnable, test_status, update_name
from .config import CLOUDFLARED_VERSION, GPU_MEMORY, PLATFORMS, TYPES, VLLM_VERSION
from .kinds import MODELS, TESTED
from .template import badge_for


def _helpers(vi: bool):
    t = (lambda en, vi_text: vi_text) if vi else (lambda en, vi_text: en)
    num = (lambda x: f"{x:.1f}".replace(".", ",")) if vi else (lambda x: f"{x:.1f}")
    return t, num, (".vi.md" if vi else ".md"), (1 if vi else 0)


def _other_language(vi: bool) -> str:
    return "[English](README.md) · **Tiếng Việt**" if vi else "**English** · [Tiếng Việt](README.vi.md)"


def models_table(kind: str, vi: bool, base: str) -> list:
    """Table of the models of one type; links start with `base` (path from the README to the type folder)."""
    t, num, sfx, i = _helpers(vi)
    lines = [t("| Model | Input | Weights | Context | Min. total GPU memory | ", "| Model | Đầu vào | Trọng số | Context | "
               "Tổng bộ nhớ GPU tối thiểu | ") + " | ".join(p["name"] for p in PLATFORMS.values()) + " |",
             "|---|---|---|---|---|" + "---|" * len(PLATFORMS)]
    for model in (m for m in MODELS if m["kind"] == kind):
        cells = []
        for key in PLATFORMS:
            if available(model, key):
                name = notebook_name(key, model)
                cell = f"[{name}]({base}{model['slug']}/{name})<br>{test_status(key, model, vi)}"
                if pending_update(key, model):
                    cell += f"<br>⚠ [{update_name(key, model)}]({base}{model['slug']}/{update_name(key, model)}) " + t(
                        "(update, not tested)", "(bản cập nhật, chưa chạy thử)")
                cells.append(cell)
            else:
                cells.append(f"— ({model['unavailable'][key][i]})")
        context = f"{model['context']:,}".replace(",", " ")
        lines.append(f"| [{model['served_name']}]({base}{model['slug']}/README{sfx}) | {model['input'][i]} | "
                     f"{num(model['weights_gib'])} GiB | {context} | ~{math.ceil(min_gpu_mib(model) / 1024)} GB | "
                     + " | ".join(cells) + " |")
    return lines


def _update_note(vi: bool) -> list:
    t = _helpers(vi)[0]
    return ["", t("⚠ marks an updated notebook that has not been tested yet. The tested notebook next to it "
                  "stays the official one until the update is tested and replaces it.",
                  "⚠ đánh dấu notebook bản cập nhật chưa chạy thử. Notebook đã chạy thử đứng cạnh vẫn là bản "
                  "chính thức cho tới khi bản cập nhật được chạy thử và thay thế.")]


def gpu_memory_lines(models: list, vi: bool, with_platform: bool = True) -> list:
    """Estimated memory left for context: one row per GPU of every platform, one column per model."""
    t = _helpers(vi)[0]
    head = t("| GPU | Platform | ", "| GPU | Nền tảng | ") if with_platform else "| GPU | "
    lines = [head + " | ".join(m["served_name"] for m in models) + " |",
             "|---|" + "---|" * (len(models) + (1 if with_platform else 0))]
    for key, platform in PLATFORMS.items():
        for gpu in platform["gpus"]:
            cells = []
            for model in models:
                if not available(model, key):
                    cells.append("—")
                elif gpu[3] < 7.0:
                    cells.append(t("✗ not supported by vLLM", "✗ vLLM không hỗ trợ"))
                elif runnable(model, gpu):
                    cells.append(f"~{kv_gib(model, gpu[1], gpu[2]):.0f} GiB")
                else:
                    cells.append(t("✗ not enough memory", "✗ thiếu bộ nhớ"))
            if all(c == "—" for c in cells):
                continue
            lines.append(f"| {gpu[0]} | " + (f"{platform['name']} | " if with_platform else "") + " | ".join(cells) + " |")
    return lines


def shared_settings(vi: bool) -> list:
    t = _helpers(vi)[0]
    return [t("## Shared settings", "## Thông số chung"), "",
            t(f"vLLM {VLLM_VERSION}, cloudflared {CLOUDFLARED_VERSION}. `--max-model-len auto`, "
              f"`--gpu-memory-utilization {GPU_MEMORY}`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` on T4); "
              "the model is split across all GPUs; per-model parser flags are listed in step 5 of each notebook. "
              "The server sets no sampling parameters and logs no prompts or answers.",
              f"vLLM {VLLM_VERSION}, cloudflared {CLOUDFLARED_VERSION}. `--max-model-len auto`, "
              f"`--gpu-memory-utilization {GPU_MEMORY}`, `--enable-prefix-caching`; dtype `bfloat16` (`float16` trên T4); "
              "model chia ra mọi GPU; cờ parser riêng của từng model ghi ở bước 5 trong notebook. "
              "Server không đặt tham số sinh, không ghi prompt hay câu trả lời.")]


def model_readme(model: dict, vi: bool) -> str:
    """notebooks/<kind>/<slug>/README(.vi).md: one model, its notebooks, GPU table and test results."""
    t, num, sfx, i = _helpers(vi)
    kind = TYPES[model["kind"]]
    ctx = f"{model['context']:,}".replace(",", " ")
    lines = [
        t(f"# {model['served_name']}: host it on Colab or Kaggle", f"# {model['served_name']}: host trên Colab hoặc Kaggle"),
        "", _other_language(vi), "",
        t(f"Back to [{kind['name'][0]}](../README{sfx}) · [all notebooks](../../README{sfx}) · "
          f"[main README](../../../README{sfx}).",
          f"Quay lại [{kind['name'][1]}](../README{sfx}) · [mọi notebook](../../README{sfx}) · "
          f"[README chính](../../../README{sfx})."), "",
        f"[{model['model_id']}](https://huggingface.co/{model['model_id']}) · {model['input'][i]} · "
        + t(f"weights {num(model['weights_gib'])} GiB (download ~{model['download_gb']} GB) · "
            f"context {ctx} tokens · needs at least ~{math.ceil(min_gpu_mib(model) / 1024)} GB of total GPU memory",
            f"trọng số {num(model['weights_gib'])} GiB (tải ~{model['download_gb']} GB) · "
            f"context {ctx} token · cần ít nhất ~{math.ceil(min_gpu_mib(model) / 1024)} GB tổng bộ nhớ GPU"),
    ]
    note = model["notes_vi"] if vi else model["notes"]
    if note:
        lines += ["", note]
    lines += ["", t("## Notebooks", "## Notebook"), "",
              t("| Platform | Notebook | Test status |", "| Nền tảng | Notebook | Chạy thử |"), "|---|---|---|"]
    for key, platform in PLATFORMS.items():
        if available(model, key):
            name = notebook_name(key, model)
            lines.append(f"| {platform['name']} | [{name}]({name}) {badge_for(key, model)} | {test_status(key, model, vi)} |")
        else:
            lines.append(f"| {platform['name']} | — | {model['unavailable'][key][i]} |")
    updates = [k for k in PLATFORMS if available(model, k) and pending_update(k, model)]
    if updates:
        lines += ["", t("## Updates not yet tested", "## Bản cập nhật chưa chạy thử"), "",
                  t("These notebooks are newer than the tested ones. They have not been run yet, so no measurement "
                    "applies to them. Use the tested notebook unless you want to try the update; once an update is "
                    "tested it replaces the tested notebook.",
                    "Các notebook này mới hơn bản đã chạy thử. Chúng chưa được chạy nên chưa có số đo. Dùng bản đã "
                    "chạy thử, trừ khi bạn muốn thử bản cập nhật; khi bản cập nhật được chạy thử, nó thay cho bản đã "
                    "chạy thử."), "",
                  t("| Update (not tested) | Replaces |", "| Bản cập nhật (chưa chạy thử) | Thay cho |"), "|---|---|"]
        for key in updates:
            name = update_name(key, model)
            lines.append(f"| ⚠ [{name}]({name}) {badge_for(key, model, update=True)} | "
                         f"[{notebook_name(key, model)}]({notebook_name(key, model)}) |")
    lines += ["", t("## Estimated memory left for context", "## Bộ nhớ ước tính còn cho context"), ""]
    lines += gpu_memory_lines([model], vi)
    lines += ["", t("## Test results", "## Kết quả chạy thử"), "",
              t("Speeds come from the GPU used for each run, often a modest one; a stronger GPU is faster.",
                "Tốc độ là của GPU dùng lúc chạy thử, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn.")]
    for key, platform in PLATFORMS.items():
        if not available(model, key):
            continue
        run = TESTED.get((key, model["slug"]))
        lines += ["", f"### {platform['name']}", ""]
        if not run:
            lines.append(t("Not tested yet.", "Chưa thử."))
            continue
        lines += [t("| Item | Result |", "| Mục | Kết quả |"), "|---|---|"]
        lines += [f"| {r[1] if vi else r[0]} | {r[3] if vi else r[2]} |" for r in run["rows"]]
    return "\n".join(lines) + "\n"


def type_readme(kind: str, vi: bool) -> str:
    """notebooks/<kind>/README(.vi).md: the models of one type, shared settings, GPU table."""
    t, num, sfx, i = _helpers(vi)
    info = TYPES[kind]
    models = [m for m in MODELS if m["kind"] == kind]
    lines = [
        f"# {info['name'][i]}: " + t("ready-made notebooks", "notebook dựng sẵn"), "", _other_language(vi), "",
        t(f"Back to [all notebooks](../README{sfx}) · [main README](../../README{sfx}).",
          f"Quay lại [mọi notebook](../README{sfx}) · [README chính](../../README{sfx})."), "",
        info["intro"][i], "", t("## Models", "## Model"), "",
    ] + models_table(kind, vi, "")
    if any(pending_update(k, m) for m in models for k in PLATFORMS if available(m, k)):
        lines += _update_note(vi)
    lines += ["", *shared_settings(vi), "", t("## Estimated memory left for context", "## Bộ nhớ ước tính còn cho context"), "",
              t("Choose the GPU before Run all; it needs at least the model's \"Min. total GPU memory\". "
                "Per-model details and test results are in each model's README.",
                "Chọn GPU trước khi Run all; GPU cần ít nhất \"Tổng bộ nhớ GPU tối thiểu\" của model. "
                "Chi tiết và kết quả chạy thử nằm trong README của từng model."), ""]
    lines += gpu_memory_lines(models, vi)
    return "\n".join(lines) + "\n"


def overview_readme(vi: bool) -> str:
    """notebooks/README(.vi).md: every type and model, with test status."""
    t, num, sfx, i = _helpers(vi)
    lines = [
        t("# Notebooks", "# Notebook"), "", _other_language(vi), "",
        t(f"Back to the [main README](../README{sfx}).", f"Quay lại [README chính](../README{sfx})."), "",
        t("Each notebook hosts one model on one platform. Notebooks are grouped by type of model, one folder per "
          "model with its notebooks, GPU table and test results. Pick a model, then a platform where your account has a "
          "GPU with enough memory; choose the GPU before Run all (at least the model's \"Min. total GPU memory\"). "
          "Speeds in the test results come from the GPU used for each run, often a modest one; a stronger GPU is faster.",
          "Mỗi notebook host một model trên một nền tảng. Notebook gom theo loại model, mỗi model một thư mục kèm "
          "notebook, bảng GPU và kết quả chạy thử. Chọn model, rồi chọn nền tảng mà tài khoản của bạn có GPU đủ bộ "
          "nhớ; chọn GPU trước khi Run all (ít nhất \"Tổng bộ nhớ GPU tối thiểu\" của model). "
          "Tốc độ trong kết quả chạy thử là của GPU dùng lúc đó, thường khiêm tốn; GPU mạnh hơn sẽ nhanh hơn."),
    ]
    for kind, info in TYPES.items():
        if not any(m["kind"] == kind for m in MODELS):
            continue
        lines += ["", f"## [{info['name'][i]}]({kind}/README{sfx})", "", info["intro"][i], ""]
        lines += models_table(kind, vi, kind + "/")
    if any(pending_update(k, m) for m in MODELS for k in PLATFORMS if available(m, k)):
        lines += _update_note(vi)
    return "\n".join(lines) + "\n"

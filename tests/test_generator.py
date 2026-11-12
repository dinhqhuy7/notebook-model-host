"""Test the notebook generator, no GPU needed.

    python -m unittest discover -s tests
"""

import ast
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "proxy"))
from nbgen.config import PLATFORMS, TYPES, profile_of  # noqa: E402
from nbgen.generate import generate, list_models  # noqa: E402
from nbgen.kinds import MODELS  # noqa: E402
from nbgen.kinds.retrieval import (QWEN3_RERANKER_PROMPT, QWEN3_RERANKER_TEMPLATE, QWEN3_VL_RERANKER_PROMPT,  # noqa: E402
                                   QWEN3_VL_RERANKER_TEMPLATE, VL_TEMPLATE_SHA256, VLLM_TEMPLATE_SHA256, _VLLM_TEMPLATE)
from proxy import PROFILES  # noqa: E402


class GeneratorTest(unittest.TestCase):
    def test_generates_valid_notebooks_for_every_model_and_platform(self) -> None:
        proxy = (ROOT / "proxy" / "proxy.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate(Path(tmp))
            self.assertGreaterEqual(len(paths), len(MODELS))
            for path in paths:
                nb = json.loads(path.read_text(encoding="utf-8"))
                embedded = None
                for cell in nb["cells"]:
                    source = "".join(cell["source"])
                    if cell["cell_type"] == "code":
                        ast.parse("\n".join(l for l in source.split("\n") if not l.lstrip().startswith(("!", "%"))))
                        self.assertEqual(cell["outputs"], [], path.name)
                        found = re.search(r"r'''(.*?)'''", source, re.S)
                        if found and "aiohttp" in found.group(1):
                            embedded = found.group(1)
                self.assertEqual(embedded.strip(), proxy.strip(), f"{path.name}: embedded proxy differs from proxy/proxy.py")

    def test_choose_models_and_platforms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate(Path(tmp), ["qwen3_4b"], ["colab"])
            self.assertEqual([p.name for p in paths], ["colab_qwen3_4b.ipynb"])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["colab_qwen3_4b.ipynb"])

    def test_no_badge_in_generated_notebooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = generate(Path(tmp), ["qwen3_4b"], ["kaggle"])[0].read_text(encoding="utf-8")
            self.assertNotIn("Open In", text)
            self.assertNotRegex(text, r"\{\{[A-Z_]+\}\}")

    def test_never_writes_into_the_sample_notebooks(self) -> None:
        for target in (ROOT / "notebooks", ROOT / "notebooks" / "language", ROOT / "notebooks" / "new"):
            with self.assertRaises(SystemExit):
                generate(target, ["qwen3_4b"], ["colab"])
        self.assertFalse((ROOT / "notebooks" / "new").exists())

    def test_unknown_model_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            generate(Path(tmp), ["no_such_model"])

    def test_omni_model_runs_on_vllm_omni_and_others_on_plain_vllm(self) -> None:
        def source(path: Path) -> str:
            nb = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join("".join(c["source"]) for c in nb["cells"])

        with tempfile.TemporaryDirectory() as tmp:
            omni = source(generate(Path(tmp), ["qwen2_5_omni_3b"], ["kaggle"])[0])
            plain = source(generate(Path(tmp), ["qwen3_4b"], ["kaggle"])[0])
        self.assertIn('pip install -q "vllm==0.32.0" "vllm-omni==0.32.0"', omni)
        self.assertIn('"--omni"', omni)
        self.assertIn("--stage-overrides", omni)
        self.assertIn('"--allowed-media-domains", "none.invalid"', omni)
        self.assertIn('"modalities": ["text"]', omni)  # the speed test must not generate speech
        self.assertNotIn("--omni", plain)
        self.assertNotIn("vllm-omni", plain)
        self.assertIn('pip install -q "vllm==0.33.0"', plain)

    def test_omni_needs_more_than_one_t4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nb = json.loads(generate(Path(tmp), ["qwen2_5_omni_3b"], ["colab"])[0].read_text(encoding="utf-8"))
        intro = "".join(nb["cells"][0]["source"])
        row = next(l for l in intro.split("\n") if l.startswith("| T4 |"))
        self.assertIn("not enough memory", row)

    def test_base_model_without_flags_and_instruct_with_tool_parser(self) -> None:
        def source(path: Path) -> str:
            nb = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join("".join(c["source"]) for c in nb["cells"])

        with tempfile.TemporaryDirectory() as tmp:
            base = source(generate(Path(tmp), ["qwen2_5_coder_3b"], ["colab"])[0])
            instruct = source(generate(Path(tmp), ["qwen2_5_coder_3b_instruct"], ["colab"])[0])
        self.assertNotIn("enable-auto-tool-choice", base)
        self.assertNotIn(",\n    ,", base)  # no empty entry left in the serve command
        self.assertIn('"--enable-auto-tool-choice", "--tool-call-parser", "hermes"', instruct)

    def test_every_type_has_a_proxy_profile_that_exists(self) -> None:
        for kind, info in TYPES.items():
            self.assertIn(info["profile"], PROFILES, kind)
        for model in MODELS:
            self.assertIn(profile_of(model), PROFILES, model["slug"])

    def test_every_model_lists_its_functions_and_the_proxy_lets_them_through(self) -> None:
        """The host must give a model all its functions: a function that is given needs APIs its profile allows."""
        for model in MODELS:
            functions = model.get("functions")
            self.assertTrue(functions, f"{model['slug']}: list the functions of the model (nbgen/functions.py)")
            allowed = {path for _, path in PROFILES[profile_of(model)]}
            for function in functions:
                self.assertEqual(len(function), 4, function)
                self.assertTrue(function[0] and function[1] and function[2], function)
                hosted = function[3]
                self.assertTrue(hosted is True or (isinstance(hosted, tuple) and len(hosted) == 2 and all(hosted)), function)
                if hosted is True:
                    how = function[2][0] if isinstance(function[2], tuple) else function[2]  # the English text
                    for path in re.findall(r"`(?:POST |GET )?(/[A-Za-z0-9_/.-]+)`", how):
                        self.assertIn(path, allowed, f"{model['slug']}: {function[0]} needs {path}, which the proxy blocks")

    def test_reranker_is_served_for_scoring_with_the_retrieval_proxy(self) -> None:
        def source(path: Path) -> str:
            nb = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join("".join(c["source"]) for c in nb["cells"])

        with tempfile.TemporaryDirectory() as tmp:
            rerank = source(generate(Path(tmp), ["qwen3_reranker_0_6b"], ["kaggle"])[0])
            chat = source(generate(Path(tmp), ["qwen3_4b"], ["kaggle"])[0])
        self.assertEqual(hashlib.sha256(_VLLM_TEMPLATE.encode()).hexdigest(), VLLM_TEMPLATE_SHA256)  # vLLM's file, unchanged
        self.assertIn("f.write(r'''" + QWEN3_RERANKER_TEMPLATE + "''')", rerank)  # the template, byte for byte
        self.assertIn('"--runner", "pooling", "--hf-overrides"', rerank)
        self.assertIn("Qwen3ForSequenceClassification", rerank)
        self.assertIn('"--chat-template", TEMPLATE_FILE', rerank)
        self.assertIn('API_KEY, "retrieval"]', rerank)  # the proxy runs with the retrieval profile
        self.assertIn("TP = len(rows) if len(rows) in (1, 2, 4, 8) else 1", rerank)  # Kaggle: both T4 are counted, both are used
        self.assertIn('"/rerank"', rerank)
        self.assertNotIn('call(LOCAL, "/v1/chat/completions"', rerank)  # nothing to generate
        self.assertNotIn('API_KEY, "', chat)  # the other notebooks start the proxy without a profile: chat
        self.assertIn("str(PORT), API_KEY],", chat)
        self.assertNotIn("--chat-template", chat)

    def test_kaggle_notebooks_use_every_gpu(self) -> None:
        """Kaggle always gives 2 x T4 and counts both against the quota: no Kaggle notebook may leave one idle."""
        with tempfile.TemporaryDirectory() as tmp:
            for path in generate(Path(tmp), None, ["kaggle"]):
                nb = json.loads(path.read_text(encoding="utf-8"))
                text = "\n".join("".join(c["source"]) for c in nb["cells"])
                if "--omni" in text:  # vLLM-Omni lays its stages out over both GPUs itself
                    self.assertIn("two GPUs: the Thinker has the first", text, path.name)
                else:
                    self.assertIn("TP = len(rows) if len(rows) in (1, 2, 4, 8) else 1", text, path.name)
                    self.assertIn("GPU_MIB = min(int(r[1]) for r in rows)", text, path.name)

    def test_colab_reranker_keeps_its_tested_one_gpu_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nb = json.loads(generate(Path(tmp), ["qwen3_reranker_0_6b"], ["colab"])[0].read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("TP = 1  # a small model: one GPU", text)  # the locked, tested notebook
        self.assertIn("even on Kaggle's 2 × T4", text)

    def test_reranker_4b_serves_like_the_small_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for slug in ("qwen3_reranker_4b", "qwen3_reranker_0_6b"):
                nb = json.loads(generate(Path(tmp), [slug], ["kaggle"])[0].read_text(encoding="utf-8"))
                text = "\n".join("".join(c["source"]) for c in nb["cells"])
                self.assertIn('"--runner", "pooling", "--hf-overrides"', text)
                self.assertIn('API_KEY, "retrieval"]', text)

    def test_reranker_template_renders_the_prompt_qwen_trained_with(self) -> None:
        try:
            from jinja2 import Environment
        except ImportError:
            self.skipTest("jinja2 is not installed")
        # the same settings as transformers uses for chat templates (a last newline is dropped)
        template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(QWEN3_RERANKER_TEMPLATE)
        messages = [{"role": "query", "content": "Q?"}, {"role": "document", "content": "DOC"}]
        self.assertEqual(template.render(messages=messages, instruction="INS"),
                         QWEN3_RERANKER_PROMPT.format(instruction="INS", query="Q?", document="DOC"))
        self.assertIn("<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n",
                      template.render(messages=messages))  # the default instruction

    def test_vl_reranker_template_is_vllms_and_renders_qwens_prompt(self) -> None:
        self.assertEqual(hashlib.sha256(QWEN3_VL_RERANKER_TEMPLATE.encode()).hexdigest(), VL_TEMPLATE_SHA256)  # vLLM's file
        try:
            from jinja2 import Environment
        except ImportError:
            self.skipTest("jinja2 is not installed")
        template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(QWEN3_VL_RERANKER_TEMPLATE)
        image = "<|vision_start|><|image_pad|><|vision_end|>"  # what vLLM puts into the text for an image
        messages = [{"role": "query", "content": "Q?"}, {"role": "document", "content": image + "DOC"}]
        self.assertEqual(template.render(messages=messages, instruction="INS"),
                         QWEN3_VL_RERANKER_PROMPT.format(instruction="INS", query="Q?", document=image + "DOC"))
        self.assertIn("<Instruct>: Given a search query, retrieve relevant candidates that answer the query.<Query>:",
                      template.render(messages=messages))  # the default instruction

    def test_vl_reranker_serves_scoring_with_media_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nb = json.loads(generate(Path(tmp), ["qwen3_vl_reranker_2b"], ["colab"])[0].read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("Qwen3VLForSequenceClassification", text)
        self.assertIn('"--allowed-media-domains", "none.invalid"', text)  # the server never fetches links
        self.assertIn('"--chat-template", TEMPLATE_FILE', text)
        self.assertIn("qwen3_vl_reranker.jinja", text)
        self.assertIn("f.write(r\'\'\'" + QWEN3_VL_RERANKER_TEMPLATE + "\'\'\')", text)
        self.assertIn('API_KEY, "retrieval"]', text)
        self.assertNotIn("<think>", QWEN3_VL_RERANKER_TEMPLATE)  # unlike the text rerankers

    def test_list_mentions_every_model(self) -> None:
        text = "\n".join(list_models())
        for model in MODELS:
            self.assertIn(model["slug"], text)
        self.assertEqual(set(PLATFORMS), {"colab", "kaggle"})


if __name__ == "__main__":
    unittest.main()

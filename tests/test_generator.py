"""Test the notebook generator, no GPU needed.

    python -m unittest discover -s tests
"""

import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from nbgen.config import PLATFORMS  # noqa: E402
from nbgen.generate import generate, list_models  # noqa: E402
from nbgen.kinds import MODELS  # noqa: E402


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
        for target in (ROOT / "notebooks", ROOT / "notebooks" / "text", ROOT / "notebooks" / "new"):
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

    def test_list_mentions_every_model(self) -> None:
        text = "\n".join(list_models())
        for model in MODELS:
            self.assertIn(model["slug"], text)
        self.assertEqual(set(PLATFORMS), {"colab", "kaggle"})


if __name__ == "__main__":
    unittest.main()

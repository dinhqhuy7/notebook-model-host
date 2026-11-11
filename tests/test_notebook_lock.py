"""Guard the notebooks that have a recorded test run, no GPU needed.

    python -m unittest discover -s tests

A tested notebook must stay byte-for-byte what was run, so its measurements stay true. Changes go to a
separate `<name>_update.ipynb` (see tools/build_notebooks.py).
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_notebooks as nb  # noqa: E402


def notebooks() -> dict:
    return {p.stem: p for p in (ROOT / "notebooks").glob("*/*/*.ipynb")}


class NotebookLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = json.loads((ROOT / "notebooks.lock.json").read_text(encoding="utf-8"))["notebooks"]

    def test_locked_notebooks_are_unchanged(self) -> None:
        files = notebooks()
        for name, expected in self.lock.items():
            self.assertIn(name, files, f"{name}: locked notebook is missing")
            text = files[name].read_text(encoding="utf-8")
            actual = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.assertEqual(actual, expected, f"{name}: a tested notebook was modified; put changes in an update file")

    def test_every_tested_notebook_is_locked(self) -> None:
        for key, _, model in nb.all_notebooks():
            if (key, model["slug"]) in nb.TESTED:
                self.assertIn(nb.stem_of(key, model), self.lock,
                              f"{nb.stem_of(key, model)} has test results but is not frozen: --freeze it")

    def test_locked_notebooks_have_no_update_banner(self) -> None:
        for name, path in notebooks().items():
            if name in self.lock:
                ids = [c.get("id") for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]
                self.assertNotIn(nb.BANNER_ID, ids, f"{name}: a tested notebook must not be an update")

    def test_update_notebooks_are_marked_not_tested(self) -> None:
        for name, path in notebooks().items():
            if name.endswith(nb.UPDATE_SUFFIX):
                first = json.loads(path.read_text(encoding="utf-8"))["cells"][0]
                self.assertEqual(first.get("id"), nb.BANNER_ID, f"{name}: missing the 'not tested' banner")
                self.assertIn("not tested", "".join(first["source"]))
                self.assertNotIn(name, self.lock, f"{name}: an update must not be locked; promote it instead")
                self.assertIn(name[: -len(nb.UPDATE_SUFFIX)], notebooks(), f"{name}: no official notebook next to it")


if __name__ == "__main__":
    unittest.main()

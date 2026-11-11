"""Generate notebooks/<kind>/<model>/<platform>_<model>.ipynb from one template (every PLATFORMS × MODELS pair).

The code is in tools/nbgen/: kinds/<type>.py (models and recorded test runs of one type), config.py (constants,
types, platforms), template.py (the notebook template), readmes.py, lock.py and commands.py.

    python tools/build_notebooks.py

Also writes the READMEs: notebooks/README(.vi).md (every type and model), notebooks/<kind>/README(.vi).md (one
type: models, shared settings, GPU table) and notebooks/<kind>/<model>/README(.vi).md (one model: notebooks, GPU
table, test results). <kind> is the model's "kind" (see TYPES): text, vision-language...

Add a model: add an entry to MODELS in nbgen/kinds/<type>.py and rerun; the GPU table and minimum memory are computed. A model that
cannot run on a platform sets "platforms" (where it runs) and "unavailable" (reason, English and Vietnamese).
Record a test run: add an entry to TESTED in the same file, rerun, then freeze the notebook (below).

Tested notebooks are frozen. A notebook that has a recorded test run is listed in notebooks.lock.json with the
SHA-256 of its content, and this script never overwrites it. When the template would now produce different
content, the new content goes to <platform>_<slug>_update.ipynb instead, with a banner saying it is not tested.
The update lives next to the official notebook until it is tested; then promote it:

    python tools/build_notebooks.py                       build; locked notebooks are never overwritten
    python tools/build_notebooks.py --check               verify the locked notebooks are unchanged
    python tools/build_notebooks.py --freeze-tested       lock every notebook that has a TESTED entry
    python tools/build_notebooks.py --freeze NAME...      lock notebooks (NAME = colab_qwen3_8b)
    python tools/build_notebooks.py --promote NAME...     replace the official notebook by its tested update
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `nbgen` imports when run as a script

from nbgen.catalog import all_notebooks, stem_of  # noqa: E402
from nbgen.commands import build_all, freeze, promote  # noqa: E402
from nbgen.kinds import TESTED  # noqa: E402
from nbgen.lock import load_lock, locked_problems  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the notebooks and READMEs; tested notebooks are never overwritten.")
    parser.add_argument("--check", action="store_true", help="verify the locked notebooks are unchanged and exit")
    parser.add_argument("--freeze", nargs="+", metavar="NAME", help="lock notebooks that have a TESTED entry")
    parser.add_argument("--freeze-tested", action="store_true", help="lock every notebook that has a TESTED entry")
    parser.add_argument("--promote", nargs="+", metavar="NAME", help="replace a notebook by its tested update")
    args = parser.parse_args()
    if args.check:
        problems = locked_problems(load_lock())
        print("\n".join(problems) if problems else f"OK: {len(load_lock())} locked notebooks unchanged")
        sys.exit(1 if problems else 0)
    if args.freeze_tested:
        freeze([stem_of(k, m) for k, _, m in all_notebooks() if (k, m["slug"]) in TESTED])
        return
    if args.freeze:
        freeze(args.freeze)
        return
    if args.promote:
        promote(args.promote)
    build_all()


if __name__ == "__main__":
    main()

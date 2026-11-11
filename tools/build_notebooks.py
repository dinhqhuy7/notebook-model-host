"""Generate your own hosting notebooks: pick models and platforms, get ready-to-run .ipynb files.

    python tools/build_notebooks.py --list                          models, types, minimum GPU memory
    python tools/build_notebooks.py --model qwen3_8b --platform colab
    python tools/build_notebooks.py                                 every model on every platform
    python tools/build_notebooks.py --model qwen3_4b --out my_notebooks

Notebooks are written to ./generated (or --out), one file per model and platform:
<platform>_<model>.ipynb. Upload one to Colab or Kaggle and run all cells. The sample notebooks in notebooks/
are separate, tested files: this tool never writes there.

To add a model, add an entry to MODELS in nbgen/kinds/<type>.py (models of one type share a family of serving
flags there), then generate it. The code is in tools/nbgen/: config.py (versions, types, platforms), kinds/
(the models), template.py (the notebook template) and generate.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `nbgen` imports when run as a script

from nbgen.config import PLATFORMS  # noqa: E402
from nbgen.generate import generate, list_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hosting notebooks into your own folder (default: ./generated).")
    parser.add_argument("--list", action="store_true", help="list the models and exit")
    parser.add_argument("--model", nargs="+", metavar="SLUG", help="models to generate (see --list); default: all")
    parser.add_argument("--platform", nargs="+", choices=sorted(PLATFORMS), help="platforms; default: all")
    parser.add_argument("--out", default="generated", metavar="DIR", help="output folder (default: ./generated)")
    args = parser.parse_args()
    if args.list:
        print("\n".join(list_models()))
        return
    for path in generate(Path(args.out), args.model, args.platform):
        print("wrote", path)


if __name__ == "__main__":
    main()

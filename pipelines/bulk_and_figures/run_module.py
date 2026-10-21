#!/usr/bin/env python3
"""Run one configured source module; no analysis is launched unless requested."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("module", help="Relative .py path under source_tree")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source_tree = root / "source_tree"
    module = (source_tree / args.module).resolve()
    if not module.is_relative_to(source_tree) or module.suffix != ".py" or not module.is_file():
        parser.error("module must be an existing Python file within source_tree")
    os.environ["CRC_TCR_BULK_CONFIG"] = str(args.config.resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    # Child processes inherit the external configuration and runtime helper path.
    prior = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + prior if prior else "")
    sys.path[:0] = [str(module.parent), str(root)]
    sys.argv = [str(module)] + (args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    runpy.run_path(str(module), run_name="__main__")


if __name__ == "__main__":
    main()

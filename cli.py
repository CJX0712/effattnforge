"""EffAttnForge command-line entry point.

Delegates to the demo pipeline. Run from the repo root:

    python cli.py                 # default benchmark -> benchmark.json
    python cli.py --seed 7 --out out.json
"""
from __future__ import annotations

from examples.run_demo import main

if __name__ == "__main__":
    raise SystemExit(main())

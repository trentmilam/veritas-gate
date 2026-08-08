"""Score veritas-gate against the RAGTruth TRAIN split, for iteration.

Never the test split -- this is where detector tuning happens, freely, with no access-log
discipline, because nothing here is the number that gets published. Same report shape as
`benchmark/run.py`, so a new detector's numbers are directly comparable across scripts once it's
frozen and run once, disciplined, against test.

    python benchmark/fetch_data.py     # once, if not already done
    python benchmark/dev_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as _run  # noqa: E402


def main() -> int:
    responses, sources = _run.load()
    train = [r for r in responses if r.get("split") == "train"]
    out = _run.score(train, responses, sources)
    if out is None:
        return 2
    _run.print_report(out, title="TRAIN SPLIT (iteration only -- never published as-is)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

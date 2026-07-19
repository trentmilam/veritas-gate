"""Fetch the RAGTruth corpus used by the benchmark.

The corpus is NOT vendored into this repository: it is third-party data with its own license and
provenance, and it is 36 MB. Run this once before `benchmark/run.py`.

    python benchmark/fetch_data.py

Source: https://github.com/ParticleMedia/RAGTruth (Niu et al., "RAGTruth: A Hallucination Corpus
for Developing Trustworthy Retrieval-Augmented Language Models").
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ("response.jsonl", "source_info.jsonl")
DEST = Path(__file__).resolve().parent / "data"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists():
            print(f"  have {name} ({target.stat().st_size:,} bytes)")
            continue
        print(f"  fetching {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", target)
        print(f"  wrote {name} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch the RAGTruth corpus used by the benchmark.

The corpus is NOT vendored into this repository: it is third-party data with its own license and
provenance, and it is 36 MB. Run this once before `benchmark/run.py`.

    python benchmark/fetch_data.py

Source: https://github.com/ParticleMedia/RAGTruth (Niu et al., "RAGTruth: A Hallucination Corpus
for Developing Trustworthy Retrieval-Augmented Language Models").

PINNED, NOT "main". A benchmark whose entire claim is reproducibility cannot fetch a moving branch:
upstream could rename it, force-push it, or delete the repo, and "reproduce it" would silently
reproduce something else, or nothing. CORPUS_COMMIT below is an explicit commit SHA, confirmed via
`git ls-remote https://github.com/ParticleMedia/RAGTruth main` on 2026-08-08. CHECKSUMS are SHA-256
over the exact bytes served at that commit, computed the same day. Every fetch -- and every load, in
benchmark/run.py -- re-verifies against these, and ABORTS rather than proceeding on any mismatch: a
silently-swapped or corrupted corpus is worse than no corpus at all.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

CORPUS_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
BASE = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{CORPUS_COMMIT}/dataset"
FILES = ("response.jsonl", "source_info.jsonl")
DEST = Path(__file__).resolve().parent / "data"

# sha256 over the exact bytes at CORPUS_COMMIT, computed 2026-08-08. Bumping CORPUS_COMMIT is a
# deliberate, reviewable, one-line-plus-checksums diff -- never an invisible moving target.
CHECKSUMS = {
    "response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(name: str, path: Path) -> bool:
    """True if ``path`` exists and its sha256 matches the pinned checksum for ``name``. Does not
    fetch anything -- callers (including benchmark/run.py's ``load()``) use this to re-check a file
    that was already downloaded, so a swap or corruption after a successful fetch is still caught."""
    return name in CHECKSUMS and path.exists() and _sha256(path) == CHECKSUMS[name]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if not target.exists():
            print(f"  fetching {name} @ {CORPUS_COMMIT[:12]} ...", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", target)

        digest = _sha256(target)
        expected = CHECKSUMS[name]
        if digest != expected:
            target.unlink()
            print(f"ABORT: {name} checksum mismatch.\n"
                  f"  expected sha256:{expected}\n"
                  f"  got      sha256:{digest}\n"
                  f"Upstream RAGTruth at commit {CORPUS_COMMIT} appears to have changed, the "
                  f"download was corrupted, or something else is being served at that URL. This "
                  f"benchmark's entire claim is reproducibility -- a silently-swapped corpus would "
                  f"be worse than no corpus. The bad file has been deleted. Refusing to proceed.",
                  file=sys.stderr)
            return 1

        print(f"  verified {name} ({target.stat().st_size:,} bytes, sha256 matches commit "
              f"{CORPUS_COMMIT[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible demonstration of the claim registry's documented boundary hazard.

claim_rules.py's module docstring and the README cite a finding from 2026: matching a short,
common term as a bare substring (no word boundary) over a private 9,963-document corpus produced a
31.9x blast radius versus the equivalent word-boundary regex. That number is real but not
reproducible here -- the corpus is private, real personal application data, never vendored into
this public repo.

What IS reproducible is the mechanism. Substring-without-word-boundary over-firing inside longer
words is a general property of English text, not something specific to that one private corpus.
This script demonstrates the same shape of result on the public, pinned RAGTruth corpus this
benchmark already uses, so a reader can verify the claim class is real without needing access to
anything private.

    python benchmark/fetch_data.py                              # once
    python benchmark/registry_blast_radius.py                   # public mode (default)
    python benchmark/registry_blast_radius.py --root <path>      # private mode, opt-in only

PRIVATE MODE (--root): scans local files under the given directory instead of RAGTruth. Never the
default. Prints and returns AGGREGATE COUNTS ONLY -- no document text, no file paths, no snippets --
and refuses to run against this repository's own tree.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as _run  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULT_PATH = HERE / "registry_blast_radius.json"

# The same class of hazard the private-corpus finding measured: a short, common substring that sits
# inside many unrelated longer words. "rl" is a general property of English, not specific to any one
# corpus -- it appears inside world, girl, early, hourly, quarterly, curl, hurl, swirl, twirl...
DEFAULT_TERM = "rl"

_WORD_RE = re.compile(r"[A-Za-z']+")


def _boundary_pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def blast_radius(texts: list, term: str) -> dict:
    """Documents matched by a bare substring vs. the same term with a word boundary, over
    ``texts``. Mirrors exactly what a claim-registry ``phrases`` clause without word boundaries
    would fire on, versus what the equivalent boundary-aware regex fires on."""
    substring_re = re.compile(re.escape(term), re.IGNORECASE)
    boundary_re = _boundary_pattern(term)

    substring_docs = boundary_docs = 0
    containing_words: set = set()
    for text in texts:
        if substring_re.search(text):
            substring_docs += 1
        if boundary_re.search(text):
            boundary_docs += 1
        if len(containing_words) < 20:
            for w in _WORD_RE.findall(text):
                low = w.lower()
                if term in low and low != term:
                    containing_words.add(low)

    ratio = (substring_docs / boundary_docs) if boundary_docs else 0.0
    return {
        "term": term,
        "n_documents": len(texts),
        "substring_matched_documents": substring_docs,
        "word_boundary_matched_documents": boundary_docs,
        "blast_radius_ratio": ratio,
        "example_containing_words": sorted(containing_words),
    }


def _public_mode() -> dict:
    responses, _sources = _run.load()
    texts = [r["response"] for r in responses if r.get("split") == "test"]
    return blast_radius(texts, DEFAULT_TERM)


def _private_mode(root: Path, term: str) -> dict:
    resolved = root.resolve()
    # Block in BOTH directions: --root inside/equal to the repo, and --root an ANCESTOR of the repo
    # (which would walk straight into it via rglob) -- checking only the first direction lets
    # `--root <parent-of-this-repo>` sail through and scan this repo's own tree anyway.
    if (resolved == REPO_ROOT or REPO_ROOT in resolved.parents
            or resolved in REPO_ROOT.parents):
        print(f"BLOCKED: --root must not point inside, at, or above this repository ({REPO_ROOT}).")
        raise SystemExit(2)
    texts = []
    for path in root.rglob("*"):
        if path.is_file():
            try:
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    out = blast_radius(texts, term)
    # AGGREGATE COUNTS ONLY, per this module's own privacy guarantee: strip any field that carries
    # words/text extracted verbatim from the caller's private files before it ever leaves this
    # function -- returned to a caller AND printed by main(), so redacting only at print time would
    # still leak through the return value.
    out["example_containing_words"] = None
    return out


def _parse_args(argv: "list[str] | None") -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="Private mode: scan local files under this directory instead of the public "
                        "RAGTruth corpus. Never the default. Prints aggregate counts only.")
    p.add_argument("--term", default=DEFAULT_TERM,
                   help=f"Boundary-hazard term to test (default: '{DEFAULT_TERM}').")
    return p.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)

    if args.root:
        print("PRIVATE MODE -- aggregate counts only below, no document text or file paths.")
        out = _private_mode(Path(args.root), args.term)
    else:
        print("PUBLIC MODE -- RAGTruth test split (2,700 responses, the same pinned corpus "
              "benchmark/run.py uses).")
        out = _public_mode()
        # Committed like results.json/results_grounding.json, so the README's cited figures (and
        # TestRegistryBlastRadius) can be checked against a public-mode run without a live corpus
        # fetch on every CI run -- only public-mode output is ever persisted, never a --root scan.
        RESULT_PATH.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote benchmark/{RESULT_PATH.name}")

    print(json.dumps(out, indent=1))
    print()
    # example_containing_words is redacted (None) in private mode -- see _private_mode's own
    # AGGREGATE-COUNTS-ONLY comment. Never format it into the summary in that case.
    words = out["example_containing_words"]
    examples = f", purely from words like: {', '.join(words[:8])}" if words else ""
    if out["word_boundary_matched_documents"]:
        print(f"'{out['term']}' matched {out['substring_matched_documents']} documents as a bare "
              f"substring vs {out['word_boundary_matched_documents']} with a word boundary -- a "
              f"{out['blast_radius_ratio']:.1f}x blast radius{examples}.")
    elif out["substring_matched_documents"]:
        print(f"'{out['term']}' never occurs as its own word in this corpus (0 word-boundary "
              f"matches), yet a bare-substring registry rule would still fire on "
              f"{out['substring_matched_documents']} of {out['n_documents']} documents -- 100% "
              f"false positives{examples}.")
    else:
        print(f"'{out['term']}' does not occur in this corpus at all, substring or boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

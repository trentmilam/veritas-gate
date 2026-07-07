"""Repetition-runaway (degeneracy) detector for LLM-generated text."""
from __future__ import annotations

import re


def is_degenerate(
    text: str, *, min_tokens: int = 400, min_unique_ratio: float = 0.30, max_run: int = 30
) -> bool:
    """True if ``text`` is a local-LLM repetition runaway (S3): the Skills section that collapsed
    into ``AI futures, AI potentials, AI possibilities, …`` for thousands of tokens. The truth
    gate passes it (no false CLAIMS) and the length floor passes it (it's huge), so it needs its
    own check. Two cheap signals, either fires:
      (1) a long draft with a collapsed vocabulary — unique/total token ratio below the floor;
      (2) a long run of consecutive comma-separated items sharing the same leading word.
    Thresholds carry a wide margin: a real résumé's unique-token ratio is well above 0.30 and it
    never repeats a leading word 30× in a row (the broken artifact's run is in the hundreds)."""
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) >= min_tokens and len(set(toks)) / len(toks) < min_unique_ratio:
        return True
    run = best = 0
    prev = None
    for item in (s.strip() for s in (text or "").split(",")):
        head = item.split()[0].lower() if item.split() else ""
        run = run + 1 if head and head == prev else 1
        best = max(best, run)
        prev = head
    return best >= max_run

"""Grounding checks for the part of a hallucination that carries no digit.

TC-3 (unverified_metric) and TC-6 (unverified_count) in ``checker.py`` are digit matchers, and only
20.8% of RAGTruth's annotated hallucination spans contain a digit (``benchmark/`` measures this
directly) -- so those two checks structurally cannot see 79.2% of the problem. The three checks
here target that remainder with the same deterministic, no-LLM-judge approach: nothing here is a
semantic entailment check, all three are literal-token presence tests against the evidence text.

Each is opt-in and OFF by default when wired into ``TruthChecker`` (unlike TC-3/TC-6, which always
run) -- see the ``enable_broadened_numeric`` / ``enable_entity_grounding`` / ``enable_novelty_check``
constructor flags. Freeze any threshold/window change only via ``benchmark/tune.py``'s train-split
CV, then cite the winning number in the comment here -- never hand-tune against a number you've
already seen on the test split.
"""
from __future__ import annotations

import re

# Standard closed-class English function words -- excluded from "content word" status everywhere in
# this module. Deliberately NOT imported from checker.py's _digits_in / anything else there: this
# module has zero dependency on checker.py so checker.py can import FROM here without a circular
# import (checker.py wires these checks into TruthChecker.check()).
_STOPWORDS = frozenset("""
a an the and or but if then so because although though while when where who whom whose which
that this these those it its it's they them their theirs he him his she her hers we us our ours
you your yours i me my mine is are was were be been being am do does did will would should could
can may might must shall not no nor as at by for from in into of on onto out over under up down
about above below between among through during before after again further once here there all any
both each few more most other some such only own same than too very just also more most much many
per via vs etc
""".split())

# Sentence-opening words that are commonly capitalized purely by position ("The company shipped...",
# "In 2019, the team..."), not because they name an entity. A SUBSET of _STOPWORDS -- only words
# that plausibly open a sentence -- so a capitalized opener doesn't get flagged as a proper noun.
_SENTENCE_OPENERS = frozenset("""
the this that these those it in on at for with according additionally also after although and
as because before but by during either following from given however if moreover of once only or
since so some that their then there though thus to unless until upon we when where whereas while
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _numeric_tokens(text: str) -> set:
    """All numeric tokens in ``text`` (comma/space-stripped digit runs). Intentionally a standalone
    copy of checker.py's ``_digits_in`` rather than an import of it -- checker.py imports the checks
    in THIS module, so importing back would be circular. Keep the regex in sync if it ever changes."""
    return {re.sub(r"[,\s]", "", t) for t in re.findall(r"\d[\d.,]*", text or "")}


def broadened_numeric_ungrounded(draft_text: str, evidence_text: str) -> list:
    """Every digit token in ``draft_text`` absent from ``evidence_text`` -- the same digit matcher
    TC-3/TC-6 use, without their %/$/multiplier/count-noun restriction. Catches a bare fabricated
    figure ("in 2019 the team shipped 47 updates") neither existing check's pattern would match."""
    evidenced = _numeric_tokens(evidence_text)
    return sorted(t for t in _numeric_tokens(draft_text) if t not in evidenced)


def _content_words(text: str) -> set:
    """Lowercased, stopword-filtered word set of ``text`` -- the shared "did the evidence mention
    this word at all" index both the entity check and the novelty-window check test against."""
    return {w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS}


def ungrounded_entities(draft_text: str, evidence_text: str) -> list:
    """Capitalized tokens in ``draft_text`` that never appear (case-insensitively) in
    ``evidence_text`` -- a cheap proxy for a fabricated proper noun (person, place, product) the
    evidence never names. A sentence-initial token is excluded when it's a common sentence-opener
    (``_SENTENCE_OPENERS``), so ordinary capitalization from sentence position isn't mistaken for a
    name."""
    evidence_words = _content_words(evidence_text)
    out = []
    for sentence in _SENTENCE_SPLIT.split(draft_text or ""):
        for i, w in enumerate(_WORD_RE.findall(sentence)):
            if not w[0].isupper():
                continue
            low = w.lower()
            if low in _STOPWORDS:
                continue
            if i == 0 and low in _SENTENCE_OPENERS:
                continue
            if low not in evidence_words:
                out.append(w)
    return out


# Winner of benchmark/tune.py's train-split CV, measured 2026-08-08 over all 15,090 train
# responses: window=10, threshold=0.6 -> P 58.2/R 87.1/F1 69.8, beating every other grid point
# tried (window in 3/5/7/10, threshold in 0.6/0.8/1.0 -- see tune.py's printed table for the full
# grid). Change only by re-running tune.py and citing the new winning number here.
_NOVELTY_WINDOW = 10
_NOVELTY_THRESHOLD = 0.6


def novel_content_windows(draft_text: str, evidence_text: str, *, window: int = _NOVELTY_WINDOW,
                          threshold: float = _NOVELTY_THRESHOLD) -> list:
    """Content-word windows of ``draft_text`` whose novel-token ratio (tokens absent from
    ``evidence_text``'s content-word set) meets ``threshold``. Returns the offending window text,
    one entry per qualifying window (overlapping windows both reported, not deduplicated -- the
    caller decides how many violations that's worth)."""
    evidence_words = _content_words(evidence_text)
    tokens = [w.lower() for w in _WORD_RE.findall(draft_text or "") if w.lower() not in _STOPWORDS]
    if len(tokens) < window:
        return []
    out = []
    for i in range(len(tokens) - window + 1):
        span = tokens[i:i + window]
        novel = sum(1 for t in span if t not in evidence_words)
        if novel / window >= threshold:
            out.append(" ".join(span))
    return out

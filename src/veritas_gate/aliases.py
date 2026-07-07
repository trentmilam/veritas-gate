"""Curated domain alias/synonym matching (boundary- and synonym-aware term lookup).

The scorer, the curation gates, and the ATS fortifier all matched job text by raw
substring (``keyword in text``), which had two failure modes:

* **missed synonyms** — a JD that said "k8s" never matched the keyword "kubernetes",
  "retrieval-augmented generation" never matched "rag", "large language models" never
  matched "llm";
* **false hits** — a short keyword bled into a longer word: "engine" matched
  "engineer", "ms" matched "teams", "scala" matched "scalable".

This module fixes both with one small, deterministic, offline primitive:

* a curated ``canonical -> equivalent surface forms`` table for the AI/ML + infra +
  finance-engineering vocabulary this tool targets, and
* **boundary-safe** matching using word-char lookarounds (the same symbol-aware idiom the
  truth-checker uses), so a term matches a whole token/phrase, never a fragment.

It is intentionally NOT a fuzzy or embedding matcher: every equivalence here is
hand-picked and explainable (defensible to a recruiter). A real embedding model can later
widen recall via the an ``Embedder`` seam without touching this
precise, auditable core.

Public API:
    ``term_in(term, text) -> bool``        — is term (or any alias) present, boundary-safe?
    ``present_terms(terms, text) -> list`` — which of ``terms`` are present (order preserved)
    ``surface_forms(term) -> tuple``       — the full alias family for term (incl. itself)
    ``canonical_terms(text) -> set``       — the alias-table canonicals present in text
"""
from __future__ import annotations

import re
from typing import Iterable

# Each tuple is one equivalence family; the FIRST entry is the canonical label. Every form
# is matched on word boundaries, so the short ones are safe. Keep this conservative and
# auditable — only add a synonym you would defend as genuinely the same skill.
_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("kubernetes", "k8s"),
    ("machine learning", "ml"),
    ("llm", "llms", "large language model", "large language models"),
    ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    ("nlp", "natural language processing"),
    ("pytorch", "torch"),
    ("mlops", "ml ops", "ml-ops"),
    ("ci/cd", "cicd", "ci cd", "continuous integration",
     "continuous delivery", "continuous deployment"),
    ("vector database", "vector databases", "vector db", "vector store", "vectordb"),
    ("gpu", "gpus", "graphics processing unit"),
    ("fine-tuning", "fine tuning", "finetuning", "fine-tune", "finetune", "fine-tuned"),
    ("infrastructure as code", "iac"),
    ("distributed training", "multi-gpu training", "data-parallel", "model-parallel"),
    ("kafka", "apache kafka"),
    ("airflow", "apache airflow"),
    ("spark", "apache spark"),
    ("postgres", "postgresql"),
)

# Reverse + family indices, built once at import.
_TO_CANON: dict[str, str] = {}
_FAMILY: dict[str, tuple[str, ...]] = {}
for _fam in _FAMILIES:
    _canon = _fam[0]
    # If a later family reuses a canon, merge rather than clobber (keeps both sets of forms).
    merged = tuple(dict.fromkeys((*_FAMILY.get(_canon, ()), *_fam)))
    _FAMILY[_canon] = merged
    for _form in _fam:
        _TO_CANON.setdefault(_form, _canon)

_PAT_CACHE: dict[str, re.Pattern] = {}


def surface_forms(term: str) -> tuple[str, ...]:
    """The full alias family for ``term`` (including ``term`` itself), lowercased.

    Works whether ``term`` is a canonical label or one of its aliases; an unknown term
    returns just ``(term,)`` so every caller gets boundary-safe matching for free.
    """
    t = (term or "").strip().lower()
    if not t:
        return ()
    canon = _TO_CANON.get(t, t)
    return _FAMILY.get(canon, (t,))


def _pattern_for(term: str) -> re.Pattern | None:
    """Compiled, cached boundary-safe alternation over ``term``'s whole alias family."""
    t = (term or "").strip().lower()
    if not t:
        return None
    canon = _TO_CANON.get(t, t)
    pat = _PAT_CACHE.get(canon)
    if pat is None:
        forms = sorted(set(surface_forms(term)), key=len, reverse=True)
        # Internal spaces match ANY whitespace run (\s+), so a multi-word phrase still matches when
        # the JD/résumé wraps it across a line or uses double spaces (audit M3: 'large language\n
        # models' was missed). Hyphens/symbols inside a token are kept literal.
        alt = "|".join(r"\s+".join(re.escape(p) for p in f.split(" ")) for f in forms)
        # Word-char lookarounds (not \b) so symbol-bearing forms ("ci/cd", "c++", ".net",
        # "c#") still match — a trailing \b never asserts right after a symbol. The optional
        # trailing ``s?`` restores the plural leniency the old substring matcher had
        # (transformer↔transformers, pipeline↔pipelines) while the boundary still blocks
        # fragment bleed (engine↮engineers, scala↮scalable). No IGNORECASE: term_in matches
        # against a lowercased copy, so the pattern stays lowercase + cheap.
        pat = re.compile(rf"(?<!\w)(?:{alt})s?(?!\w)")
        _PAT_CACHE[canon] = pat
    return pat


def term_in(term: str, text: str) -> bool:
    """True when ``term`` — or any of its curated aliases — appears in ``text`` as a whole
    token/phrase (boundary-safe, case-insensitive). Fragments never match
    (``scala`` ∉ ``scalable``, ``engine`` ∉ ``engineer``, ``ms`` ∉ ``teams``)."""
    if not term or not text:
        return False
    forms = surface_forms(term)
    if not forms:
        return False
    # Match against a lowercased copy (skipped when already lower — the hot path from the
    # scorers, which lowercase upstream). The substring pre-check is a NECESSARY condition
    # for a boundary match, and `in` is C-fast, so it skips the regex for the ~90% of terms
    # that don't occur at all — ~5x faster on a full-corpus rescore, identical results.
    low = text if text.islower() else text.lower()
    # Pre-check on the FIRST token of each form — a necessary condition that survives line-wraps
    # (the full-phrase substring would not, now that internal spaces match \s+; audit M3).
    if not any(f.split(" ", 1)[0] in low for f in forms):
        return False
    pat = _pattern_for(term)
    return pat is not None and pat.search(low) is not None


def present_terms(terms: Iterable[str], text: str) -> list[str]:
    """The subset of ``terms`` present in ``text`` (input order preserved)."""
    if not text:
        return []
    return [t for t in terms if term_in(t, text)]


def canonical_terms(text: str) -> set[str]:
    """The alias-table canonicals present in ``text`` (e.g. a JD saying "k8s" yields
    ``{"kubernetes"}``). For matching against an arbitrary vocabulary use
    :func:`present_terms`."""
    if not text:
        return set()
    return {canon for canon in _FAMILY if term_in(canon, text)}

"""veritas-gate — deterministic truth-gating and quality scoring for LLM-generated text.

A fast, reproducible alternative to LLM-as-judge for production LLM pipelines:

- ``TruthChecker`` — traces every claim in a draft to an evidence bank and flags fabrication,
  forbidden/over-claim phrases, not-claimable-skill assertions (with an honest-omission
  whitelist), unverified impact metrics/counts, and optional misattribution — all by
  deterministic rules, no model call.
- ``rubric_score`` — a deterministic quality score (keyword alignment without stuffing, title/
  intent alignment, real-metric density, parseable structure) with no LLM judge.
- ``is_degenerate`` — catches repetition-runaway output that slips past both honesty checks and
  length floors.

Everything is candidate-/domain-agnostic: you supply the evidence and the vocabularies.
"""
from __future__ import annotations

from .aliases import surface_forms, term_in
from .checker import ClaimViolation, TruthChecker, TruthCheckResult
from .degeneracy import is_degenerate
from .rubric import ats_rubric_score as rubric_score
from .rubric import jd_keyword_gap, title_alignment_pct

__all__ = [
    "TruthChecker",
    "TruthCheckResult",
    "ClaimViolation",
    "rubric_score",
    "jd_keyword_gap",
    "title_alignment_pct",
    "is_degenerate",
    "term_in",
    "surface_forms",
]

__version__ = "0.1.0"

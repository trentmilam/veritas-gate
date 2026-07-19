"""veritas-gate — a deterministic, domain-configured gate for LLM-generated text.

Rules instead of a judge model: fast, reproducible, auditable. It is NOT a general hallucination
detector, and the benchmark in ``benchmark/`` measures exactly where that line falls — generic-only
F1 5.0% against a 51.8% always-positive floor, because the corpus-agnostic checks are digit matchers
and only 20.8% of RAGTruth's annotated spans contain a digit. Speed and determinism do hold. Read
the README's benchmark section before reaching for this as semantic verification.

- ``TruthChecker`` — enforces the constraints you configure against an evidence bank: forbidden and
  over-claim phrases, not-claimable-skill assertions (with an honest-omission whitelist), impact
  metrics and counts absent from the evidence, optional misattribution, credentials and named
  entities. It does not parse arbitrary prose into claims and verify each one.
- ``rubric_score`` — a deterministic quality score (keyword alignment without stuffing, title/
  intent alignment, real-metric density, parseable structure) with no LLM judge.
- ``is_degenerate`` — catches repetition-runaway output that slips past both honesty checks and
  length floors.
- ``check_claim_rules`` / ``prompt_rules_block`` — a declarative claim registry where one row is
  both the rule the gate enforces and the instruction the prompt renders, so the two cannot drift.
  Its docstring documents, with measurements, which rules must stay in code instead.

Everything is domain-agnostic: you supply the evidence, the vocabularies, and the rules.
"""
from __future__ import annotations

from .aliases import surface_forms, term_in
from .checker import ClaimViolation, TruthChecker, TruthCheckResult
from .claim_rules import RuleFinding, check_claim_rules, load_rules, prompt_rules_block
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
    "check_claim_rules",
    "prompt_rules_block",
    "load_rules",
    "RuleFinding",
    "term_in",
    "surface_forms",
]

__version__ = "0.1.0"

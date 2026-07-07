"""Regression tests for the 2026-07-01 audit truth-gate bugs (Claude-690c/5fvs/6qbu/s1am/5kyc/0q4l).

Each test encodes the EXACT failing input recorded in the bead so a regression re-fires it.
"""
from __future__ import annotations

from veritas_gate import TruthChecker
from veritas_gate.checker import TruthChecker as _TC


def _highs(gate: TruthChecker, draft: str) -> list:
    return [v for v in gate.check(draft).violations if v.severity == "high"]


# ---------------------------------------------------------------------------
# Claude-690c: an affirmative forbidden-skill claim sharing a period-clause with a broad omission
# marker must NOT be whitelisted. Comma-split + affirmative-competence guard.
# ---------------------------------------------------------------------------
def test_690c_affirmative_skill_with_aspirational_marker_is_blocked() -> None:
    g = TruthChecker(forbidden_skills=["kubernetes", "fine-tuning"])
    # Every one of these used to pass with ZERO violations (the whitelist bypass).
    assert _highs(g, "Kubernetes expert, plan to learn even more.")
    assert _highs(g, "Expert in Kubernetes and eager to learn more.")
    assert _highs(g, "Extensive fine-tuning work, not relevant to prove here.")
    assert _highs(g, "I do not have gaps in my Kubernetes expertise.")


def test_690c_genuine_skill_omission_still_passes() -> None:
    g = TruthChecker(forbidden_skills=["kubernetes", "fine-tuning"])
    # Honest disclosures the whitelist is FOR must still pass.
    assert not _highs(g, "I have no hands-on experience with Kubernetes or fine-tuning.")
    assert not _highs(g, "Kubernetes: omitted -- no verified experience.")
    assert not _highs(g, "EVIDENCE FLAGS: kubernetes not in evidence.")


def test_690c_affirmative_then_separate_omission_sentence_still_blocked() -> None:
    # Pre-existing HIGH guard: an affirmative claim must not be whitelisted by an omission in a
    # different sentence on the same line.
    g = TruthChecker(forbidden_skills=["kubernetes"])
    assert _highs(g, "expert in Kubernetes. I have yet to work on mobile.")


# ---------------------------------------------------------------------------
# Claude-5fvs: an honest aspirational/omission credential disclosure must NOT hard-fail the draft.
# ---------------------------------------------------------------------------
def test_5fvs_honest_credential_disclosure_passes() -> None:
    g = TruthChecker(experience_evidence="Built a RAG system")
    assert not _highs(g, "I do not have a CPA or CFA.")
    assert not _highs(g, "Currently studying for the CFA exam.")
    assert not _highs(g, "No verified PMP certification.")


def test_5fvs_affirmative_unevidenced_credential_still_blocked() -> None:
    # The gate must still catch an affirmatively-CLAIMED credential not in the evidence.
    g = TruthChecker(experience_evidence="Built a RAG system")
    assert _highs(g, "Licensed CPA and CFA charterholder.")
    assert _highs(g, "I am a Certified Public Accountant.")


# ---------------------------------------------------------------------------
# Claude-6qbu: hyphen/dash/whitespace variants of a forbidden phrase must not evade the gate.
# ---------------------------------------------------------------------------
def test_6qbu_hyphen_and_dash_variants_are_blocked() -> None:
    g = TruthChecker(forbidden_claims="cloud-native\nfull-stack")
    assert _highs(g, "Building cloud native platforms.")       # space instead of hyphen
    assert _highs(g, "Skilled in cloud‑native design.")   # non-breaking hyphen U+2011
    assert _highs(g, "A cloud–native architecture.")      # en-dash
    assert _highs(g, "A cloud—native architecture.")      # em-dash
    assert _highs(g, "Seasoned engineer building cloud-native platforms.")  # exact


# ---------------------------------------------------------------------------
# Claude-s1am: forbidden phrase must match a whole token/phrase, not a fragment of a longer word.
# ---------------------------------------------------------------------------
def test_s1am_forbidden_phrase_is_boundary_aware() -> None:
    g = TruthChecker(forbidden_claims="full-stack")
    assert not _highs(g, "I read full-stackoverflow daily.")   # substring, must NOT fire
    assert _highs(g, "I am a full-stack engineer.")            # whole phrase, must fire


# ---------------------------------------------------------------------------
# Claude-5kyc: short (<=3 char) forbidden phrases must still enforce (they are caller-curated).
# ---------------------------------------------------------------------------
def test_5kyc_short_forbidden_phrases_fire() -> None:
    g = TruthChecker(forbidden_claims="PhD\nSQL\nGPT")
    assert _highs(g, "I hold a PhD in physics.")
    assert _highs(g, "Expert in SQL.")
    assert _highs(g, "Trained a GPT model.")


def test_5kyc_short_forbidden_phrase_still_boundary_aware() -> None:
    # Dropping the length guard must not create fragment false-positives.
    g = TruthChecker(forbidden_claims="SQL")
    assert not _highs(g, "I use MySQL and PostgreSQL databases.")


# ---------------------------------------------------------------------------
# Claude-0q4l: a claim beginning with a decimal must not have its leading 'N.' stripped.
# ---------------------------------------------------------------------------
def test_0q4l_leading_decimal_is_not_mangled() -> None:
    assert _TC()._split_claims("3.5 Kubernetes clusters managed") == ["3.5 Kubernetes clusters managed"]
    assert _TC()._split_claims("3.5% growth in revenue") == ["3.5% growth in revenue"]


def test_0q4l_violation_claim_preserves_full_decimal() -> None:
    g = TruthChecker(forbidden_skills=["kubernetes"])
    res = g.check("3.5 Kubernetes clusters managed")
    # The reported violation claim must be the intact line, not '5 Kubernetes clusters managed'.
    assert any(v.claim == "3.5 Kubernetes clusters managed" for v in res.violations)


def test_0q4l_real_list_marker_is_still_stripped() -> None:
    # A genuine numbered/bulleted list index must still be removed.
    assert _TC()._split_claims("3. Built the RAG pipeline") == ["Built the RAG pipeline"]
    assert _TC()._split_claims("1. First item") == ["First item"]

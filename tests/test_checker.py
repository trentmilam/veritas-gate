"""TruthChecker — deterministic claim-verification gating, on generic example data."""
from __future__ import annotations

from veritas_gate import TruthChecker

EVIDENCE = (
    "Built a retrieval-augmented generation system over a 100,000+ document corpus that cut "
    "research time 85%. Stood up local model serving with llama.cpp. Wrote 445 automated tests."
)


def _gate() -> TruthChecker:
    return TruthChecker(
        experience_evidence=EVIDENCE,
        forbidden_skills=["kubernetes", "lora", "fine-tuning", "rlhf"],
        forbidden_claims="cloud-native\nfull-stack\njailbreak classifier",
    )


def _highs(draft: str) -> list:
    return [v for v in _gate().check(draft).violations if v.severity == "high"]


def test_honest_evidence_backed_claim_passes() -> None:
    assert _gate().check("Built a RAG pipeline over a 100,000+ document corpus.").is_valid


def test_not_claimable_skill_is_blocked() -> None:
    assert _highs("Expert in Kubernetes with deep LoRA fine-tuning experience.")


def test_honest_omission_of_a_skill_passes() -> None:
    # The omission-disclosure whitelist: naming a not-claimable skill inside an honest
    # "no experience with X" sentence is a disclosure, not a claim.
    assert not _highs("I have no hands-on experience with Kubernetes or fine-tuning.")


def test_forbidden_phrase_is_blocked() -> None:
    assert _highs("Seasoned engineer building cloud-native platforms.")
    assert _highs("Built a jailbreak classifier for the platform.")


def test_unverified_metric_is_flagged() -> None:
    res = _gate().check("Improved throughput by 4200% across the fleet.")
    assert any(v.violation_type == "unverified_metric" for v in res.violations)


def test_attribution_is_inert_without_markers() -> None:
    # Default: no employer/self-project markers supplied -> the attribution check never fires.
    res = _gate().check("ACME CORP\n- Built my own side-project inference cluster at home")
    assert not any(v.violation_type == "misattribution" for v in res.violations)


def test_attribution_fires_when_markers_configured() -> None:
    gate = TruthChecker(
        experience_evidence=EVIDENCE,
        employer_markers=("acme corp",),
        self_project_markers=("home inference cluster",),
    )
    draft = "ACME CORP - Senior Engineer\n- Built a home inference cluster on nights and weekends"
    res = gate.check(draft)
    assert any(v.violation_type == "misattribution" and v.severity == "high"
               for v in res.violations)

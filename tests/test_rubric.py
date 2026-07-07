"""rubric_score — deterministic, no-LLM-judge quality scoring, on generic data."""
from __future__ import annotations

from veritas_gate import rubric_score, title_alignment_pct

POSTING = {"title": "Forward Deployed AI Engineer", "description": "python rag llm inference docker"}
KW = frozenset({"python", "rag", "llm", "inference", "docker"})
CLEAN = (
    "Alex Doe\nForward Deployed AI Engineer\n\n"
    "Summary\nApplied AI engineer.\n"
    "Experience\n"
    "- Cut research time 85% over a 100,000-document corpus\n"
    "- Served 5M inference requests/day on a Docker deployment\n"
    "Skills\npython, rag, llm, inference, docker\nEducation\nBS"
)


def test_title_alignment_rewarded() -> None:
    assert title_alignment_pct(CLEAN, POSTING) == 100
    assert title_alignment_pct("Alex Doe\n\nSummary\nx", POSTING) == 0


def test_clean_resume_scores_well_with_no_stuffing_penalty() -> None:
    s = rubric_score(CLEAN, POSTING, candidate_keywords=KW)
    assert s["stuffing_penalty"] == 0
    assert s["title_pct"] == 100
    assert s["score"] >= 80


def test_keyword_cramming_is_penalized() -> None:
    crammed = ("Alex\nGuy\n\nSummary\nx\nExperience\n- " + ("rag " * 16) + "and "
               + ("python " * 8) + "work\nSkills\npython\nEducation\nBS")
    clean = rubric_score(CLEAN, POSTING, candidate_keywords=KW)["score"]
    cram = rubric_score(crammed, POSTING, candidate_keywords=KW)
    assert cram["stuffing_penalty"] > 0
    assert cram["score"] < clean


def test_quantified_requires_real_impact_metric() -> None:
    bare = "Summary\nx\nExperience\n- Worked with Python 3.11 for 3 years on the 2024 team\nSkills\npython\nEducation\nBS"
    impact = "Summary\nx\nExperience\n- Cut latency 85% across a 28,701-row pipeline\nSkills\npython\nEducation\nBS"
    assert rubric_score(bare, POSTING, candidate_keywords=KW)["quant_pct"] == 0
    assert rubric_score(impact, POSTING, candidate_keywords=KW)["quant_pct"] == 100

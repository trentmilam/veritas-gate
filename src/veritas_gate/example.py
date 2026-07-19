"""Runnable 30-second demo on fake data — `python -m veritas_gate.example`.

Shows the three gates working together with ZERO private data: a fake evidence bank, a fake job
description, one honest draft and one fabricated draft.
"""
from __future__ import annotations

from . import TruthChecker, is_degenerate, rubric_score

# A fake "candidate" evidence bank — what the subject can genuinely back.
EVIDENCE = (
    "Built a retrieval-augmented generation system over a 35,000+ document corpus that cut "
    "research time 85%. Stood up local model serving with llama.cpp and an OpenAI-compatible "
    "gateway. Wrote 445 automated tests."
)
FORBIDDEN_SKILLS = ["kubernetes", "lora", "fine-tuning", "rlhf"]  # things the subject has NOT done
POSTING = {"title": "Forward Deployed AI Engineer", "description": "python rag llm inference docker"}
KEYWORDS = frozenset({"python", "rag", "llm", "inference", "docker", "llama.cpp"})

HONEST = (
    "Alex Doe\nForward Deployed AI Engineer\n\n"
    "Summary\nApplied AI engineer who ships tested, reliable systems.\n"
    "Experience\n- Cut research time 85% over a 35,000-document RAG corpus\n"
    "- Stood up local LLM serving with llama.cpp behind an OpenAI-compatible gateway\n"
    "Skills\npython, rag, llm, inference, docker\nEducation\nBS"
)
FABRICATED = (
    "Alex Doe\nSenior Engineer\n\nSummary\nSeasoned engineer building cloud-native platforms.\n"
    "Experience\n- Expert in Kubernetes and LoRA fine-tuning of production models\n"
    "- Improved throughput 4200% across the fleet\nSkills\nkubernetes, lora\nEducation\nBS"
)


def _report(label: str, draft: str) -> None:
    gate = TruthChecker(experience_evidence=EVIDENCE, forbidden_skills=FORBIDDEN_SKILLS,
                        forbidden_claims="cloud-native\nfull-stack")
    res = gate.check(draft)
    score = rubric_score(draft, POSTING, candidate_keywords=KEYWORDS)
    print(f"\n=== {label} ===")
    print(f"  truth: valid={res.is_valid}  high={sum(1 for v in res.violations if v.severity=='high')}"
          f"  total={len(res.violations)}")
    for v in res.violations[:4]:
        print(f"    [{v.severity}] {v.violation_type}: {v.suggestion[:64]}")
    print(f"  rubric: score={score['score']}  title%={score['title_pct']}  "
          f"quant%={score['quant_pct']}  stuffing={score['stuffing_penalty']}")
    print(f"  degenerate: {is_degenerate(draft)}")


def main() -> None:
    _report("HONEST draft", HONEST)
    _report("FABRICATED draft", FABRICATED)


if __name__ == "__main__":
    main()

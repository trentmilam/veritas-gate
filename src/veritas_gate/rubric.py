"""ATS / rubric-aware resume fortification (truth-preserving).

Grades a tailored resume the way applications are actually screened — ATS keyword
coverage, quantification (STAR-style metrics), and ATS-safe formatting — and surfaces the
TRUTHFUL missing keywords the candidate's profile already supports. It never invents
claims: JD terms the profile cannot back are returned as ``qa_gaps`` (questions for the QA
session), NOT as resume suggestions. The hard anti-fabrication gate stays the
the truth checker; this layer is the offensive, evidence-bounded optimizer.

Research basis (2026): ATS (97.8% of F500, 99.7% keyword filters) reward ~85% JD keyword
match but penalize stuffing (~30%); Greenhouse-style scorecards grade must-have competencies;
Eightfold-style screeners do transferable-skill matching; HireVue/STAR reward quantified bullets.
"""
from __future__ import annotations

import re
from typing import Optional

from .aliases import surface_forms, term_in
# Generic gap vocabulary: tech/skill terms a JD commonly REQUIRES. In the original tool this was
# candidate-derived; here it is a self-contained default so the library has zero private-data deps.
# Callers pass their own candidate_keywords (what the subject genuinely has); a None default
# yields an empty set (score then reflects gap-vocab coverage only).
_GAP_VOCAB = (
    "kubernetes", "terraform", "spark", "kafka", "airflow", "snowflake", "react",
    "typescript", "scala", "golang", "kotlin", "tensorflow", "jax", "triton", "vllm",
    "kubeflow", "mlflow", "sagemaker", "vertex ai", "bigquery", "graphql", "grpc",
    "microservices", "feature store", "reinforcement learning", "computer vision",
    "security clearance", "phd", "5+ years", "10+ years", "on-call",
)


def _candidate_keywords() -> frozenset:
    """Library default: no built-in candidate keywords -- pass them explicitly."""
    return frozenset()

_STD_HEADERS = ("summary", "experience", "skills", "education")
_DIGIT_RE = re.compile(r"\d")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
# Gap-vocab entries that are REQUIREMENTS / anti-signals, not skills to MIRROR. Keeping them in the
# coverage pool let a résumé earn keyword credit for echoing "PhD" / "on-call" / "5+ years" — worse
# than useless (mirroring a constraint you don't meet). They still surface as gaps via curate; they
# just must not inflate keyword coverage here (hostile-critique F4).
_NON_SKILL_GAP = {"security clearance", "phd", "5+ years", "10+ years", "on-call"}
# Seniority/filler tokens stripped from a JD title before measuring headline alignment — we reward
# mirroring the ROLE ("ml infra engineer"), never the seniority claim ("senior"/"staff").
_TITLE_STOP = {
    "the", "a", "an", "of", "and", "for", "to", "in", "ii", "iii", "sr", "jr",
    "senior", "staff", "lead", "principal", "junior", "i",
}
# A bullet is QUANTIFIED only if it carries a real IMPACT figure — a percentage, a dollar amount, an
# N× multiplier, a comma-grouped thousand, or a k/m/b magnitude. Bare integers ("3 years",
# "5-person team"), versions ("Python 3.11") and years ("2024") are NOT impact metrics; the old
# "any digit" rule (hostile-critique F3) rewarded number-stuffing without rewarding real results.
_IMPACT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%"            # 85%
    r"|\$\s*\d"                     # $2 / $2M
    r"|\d+(?:\.\d+)?\s*[x×]\b"      # 3x / 2.5×
    r"|\b\d{1,3}(?:,\d{3})+\b"      # 28,701
    r"|\b\d+(?:\.\d+)?\s*[kmb]\b",  # 5M / 100k / 1.2b
    re.IGNORECASE,
)
# A markdown table's defining delimiter row, e.g. ``| --- | :--: |`` — the thing that makes a
# renderer/ATS treat the block as a column grid (vs. a lone "City | Phone" inline separator).
_MD_TABLE_SEP_RE = re.compile(r"\|?\s*:?-{3,}:?\s*\|")


def _jd_terms(jd: str, cand: frozenset) -> set:
    """Relevant SKILL terms appearing in the JD — the candidate's own keywords (things they have)
    plus the skill-bearing gap vocabulary (things a JD commonly requires). Requirement/anti-signal
    gap entries (PhD / on-call / N+ years / clearance) are excluded so they can't earn coverage
    credit for being mirrored (hostile-critique F4)."""
    pool = set(cand) | ({g.strip() for g in _GAP_VOCAB} - _NON_SKILL_GAP)
    return {t for t in pool if t and term_in(t, jd)}


def title_alignment_pct(resume_text: str, posting: dict) -> int:
    """0-100: does the résumé's headline zone (first ~3 non-empty lines = name + positioning
    headline) mirror the JD title's significant tokens? Title/headline alignment is the single
    highest-weight real ATS signal (Workday), so the rubric rewards a TRUTHFUL title-mirroring
    headline (hostile-critique F2). Seniority words are stripped (``_TITLE_STOP``) — we reward
    matching the role, never the seniority claim. Synonym-aware via ``aliases.term_in``."""
    toks = {w for w in re.findall(r"[a-z][a-z+/#.\-]{1,}", (posting.get("title") or "").lower())
            if w not in _TITLE_STOP}
    if not toks:
        return 0
    head_lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()][:3]
    head = " ".join(head_lines).lower()
    hit = sum(1 for w in toks if term_in(w, head))
    return round(100 * hit / len(toks))


def _is_header(line: str, header: str) -> bool:
    """True if ``line`` is a standalone section header for ``header`` (case-insensitive, tolerant
    of markdown decoration like ``## Experience`` / ``**EXPERIENCE**`` / ``Experience:``)."""
    return line.strip().strip("#* ").rstrip(":").strip().lower() == header


def _experience_bullets(resume_text: str) -> list[str]:
    """Bullet lines inside the EXPERIENCE section only — delimited by the EXPERIENCE header and the
    next standard section header (SKILLS / SUMMARY / EDUCATION). A SKILLS list is bullets too, and
    skills lines never carry numbers, so counting them unfairly deflates the quantified-bullets
    ratio. Falls back to ALL bullets when the resume has no recognizable EXPERIENCE header."""
    lines = resume_text.splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if _is_header(ln, "experience")), None)
    if start is None:
        return [ln for ln in lines if _BULLET_RE.match(ln)]
    others = [h for h in _STD_HEADERS if h != "experience"]
    end = next((j for j in range(start, len(lines))
                if any(_is_header(lines[j], h) for h in others)), len(lines))
    return [ln for ln in lines[start:end] if _BULLET_RE.match(ln)]


def leads_with_metric_pct(resume_text: str) -> int:
    """ADVISORY (non-blocking): the fraction of EXPERIENCE bullets that OPEN with a number, as a
    0-100 percent. A signal for the research lever "lead the bullet with the evidenced production
    metric where one genuinely exists" — high when strong bullets front-load their real figure.

    Pure measurement, never a gate: it reports what the draft does and CANNOT (and must not) cause
    a number to be invented — the anti-fabrication gate stays the truth_checker. Reuses
    ``_experience_bullets`` + ``_DIGIT_RE`` so it agrees with :func:`ats_rubric_score`'s
    quantified ratio (which counts a digit ANYWHERE in the bullet; this one only counts a digit in
    the LEADING token — i.e. the bullet actually starts with the number)."""
    bullets = _experience_bullets(resume_text or "")
    if not bullets:
        return 0
    leads = 0
    for b in bullets:
        body = _BULLET_RE.sub("", b).strip()      # strip the "- "/"* "/"• " marker
        head = body.split(maxsplit=1)[0] if body else ""
        if _DIGIT_RE.search(head):
            leads += 1
    return round(100 * leads / len(bullets))


def ats_rubric_score(
    resume_text: str,
    posting: dict,
    candidate_keywords: Optional[frozenset] = None,
) -> dict:
    """Score a resume draft against the posting. Returns an application-strength dict:
    ``score`` (0-100), ``keyword_pct`` + ``present`` + ``add_truthful`` (profile-supported
    terms the resume omitted — safe to add) + ``qa_gaps`` (JD wants, profile can't back —
    ask in QA, never invent), ``quant_pct``, and ``format_flags``/``format_score``."""
    cand = candidate_keywords if candidate_keywords is not None else _candidate_keywords()
    jd = f" {(posting.get('title') or '')} {(posting.get('description') or '')} ".lower()
    rt = (resume_text or "").lower()

    terms = _jd_terms(jd, cand)
    # Synonym-aware résumé coverage: a résumé saying "k8s" covers the JD term "kubernetes".
    present = sorted(t for t in terms if term_in(t, rt))
    missing = [t for t in terms if not term_in(t, rt)]
    add_truthful = sorted(t for t in missing if t in cand)        # candidate HAS it -> add it
    qa_gaps = sorted(t for t in missing if t not in cand)         # candidate lacks it -> QA
    kw_pct = round(100 * len(present) / len(terms)) if terms else 0

    bullets = [ln for ln in (resume_text or "").splitlines() if _BULLET_RE.match(ln)]
    # Quantified-bullets ratio is over EXPERIENCE bullets only — a SKILLS list never carries
    # numbers, so including it would unfairly drag the percentage down (and the score with it).
    exp_bullets = _experience_bullets(resume_text or "")
    # Quantified = carries a real IMPACT metric (%, $, ×, thousands, k/m/b) — NOT just "any digit"
    # (which rewarded years / versions / "3 years"; hostile-critique F3).
    quant_pct = (
        round(100 * sum(1 for b in exp_bullets if _IMPACT_RE.search(b)) / len(exp_bullets))
        if exp_bullets else 0
    )

    rlines = (resume_text or "").splitlines()
    fmt_flags: list[str] = []
    # Count REAL standalone section headers (``_is_header``), not the substring "experience"
    # appearing inside prose like "5 years of experience" (hostile-critique F8).
    headers_found = sum(1 for h in _STD_HEADERS if any(_is_header(ln, h) for ln in rlines))
    if headers_found < 3:
        fmt_flags.append("add standard section headers (Summary / Experience / Skills / Education)")
    # Real ATS-hostile layout = a multi-COLUMN grid (the parser reads across columns and scrambles
    # field order), NOT an inline "City | Phone | Email" separator on one linear text line (ATS-safe).
    # Flag only an HTML table, a markdown delimiter row, or 2+ CONSECUTIVE rows each carrying 2+ pipes
    # (a real column grid / pipe-wall). A lone separator line must not trip this — penalizing it
    # docked every clean résumé 25 pts for a non-problem and made the strength score untrustworthy.
    _pipey = [ln.count("|") >= 2 for ln in rlines]
    _pipe_grid = any(a and b for a, b in zip(_pipey, _pipey[1:]))
    _md_table = any(_MD_TABLE_SEP_RE.search(ln) for ln in rlines)
    if "<table" in rt or _md_table or _pipe_grid:
        fmt_flags.append("avoid tables/pipe layouts — ATS may scramble them")
    if not bullets:
        fmt_flags.append("use bullet points for achievements")
    fmt_score = max(0, 100 - 25 * len(fmt_flags))

    # Title/headline alignment — the highest-weight REAL ATS signal (hostile-critique F2).
    title_pct = title_alignment_pct(resume_text, posting)

    # Anti-stuffing (hostile-critique F1): coverage rewards breadth; this opposes REPETITION.
    # Research: keyword stuffing REDUCES outcomes ~30%. The honest signal is a single term repeated
    # many times — NOT aggregate keyword density (a concise skills list is legitimately keyword-dense,
    # so an aggregate-density penalty wrongly docks clean résumés). A central skill ("python") may
    # appear a handful of times; only the EXCESS above 6 occurrences of any one term is penalized,
    # soft-ramped and capped, so a clean draft pays ~0 and only genuine cramming/runaway is docked.
    over = 0
    for t in present:
        c = sum(len(re.findall(rf"(?<!\w){re.escape(f)}(?!\w)", rt)) for f in surface_forms(t))
        if c > 6:
            over += c - 6
    stuffing_penalty = min(30.0, over * 2.0)

    # Rebalanced onto the evidence-based levers: keyword coverage loses its monopoly, title
    # alignment carries real weight, quantification + parseable format round it out, minus stuffing.
    score = round(max(0.0, min(100.0,
        0.35 * kw_pct + 0.25 * title_pct + 0.20 * quant_pct + 0.20 * fmt_score
        - stuffing_penalty)))
    return {
        "score": score,
        "keyword_pct": kw_pct,
        "title_pct": title_pct,
        "present": present,
        "add_truthful": add_truthful,
        "qa_gaps": qa_gaps,
        "quant_pct": quant_pct,
        "format_flags": fmt_flags,
        "format_score": fmt_score,
        "stuffing_penalty": round(stuffing_penalty),
    }


def jd_keyword_gap(posting: dict, candidate_keywords: Optional[frozenset] = None) -> dict:
    """JD-vs-candidate keyword gap — **no resume needed** (vs. :func:`ats_rubric_score`, which grades
    an existing draft). Decided the single highest ATS lever in the 2026 market research: postings get
    filtered for naming the JD's exact tool terms. Surfaced at the apply decision so the candidate
    knows, before tailoring, which terms to MIRROR verbatim (JD asks AND evidence backs it) and which
    are GAPS (JD asks, no evidence — never fake). Reuses the same ``_jd_terms`` + candidate-keyword
    primitives as :func:`ats_rubric_score`, so the cockpit gap view and the résumé scorer agree.

    Returns ``mirror`` (put these in the résumé), ``gaps`` (don't fake — QA/learn), ``coverage``
    (% of relevant JD terms the candidate can truthfully back), and ``jd_term_count``.
    """
    cand = candidate_keywords if candidate_keywords is not None else _candidate_keywords()
    jd = f" {(posting.get('title') or '')} {(posting.get('description') or '')} ".lower()
    terms = _jd_terms(jd, cand)
    mirror = sorted(t for t in terms if t in cand)      # JD wants it AND you can back it -> mirror
    gaps = sorted(t for t in terms if t not in cand)    # JD wants it, no evidence -> don't fake
    coverage = round(100 * len(mirror) / len(terms)) if terms else 0
    return {"mirror": mirror, "gaps": gaps, "coverage": coverage, "jd_term_count": len(terms)}

"""Truth checker: validates draft claims against candidate evidence.

Inputs:
- candidate profile text or YAML
- experience evidence bank
- resume blocks
- forbidden claims
- generated draft text
- selected job description (context only, cannot prove claims)

job_requirements may be used as context but cannot prove a user claim is true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Professional credentials the checker flags when asserted-but-unevidenced. The index builder and
# the draft checker MUST use this same set, or a real credential is recognized by one and not the
# other (which silently withholds a truthful résumé).
_CREDENTIAL_TOKENS = "CFA|PMP|FRM|CGA|CEP|CPA|CISA|CISSP"

# Spelled-out credential names mapped to their core token, so a fabricated credential written out
# in full ('Certified Public Accountant') is caught too — not only the abbreviation (audit HIGH).
_CREDENTIAL_FULL = {
    "cfa": "chartered financial analyst",
    "cpa": "certified public accountant",
    "pmp": "project management professional",
    "frm": "financial risk manager",
    "cisa": "certified information systems auditor",
    "cissp": "certified information systems security professional",
    "cga": "certified general accountant",
    "cep": "certified equity professional",
}

# Phrases that mark an HONEST omission/disclosure footnote — the model disclosing what it
# intentionally left OUT for lack of evidence ("Kubernetes: omitted — no verified experience",
# "EVIDENCE FLAGS: AWS/GCP not in evidence"). Naming a not-claimable skill INSIDE such a line is a
# truthful disclosure, NOT an affirmative claim, so the forbidden-skill gate must skip it. Without
# this, the gate hard-blocks honest drafts (e.g. a résumé falls back to a weaker variant because its
# omission footnote names "Kubernetes"/"AWS"). Affirmative claims are unaffected.
_OMISSION_MARKERS = (
    "omit", "not claim", "no verified", "without verified", "absence of evidence",
    "evidence flag", "no hands-on", "not yet hands-on", "no direct experience",
    "not in evidence", "do not have", "don't have", "does not have", "no experience with",
    "no exposure", "not applicable", "not relevant", "yet to work", "have yet to",
    "willing to learn", "open to learning", "eager to learn", "plan to learn",
)


def _is_omission_line(claim_lower: str) -> bool:
    """True when a line is an honest omission/disclosure footnote (see _OMISSION_MARKERS),
    so a not-claimable skill named inside it is a truthful disclosure, not a claim."""
    return any(m in claim_lower for m in _OMISSION_MARKERS)


# Hyphen/dash variants a model or user might use to reword a forbidden phrase. Normalized to a
# single space so 'cloud-native', 'cloud native', 'cloud‑native' (non-breaking hyphen) and
# 'cloud–native' (en-dash) all compare equal (audit HIGH: hyphen swaps evaded the gate).
_DASH_CHARS = "-‐‑‒–—―−"
_DASH_RE = re.compile(rf"[{_DASH_CHARS}]")


def _normalize_phrase(text: str) -> str:
    """Whitespace- and hyphen-robust normalization of a phrase/claim for the forbidden-claim
    substring test: dashes -> space, then whitespace runs collapsed. So a re-spaced or re-hyphenated
    forbidden phrase can't slip past the match (audit M12 + hyphen-evasion HIGH)."""
    return re.sub(r"\s+", " ", _DASH_RE.sub(" ", text or "")).strip()


def _digits_in(text: str) -> set:
    """All numeric tokens in ``text`` (comma/space-stripped digit runs) — for matching the
    impact metrics in a draft against the figures that actually appear in the evidence."""
    return {re.sub(r"[,\s]", "", t) for t in re.findall(r"\d[\d.,]*", text or "")}


def _credentials_in(text: str) -> set:
    """Core credential tokens present in ``text`` — by abbreviation OR spelled-out full name.
    Shared by the evidence-index builder and the draft checker so both recognize the same set."""
    found = {m.lower() for m in re.findall(rf"\b({_CREDENTIAL_TOKENS})\b", text or "", re.IGNORECASE)}
    low = (text or "").lower()
    for core, full in _CREDENTIAL_FULL.items():
        if full in low:
            found.add(core)
    return found


_CLAUSE_SPLIT = re.compile(r"[.;\n!?]")


def _clause_at(text: str, pos: int) -> str:
    """The sentence/clause of ``text`` containing character offset ``pos`` (split on . ; ! ? newline)
    — so an omission disclosure in one sentence can't whitelist an affirmative claim in another."""
    return _clause_around(text, pos, _CLAUSE_SPLIT)


def _clause_around(text: str, pos: int, splitter: re.Pattern) -> str:
    """The clause of ``text`` containing character offset ``pos`` per ``splitter``."""
    start, end = 0, len(text)
    for m in splitter.finditer(text):
        if m.start() < pos:
            start = m.start() + 1
        else:
            end = m.start()
            break
    return text[start:end]


# The forbidden-skill omission whitelist splits on COMMAS too (not only . ; ! ? newline): an
# affirmative claim and an aspirational marker sharing a period-clause but separated by a comma
# ('Kubernetes expert, plan to learn even more.') must NOT whitelist the affirmative claim
# (audit HIGH: omission-marker clause whitelist bypass).
_OMISSION_CLAUSE_SPLIT = re.compile(r"[.;\n!?,]")

# Affirmative-competence words. When one of these shares the skill's omission-clause, the mention is
# an AFFIRMATIVE claim, not an honest omission — so the omission whitelist must not apply even if a
# marker is co-present ('Expert in Kubernetes and eager to learn', 'gaps in my Kubernetes expertise'
# — audit HIGH). Word-boundary matched; 'experience'/'experienced' are deliberately absent so the
# 'no experience with X' disclosure marker itself is not mistaken for an affirmative claim.
_AFFIRMATIVE_SKILL_WORDS = re.compile(
    r"\b(?:expert|expertise|proficient|proficiency|skilled|extensive|deep|senior|lead|advanced|"
    r"strong|solid|professional|mastery|specializ(?:e|ed|ing)|years?)\b",
    re.IGNORECASE)


def _is_honest_skill_omission(claim_lower: str, skill_pos: int) -> bool:
    """True when the skill mention at ``skill_pos`` sits inside an HONEST omission disclosure: its
    comma-delimited clause carries an omission marker AND has no affirmative-competence word (so an
    affirmative claim that merely shares a period-clause / conjunction with an aspirational marker is
    still gated). See audit HIGH: omission-marker clause whitelist bypass."""
    clause = _clause_around(claim_lower, skill_pos, _OMISSION_CLAUSE_SPLIT)
    if not _is_omission_line(clause):
        return False
    return not _AFFIRMATIVE_SKILL_WORDS.search(clause)


# An aspirational/negative clause around a credential mention ('studying for the CPA exam',
# 'CPA candidate', 'not yet a CPA') means it is NOT held (audit M10).
_CREDENTIAL_ASPIRATIONAL = re.compile(
    r"\b(?:studying|study|pursuing|pursue|working\s+toward|preparing|prepare|sitting\s+for|"
    r"candidate|not\s+yet|in\s+progress|plan(?:ning)?\s+to|coursework|towards?|aspiring|exam)\b",
    re.IGNORECASE)


def _held_credentials_in(text: str) -> set:
    """Credentials the evidence says are actually HELD — a mention inside an aspirational/negative
    clause is excluded, so 'studying for the CPA exam' does not whitelist a 'licensed CPA' claim."""
    held: set = set()
    for m in re.finditer(rf"\b({_CREDENTIAL_TOKENS})\b", text or "", re.IGNORECASE):
        if not _CREDENTIAL_ASPIRATIONAL.search(_clause_at(text, m.start())):
            held.add(m.group(1).lower())
    low = (text or "").lower()
    for core, full in _CREDENTIAL_FULL.items():
        idx = low.find(full)
        if idx != -1 and not _CREDENTIAL_ASPIRATIONAL.search(_clause_at(text, idx)):
            held.add(core)
    return held


def _credential_is_disclosed(text: str, pos: int) -> bool:
    """True when a credential mention at offset ``pos`` sits in an aspirational ('studying for the
    CFA exam') OR honest-omission ('I do not have a CPA', 'No verified PMP') clause — i.e. NOT an
    affirmative claim to hold it. Mirrors the evidence-side aspirational filter plus the
    forbidden-skill omission whitelist, so an honest disclosure of a not-held credential does not
    hard-fail a truthful draft (audit HIGH: credential omission hard-fail)."""
    clause = _clause_at(text, pos)
    return bool(_CREDENTIAL_ASPIRATIONAL.search(clause)) or _is_omission_line(clause.lower())


def _claimed_credentials_in(text: str) -> set:
    """Core credential tokens AFFIRMATIVELY CLAIMED in ``text`` — a mention inside an aspirational or
    honest-omission clause is excluded (see _credential_is_disclosed). Used on the draft side so
    honest disclosures of a not-held credential are not flagged as fabricated."""
    claimed: set = set()
    for m in re.finditer(rf"\b({_CREDENTIAL_TOKENS})\b", text or "", re.IGNORECASE):
        if not _credential_is_disclosed(text, m.start()):
            claimed.add(m.group(1).lower())
    low = (text or "").lower()
    for core, full in _CREDENTIAL_FULL.items():
        idx = low.find(full)
        if idx != -1 and not _credential_is_disclosed(text, idx):
            claimed.add(core)
    return claimed


# Impact-metric patterns whose numeric core MUST trace to the evidence (TC-3): percentages,
# dollar amounts, and multipliers. Bare counts are intentionally not matched (too noisy).
_IMPACT_RES = (
    re.compile(r"(\d[\d.,]*)\s*(?:%|percent\b)", re.IGNORECASE),
    re.compile(r"\$\s*(\d[\d.,]*)\s*(?:million|billion|thousand|[kmb])?", re.IGNORECASE),
    re.compile(r"(\d[\d.,]*)\s*(?:x\b|-?fold\b)", re.IGNORECASE),
)


def _impact_metrics(text: str) -> list:
    """``[(token, numeric_core), ...]`` for each %/$/multiplier impact metric in ``text``."""
    out = []
    for rx in _IMPACT_RES:
        for m in rx.finditer(text or ""):
            out.append((m.group(0).strip(), m.group(1).replace(",", "")))
    return out


# TC-6 (counts): a draft can state a fabricated/stale plain COUNT ("23,000+ postings",
# "250+ automated tests", "35,000 documents") that the %/$/multiplier impact gate never sees —
# so "Valid: Yes, 0 violations" while a number is wrong. These count-nouns mark a number as a
# significant magnitude claim that MUST trace to the evidence.
_COUNT_NOUNS = (
    "posting", "postings", "job", "jobs", "test", "tests", "line", "lines", "loc",
    "document", "documents", "doc", "docs", "file", "files", "commit", "commits",
    "context", "contexts", "unit", "units", "board", "boards", "engine", "engines",
    "record", "records", "row", "rows", "user", "users", "customer", "customers",
    "company", "companies", "employer", "employers", "candidate", "candidates",
    "applicant", "applicants", "application", "applications", "request", "requests",
    "query", "queries", "transaction", "transactions", "node", "nodes", "agent", "agents",
    "model", "models", "token", "tokens", "page", "pages", "endpoint", "endpoints",
    "repository", "repositories", "repo", "repos", "branch", "branches", "vacancy", "vacancies",
)
# A number (with optional thousands separators / decimal) followed by an optional +/k/K suffix and
# then a count-noun, allowing up to two intervening modifier words ("250+ automated tests",
# "35,000+ source documents") — e.g. "28,701 postings", "32k LOC". Conservative: requires the
# count-noun, so years/phone numbers/bare integers are not matched. The modifier run is
# letter-only (no digits) so it can't swallow a second number.
_COUNT_RE = re.compile(
    rf"(\d[\d.,]*)\s*([+kK]?)\s+(?:of\s+)?(?:[A-Za-z][A-Za-z-]*\s+){{0,2}}({'|'.join(_COUNT_NOUNS)})\b",
    re.IGNORECASE,
)
# A year (1900-2099) standing on its own — excluded so "graduated in 2025" never flags. Only used
# to skip the bare-year case where the digits happen to precede a count-noun by coincidence.
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _normalize_count(core: str, suffix: str) -> str:
    """Canonical digit string for a count's numeric core, collapsing thousands separators and
    expanding a 'k'/'K' suffix to its ×1000 value ('32k' -> '32000', '23,000' -> '23000'), so a
    draft's 'Nk' and an evidence 'N,000' compare equal."""
    digits = re.sub(r"[,\s]", "", core)
    if suffix.lower() == "k":
        # Multiply by 1000; support a decimal 'k' like '2.5k' -> '2500'.
        try:
            val = float(digits)
        except ValueError:
            return digits
        scaled = val * 1000
        return str(int(scaled)) if scaled == int(scaled) else str(scaled)
    # Drop a trailing decimal '.0' so '23000.0' never sneaks past the integer evidence index.
    if digits.endswith(".0"):
        digits = digits[:-2]
    return digits


def _is_significant_count(core: str, suffix: str) -> bool:
    """A count is 'significant' (worth verifying) when it is >= 1000, OR carries a thousands
    separator, OR a '+'/'k'/'K' suffix. A bare small integer (e.g. '5 tests') is too noisy to
    flag, and a year is never significant on its own."""
    digits_only = re.sub(r"[,\s]", "", core)
    if _YEAR_RE.match(digits_only) and not suffix:
        return False
    if "," in core:
        return True
    if suffix in ("+", "k", "K"):
        return True
    try:
        return float(digits_only) >= 1000
    except ValueError:
        return False


def _significant_counts(text: str) -> list:
    """``[(token, normalized_core), ...]`` for each significant, count-noun-anchored magnitude in
    ``text`` (see _is_significant_count). The token is the matched span for the violation message;
    the normalized core is what to look up in the evidence figure index."""
    out = []
    for m in _COUNT_RE.finditer(text or ""):
        core, suffix = m.group(1), m.group(2)
        if not _is_significant_count(core, suffix):
            continue
        token = re.sub(r"\s+", " ", m.group(0).strip())
        out.append((token, _normalize_count(core, suffix)))
    return out


def _evidence_count_index(text: str) -> set:
    """The set of normalized count cores present in ``text`` — every numeric token, plus its
    k-expanded form when 'k'-suffixed — so a draft's 'Nk' or 'N,000' matches an evidence figure
    regardless of which notation the evidence used."""
    index: set = set()
    for raw in _digits_in(text):
        norm = raw[:-2] if raw.endswith(".0") else raw
        index.add(norm)
    # Also index any 'Nk'/'NK' figures in the evidence in their expanded form.
    for m in re.finditer(r"(\d[\d.,]*)\s*[kK]\b", text or ""):
        index.add(_normalize_count(m.group(1), "k"))
    return index


# Attribution (TC-5): flag a SELF-DIRECTED / personal-project signature listed under an EMPLOYER
# block (implying the employer owns work it didn't). Both marker sets are caller-supplied
# (constructor params, default empty -> the check is inert); the self-header markers below are
# generic and let the walker tell an employer section from a self-directed one.
_SELF_HEADER_MARKERS = (
    "self-directed", "self directed", "personal project", "independent",
    "infrastructure lab", "home lab", "side project", "personal infrastructure",
)


def _attribution_violation(draft_text, employer_markers, self_project_markers) -> Optional[str]:
    """Return the offending line if a self-directed-project signature (``self_project_markers``)
    appears UNDER an employer block (``employer_markers``) — after an employer header with no
    self-directed sub-header since. Returns None when clean, or when either marker set is empty.
    Walks sections by header line."""
    if not employer_markers or not self_project_markers:
        return None
    section = None  # None | "employer" | "self"
    for line in (draft_text or "").splitlines():
        ll = line.lower()
        if any(h in ll for h in _SELF_HEADER_MARKERS):
            section = "self"
        elif any(e in ll for e in employer_markers):
            section = "employer"
        if section == "employer" and any(m in ll for m in self_project_markers):
            return line.strip()
    return None


@dataclass
class ClaimViolation:
    """A single truth violation found in the draft text."""

    claim: str
    violation_type: str
    severity: str
    suggestion: str = ""


@dataclass
class TruthCheckResult:
    """Result of a truth check run."""

    draft_text: str
    violations: list[ClaimViolation] = field(default_factory=list)
    is_valid: bool = True
    summary: str = ""

    def add_violation(
        self,
        claim: str,
        violation_type: str,
        severity: str = "high",
        suggestion: str = "",
    ) -> None:
        self.violations.append(ClaimViolation(
            claim, violation_type, severity, suggestion))
        if severity == "high":
            self.is_valid = False

    def generate_summary(self) -> str:
        if not self.violations:
            self.summary = "All claims verified against evidence. No violations found."
        else:
            high = [v for v in self.violations if v.severity == "high"]
            medium = [v for v in self.violations if v.severity == "medium"]
            low = [v for v in self.violations if v.severity == "low"]
            parts = []
            if high:
                parts.append(f"HIGH ({len(high)}): {[v.claim for v in high]}")
            if medium:
                parts.append(
                    f"MEDIUM ({len(medium)}): {[v.claim for v in medium]}")
            if low:
                parts.append(f"LOW ({len(low)}): {[v.claim for v in low]}")
            self.summary = "Truth check violations:\n" + "\n".join(parts)
        return self.summary


class TruthChecker:
    """Validates generated draft text against candidate evidence sources."""

    def __init__(
        self,
        candidate_profile: str = "",
        experience_evidence: str = "",
        resume_blocks: str = "",
        forbidden_claims: str = "",
        forbidden_skills: Optional[list[str]] = None,
        employer_markers: Optional[tuple] = None,
        self_project_markers: Optional[tuple] = None,
    ) -> None:
        self.candidate_profile = candidate_profile
        self.experience_evidence = experience_evidence
        self.resume_blocks = resume_blocks
        self.forbidden_claims = forbidden_claims
        # Attribution (TC-5) markers, caller-supplied. Default empty -> the check is inert.
        self._employer_markers = tuple(employer_markers or ())
        self._self_project_markers = tuple(self_project_markers or ())

        # Whitespace- AND hyphen-normalized so a re-spaced ('active   secret clearance') or
        # re-hyphenated ('cloud native' vs 'cloud-native') forbidden phrase can't bypass the
        # substring match (audit M12 + hyphen-evasion HIGH) — the claim is normalized the same way
        # in check().
        self._forbidden_phrases: list[str] = [
            _normalize_phrase(line.lower())
            for line in self._split_claims(forbidden_claims)
            if _normalize_phrase(line.lower())
        ]
        # Frameworks/tech the candidate has NO verified hands-on (from qa_enrichment's
        # "NOT claimable" list). Matched on WORD BOUNDARIES so legitimate words don't trip
        # (e.g. "scalable" must not match "scala", "reaction" must not match "react").
        self._forbidden_skills: list[str] = [
            s.strip().lower() for s in (forbidden_skills or []) if s.strip()
        ]

        self._known_credentials: set[str] = set()
        self._known_entities: set[str] = set()
        # TC-3: the set of numeric figures that genuinely appear in the candidate's material, so a
        # %/$/multiplier in a draft that ISN'T one of these can be flagged as an unverified metric.
        self._evidenced_numbers: set[str] = _digits_in(
            f"{candidate_profile}\n{experience_evidence}\n{resume_blocks}"
        )
        # TC-6: normalized count index over the same evidence text, so a plain COUNT in a draft
        # ("23,000+ postings") can be traced to a real evidence figure with k/+/comma notations
        # treated equivalently.
        self._evidence_counts: set[str] = _evidence_count_index(
            f"{candidate_profile}\n{experience_evidence}\n{resume_blocks}"
        )
        self._build_entity_index()

    def _split_claims(self, text: str) -> list[str]:
        lines = text.splitlines()
        claims = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Require whitespace AFTER the marker dot so a real list index ('3. Built ...') is
                # stripped but a leading DECIMAL ('3.5 Kubernetes clusters') is left intact \u2014 the
                # old \s* matched the 'N.' of a decimal and mangled the claim (audit: leading-decimal
                # magnitude silently altered).
                cleaned = re.sub(r"^[\d\*\-\u2022]+\.\s+", "", stripped)
                claims.append(cleaned)
        return claims if claims else [text]

    def _build_entity_index(self) -> None:
        org_patterns = [
            r"\b(NASA|SpaceX|Blue Origin|Boeing|Lockheed|Northrop|Raytheon|JPL|ESA|CERN)\b",
        ]
        for pattern in org_patterns:
            matches = re.findall(pattern, self.experience_evidence)
            for m in matches:
                self._known_entities.add(m.lower())

        # Build the credential index from the SAME token set the checker flags, case-insensitively
        # against the original evidence. (The old code matched uppercase patterns against an
        # already-lowercased string with no IGNORECASE, so the index was ALWAYS empty and every
        # real, evidence-backed credential got falsely flagged + the truthful résumé withheld.)
        self._known_credentials |= _held_credentials_in(self.experience_evidence)

    def check(self, draft_text: str, job_description: str = "") -> TruthCheckResult:
        result = TruthCheckResult(draft_text=draft_text)
        draft_claims = self._split_claims(draft_text)

        for claim in draft_claims:
            claim_lower = claim.lower()
            # Whitespace- AND hyphen-robust so a re-spaced/re-hyphenated forbidden phrase is caught
            # (audit M12 + hyphen-evasion HIGH).
            claim_norm = _normalize_phrase(claim_lower)
            if not claim.strip():
                continue

            for forbidden in self._forbidden_phrases:
                # Word-char lookaround boundaries (matching the forbidden-skill matcher below) so a
                # phrase matches a whole token/phrase, not a fragment of a longer word ('full-stack'
                # must not fire on 'full-stackoverflow'). No length guard: forbidden phrases are
                # caller-curated (short terms like 'PhD'/'SQL' must still enforce).
                if re.search(rf"(?<!\w){re.escape(forbidden)}(?!\w)", claim_norm):
                    result.add_violation(
                        claim=claim,
                        violation_type="forbidden",
                        severity="high",
                        suggestion=f"Remove or replace '{forbidden}' -- this claim is forbidden.",
                    )
                    break

            # Only AFFIRMATIVE mentions of a not-claimable skill are violations (TC-1): an honest
            # "I omitted X / no verified experience with X" disclosure must not trip the gate. The
            # omission test is per-CLAUSE (the sentence containing the skill), NOT per-line — a
            # separate omission sentence on the same line must not whitelist an affirmative claim
            # (audit HIGH: "expert in Kubernetes. I have yet to work on mobile." used to slip through).
            for skill in self._forbidden_skills:
                # Boundary via word-char lookarounds (not \b) so symbol-bearing skills like
                # "c++"/"c#"/".net" can still be matched (a trailing \b never asserts after a symbol).
                sm = re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", claim_lower)
                if sm and not _is_honest_skill_omission(claim_lower, sm.start()):
                    result.add_violation(
                        claim=claim,
                        violation_type="unsupported_skill",
                        severity="high",
                        suggestion=f"'{skill}' is not in the evidence bank (NOT-claimable) -- remove it.",
                    )
                    break

            # Credential claims by abbreviation OR spelled-out full name (audit HIGH: a fabricated
            # "Certified Public Accountant" written out in full bypassed the abbreviation-only
            # check). Only AFFIRMATIVELY CLAIMED credentials count — an honest aspirational/omission
            # disclosure ('studying for the CFA', 'I do not have a CPA') must not hard-fail a
            # truthful draft (audit HIGH: credential omission hard-fail).
            for core in sorted(_claimed_credentials_in(claim)):
                if core not in self._known_credentials:
                    result.add_violation(
                        claim=claim,
                        violation_type="credential_missing",
                        severity="high",
                        suggestion=f"Credential '{core.upper()}' asserted but not in the evidence bank.",
                    )

            for entity_pattern in [
                r"\b(NASA|SpaceX|Blue Origin|Boeing|Lockheed|Northrop|Raytheon|JPL|ESA|CERN)\b",
            ]:
                entity_match = re.search(entity_pattern, claim, re.IGNORECASE)  # L3: 'worked at nasa' too
                if entity_match:
                    entity = entity_match.group(1).lower()
                    if entity not in self._known_entities:
                        result.add_violation(
                            claim=claim,
                            violation_type="no_evidence",
                            severity="medium",
                            suggestion=f"'{entity_match.group(0)}' not found in experience evidence.",
                        )

            skill_patterns = [
                r"\b(experienced\s+in|proficient\s+in|skilled\s+in|expert\s+in)\s+([a-z\s,]+?)\b",
                r"\b(have\s+(?:worked\s+)?with)\s+([a-z\s,]+?)\b",
                r"\b(built|developed|designed|managed|owned)\s+(the\s+)?([a-z\s,]+?)\b",
            ]
            for pattern in skill_patterns:
                match = re.search(pattern, claim_lower)
                if match:
                    skill = match.group(match.lastindex or 2).strip()
                    if len(skill) > 3 and skill not in ("the", "a", "an", "and", "or"):
                        if skill not in self.experience_evidence.lower():
                            result.add_violation(
                                claim=claim,
                                violation_type="no_evidence",
                                severity="low",
                                suggestion=f"Skill '{skill}' may need evidence support.",
                            )

        # TC-5: a self-directed/personal project listed under an employer block (HIGH — implies the
        # employer owns work it didn't). Inert unless the caller supplied both marker sets.
        offending = _attribution_violation(
            draft_text, self._employer_markers, self._self_project_markers)
        if offending:
            result.add_violation(
                claim=offending[:160],
                violation_type="misattribution",
                severity="high",
                suggestion="A self-directed/personal-project signature appears under an employer "
                           "block — move it under a clearly self-directed header so the employer is "
                           "not implied to own it.",
            )

        # TC-3: surface impact metrics (%, $, multipliers) whose value isn't in the evidence. The
        # prompt forbids inventing numbers, but local models occasionally slip; flag NON-blocking
        # (medium) so a fabricated stat is caught in the cockpit's truth warnings before it is sent,
        # without withholding the whole résumé over a borderline figure.
        for token, core in _impact_metrics(draft_text):
            if core and core not in self._evidenced_numbers:
                result.add_violation(
                    claim=token,
                    violation_type="unverified_metric",
                    severity="medium",
                    suggestion=f"'{token}' is not a figure in your evidence — verify it's real or "
                               "remove it before sending.",
                )

        # TC-6: surface a significant plain COUNT ("23,000+ postings", "250+ tests", "35,000
        # documents") whose numeric core isn't in the evidence figure index — the class of overclaim
        # the %/$/multiplier gate never sees. NON-blocking (medium): a fabricated/stale count is
        # caught in the cockpit's truth warnings before sending, without hard-failing a truthful draft.
        for token, core in _significant_counts(draft_text):
            if core and core not in self._evidence_counts:
                result.add_violation(
                    claim=token,
                    violation_type="unverified_count",
                    severity="medium",
                    suggestion=f"count {core} not found in evidence; use the evidence figure "
                               f"(verify '{token}' is real or replace it before sending).",
                )

        result.generate_summary()
        return result


def build_truth_checker_from_paths(
    profile_path: str = "",
    evidence_path: str = "",
    resume_blocks_path: str = "",
    forbidden_claims_path: str = "",
) -> TruthChecker:
    def _read(path: str) -> str:
        if not path:
            return ""
        try:
            with open(path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    return TruthChecker(
        candidate_profile=_read(profile_path),
        experience_evidence=_read(evidence_path),
        resume_blocks=_read(resume_blocks_path),
        forbidden_claims=_read(forbidden_claims_path),
    )

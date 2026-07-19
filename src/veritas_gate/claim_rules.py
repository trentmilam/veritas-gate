"""Declarative claim rules: one structured row per fact, enforced by a gate and rendered to a prompt.

WHAT PROBLEM THIS SOLVES
When an LLM pipeline keeps producing the same class of wrong claim, the usual fix is to add a
hand-written detector for that incident. Do that seventeen times and the nuances live in two places
that drift: the prompt re-types each fact as a literal instruction, and the checker encodes it as a
regex. The prompt can read prose, so a policy written in a spec file reaches the model immediately;
the gate cannot, so the same policy stays invisible to it until it leaks into a live draft and
someone writes another regex. A declarative registry closes that gap: one row is both the rule the
gate enforces and the instruction the model reads.

THE INTERESTING PART IS WHERE THIS STOPS WORKING
This module deliberately does NOT try to express every rule declaratively, because that was tried,
measured, and rejected. The full migration was tested against 9,963 real generated documents
(36.7M characters). Rules here match literal, lower-cased substrings with no word boundaries, and
at that scale the consequences are not subtle:

  * A two-character term from a real capability check -- ``rl`` -- fired on 3,410 of 9,963
    documents where the equivalent regex fired on 107. A 31.9x blast radius, 3,328 of them pure
    false positives, because "rl" sits inside world, girl, early, hourly, quarterly.
  * ``done`` matches inside abandoned, condone, undone. ``grow`` matches inside outgrew.
    ``initial`` matches inside uninitialized.
  * A single-word company-name ban matched a DIFFERENT real company that contained it as a
    substring.
  * Translating one numeric check into rows produced 112 FALSE NEGATIVES on the exact incident
    class that check existed to catch.

So: a rule belongs here when its terms are multi-word and cannot occur inside a larger word.
A rule belongs in code when it needs word boundaries, cross-sentence state, occurrence counting,
open-ended numeric comparison, negative lookbehind, or the surrounding document. Adding a
boundary-sensitive rule here does not make a gate stricter -- it makes it wrong in both directions
at once, over-firing on substrings and under-firing on the real pattern.

NEAREST-MARKER ATTRIBUTION
The non-obvious mechanism. A rule can forbid a term only when it belongs to a particular subject,
which matters because one real sentence often covers several subjects legitimately:

    "I initiated the retrieval platform and was assigned to the migration workstream."

Banning "assigned" whenever the sentence mentions the retrieval platform would flag this correct
sentence. Co-occurrence is the wrong test; ownership is the right one. Each forbidden term is
attributed to whichever subject's marker sits NEAREST to it in the sentence, so "assigned" attaches
to "migration workstream" and the sentence passes. Rewrite it as "I was assigned to the retrieval
platform" and the nearest marker changes, so it fires.

Rules are plain lists of dicts -- load them from JSON, a literal, or your own YAML reader. This
module stays zero-dependency and does not pick a config format for you.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

__all__ = ["RuleFinding", "check_claim_rules", "prompt_rules_block", "load_rules"]


@dataclass(frozen=True)
class RuleFinding:
    """One registry-rule violation."""

    rule_id: str
    violation_type: str
    severity: str
    claim: str
    message: str
    suggestion: str


def load_rules(source: Any) -> tuple:
    """Rules from a JSON file path, a JSON string, or an already-parsed list.

    No YAML reader is bundled: this package has zero runtime dependencies, and a caller who wants
    YAML can parse it themselves and pass the resulting list straight in.
    """
    if isinstance(source, (list, tuple)):
        return tuple(source)
    if not isinstance(source, (str, Path)):
        return ()
    try:
        # is_file(), not exists(): Path("") normalizes to Path("."), so an empty string would
        # otherwise be treated as a path and raise PermissionError trying to read a directory.
        text = Path(source).read_text(encoding="utf-8") if Path(source).is_file() else str(source)
        parsed = json.loads(text)
    except (OSError, ValueError):
        return ()
    return tuple(parsed) if isinstance(parsed, list) else ()


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _first_hit(haystack_low: str, needles: Optional[Iterable]) -> Optional[str]:
    for n in needles or ():
        if str(n).lower() in haystack_low:
            return str(n)
    return None


def _positions(haystack_low: str, needles: Optional[Iterable]) -> list[int]:
    """Every offset at which any needle occurs -- all occurrences, not just the first.

    All of them, because attribution compares distances: a subject marker that appears twice in a
    sentence must be able to claim a term next to EITHER occurrence.
    """
    out: list[int] = []
    for n in needles or ():
        needle = str(n).lower()
        i = haystack_low.find(needle)
        while i != -1:
            out.append(i)
            i = haystack_low.find(needle, i + 1)
    return out


def _nearest(pos: int, positions: Sequence[int]) -> float:
    return min((abs(pos - p) for p in positions), default=float("inf"))


def check_claim_rules(draft_text: str, rules: Any) -> list[RuleFinding]:
    """Every registry violation in ``draft_text``.

    Two clause shapes:

    ``phrases``          forbidden anywhere in the document. Use for closed, enumerable sets of
                         multi-word phrases.
    ``within_sentence``  forbidden only when the term belongs to ``subject``, decided by
                         nearest-marker attribution against the rule's other subjects.

    At most one finding per clause: a phrase repeated eleven times is one problem to fix, and
    reporting it eleven times buries the other findings.
    """
    rules = load_rules(rules)
    if not rules or not (draft_text or "").strip():
        return []

    findings: list[RuleFinding] = []
    sentences = _sentences(draft_text)
    low_sentences = [s.lower() for s in sentences]
    low_all = (draft_text or "").lower()

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or "unnamed")
        vtype = str(rule.get("violation_type") or "claim_rule_violation")
        severity = str(rule.get("severity") or "high")
        subjects = rule.get("subjects") or {}

        for clause in rule.get("forbid") or ():
            if not isinstance(clause, dict):
                continue
            message = " ".join(str(clause.get("message") or "").split())
            suggestion = " ".join(str(clause.get("suggestion") or "").split())

            phrases = clause.get("phrases")
            if phrases:
                hit = _first_hit(low_all, phrases)
                if hit:
                    findings.append(RuleFinding(rule_id, vtype, severity, hit, message, suggestion))
                continue

            subject_name = clause.get("subject")
            markers = (subjects.get(subject_name) or {}).get("any_of") if subject_name else None
            terms = clause.get("within_sentence")
            if not (markers and terms):
                continue

            competing = [m for name, spec in subjects.items() if name != subject_name
                         for m in ((spec or {}).get("any_of") or ())]
            exonerators = clause.get("exonerated_by") or ()

            for sent, low in zip(sentences, low_sentences):
                # This subject is already attributed correctly somewhere in the sentence.
                if exonerators and _first_hit(low, exonerators):
                    continue
                own = _positions(low, markers)
                if not own:
                    continue
                other = _positions(low, competing)
                term = next((t for t in terms
                             for ti in _positions(low, [t])
                             if _nearest(ti, own) <= _nearest(ti, other)), None)
                if term:
                    findings.append(RuleFinding(
                        rule_id, vtype, severity,
                        claim=" ".join(sent.split())[:180],
                        message=message,
                        suggestion=suggestion or f"remove '{term}' from this statement",
                    ))
                    break

    return findings


def prompt_rules_block(rules: Any) -> str:
    """The registry rendered as system-prompt instructions -- the same rows the gate enforces, so a
    new rule reaches the model and the checker from a single edit.

    Rows flagged ``render_to_prompt: false`` are omitted deliberately. For some rules, naming the
    banned vocabulary is what teaches it to a model that would never have produced it unprompted;
    those stay gate-only. That is a real tradeoff, not an oversight: the gate catches them if they
    appear, and the prompt does not suggest them.
    """
    lines: list[str] = []
    for rule in load_rules(rules):
        if not isinstance(rule, dict) or not rule.get("render_to_prompt", True):
            continue
        instruction = " ".join(str(rule.get("prompt_instruction") or "").split())
        if instruction:
            lines.append(f"- {instruction}")
    return "\n".join(lines)

"""Contract for the declarative claim registry.

The attribution tests are the load-bearing ones. Everything else here is bookkeeping; nearest-marker
attribution is the part that is easy to get subtly wrong and expensive to debug in production,
because the failure mode is a CORRECT sentence being blocked, which reads as the gate working.
"""
from __future__ import annotations

import json

import pytest

from veritas_gate.claim_rules import (RuleFinding, check_claim_rules, load_rules,
                                      prompt_rules_block)

# A neutral registry: two subjects that a single sentence can legitimately cover at once.
ATTRIBUTION_RULES = [{
    "id": "initiative-attribution",
    "violation_type": "attribution_error",
    "severity": "high",
    "render_to_prompt": True,
    "prompt_instruction": "Describe the retrieval platform as self-initiated; the migration "
                          "workstream was assigned.",
    "subjects": {
        "self_initiated": {"any_of": ["retrieval platform", "search platform"]},
        "assigned_work": {"any_of": ["migration workstream", "reporting workstream"]},
    },
    "forbid": [{
        "subject": "self_initiated",
        "within_sentence": ["was assigned", "were assigned", "tasked with"],
        "exonerated_by": ["initiated the retrieval platform"],
        "message": "the retrieval platform was self-initiated, not assigned",
        "suggestion": "say 'initiated' or 'proposed' for the retrieval platform",
    }],
}]

PHRASE_RULES = [{
    "id": "banned-phrases",
    "violation_type": "overclaim",
    "severity": "high",
    "render_to_prompt": False,
    "prompt_instruction": "never say 'industry leading' or 'world class'",
    "forbid": [{
        "phrases": ["industry leading", "world class"],
        "message": "unsupportable superlative",
    }],
}]


class TestNearestMarkerAttribution:
    def test_a_sentence_covering_both_subjects_correctly_is_not_flagged(self):
        """The whole reason co-occurrence is not the test. This sentence is TRUE."""
        text = ("I initiated the retrieval platform, and separately I was assigned to the "
                "migration workstream.")
        assert check_claim_rules(text, ATTRIBUTION_RULES) == []

    def test_the_term_attached_to_the_wrong_subject_is_flagged(self):
        text = "I was assigned to the retrieval platform last quarter."
        found = check_claim_rules(text, ATTRIBUTION_RULES)
        assert len(found) == 1
        assert found[0].rule_id == "initiative-attribution"
        assert found[0].violation_type == "attribution_error"

    def test_attribution_follows_distance_not_word_order(self):
        """Both subjects present, forbidden term nearest the self-initiated one -> fires, even
        though the assigned subject appears FIRST in the sentence."""
        text = "Beyond the migration workstream, the retrieval platform was assigned to me."
        assert len(check_claim_rules(text, ATTRIBUTION_RULES)) == 1

    def test_an_exonerating_phrase_skips_the_clause(self):
        text = ("I initiated the retrieval platform after I was assigned to adjacent work on the "
                "retrieval platform roadmap.")
        assert check_claim_rules(text, ATTRIBUTION_RULES) == []

    def test_the_subject_must_actually_appear(self):
        assert check_claim_rules("I was assigned to unrelated work.", ATTRIBUTION_RULES) == []

    def test_a_repeated_subject_marker_can_claim_a_term_next_to_either_occurrence(self):
        """_positions collects ALL offsets, not the first. With only the first, the term here would
        measure against the distant leading mention and be misattributed."""
        text = ("The retrieval platform shipped in March; the migration workstream followed, and "
                "the retrieval platform was assigned to me.")
        assert len(check_claim_rules(text, ATTRIBUTION_RULES)) == 1


class TestPhraseClauses:
    def test_a_banned_phrase_is_caught_anywhere_in_the_document(self):
        found = check_claim_rules("Our industry leading platform.", PHRASE_RULES)
        assert len(found) == 1 and found[0].claim == "industry leading"

    def test_one_finding_per_clause_even_when_repeated(self):
        text = "world class. " * 12
        assert len(check_claim_rules(text, PHRASE_RULES)) == 1

    def test_matching_is_case_insensitive(self):
        assert check_claim_rules("WORLD CLASS results.", PHRASE_RULES)

    def test_clean_text_yields_nothing(self):
        assert check_claim_rules("A measured description of the work.", PHRASE_RULES) == []


class TestSubstringHazard:
    def test_the_documented_hazard_is_real_and_therefore_not_papered_over(self):
        """This module matches literal substrings with NO word boundaries, and the docstring says so
        loudly because the migration that assumed otherwise produced a 31.9x blast radius on a
        two-character term. This test PINS the sharp edge rather than hiding it: if someone later
        adds word-boundary matching, this fails and forces them to revisit the scope guidance.
        """
        rules = [{"id": "hazard", "violation_type": "t", "severity": "high",
                  "forbid": [{"phrases": ["rl"], "message": "m"}]}]
        assert check_claim_rules("She traveled the world early each quarter.", rules), (
            "substring matching is the documented behavior; 'rl' inside world/early is exactly why "
            "short terms must live in code instead of the registry"
        )


class TestPromptRendering:
    def test_rows_render_as_instructions(self):
        assert prompt_rules_block(ATTRIBUTION_RULES).startswith("- Describe the retrieval platform")

    def test_gate_only_rows_are_withheld_from_the_prompt(self):
        """Naming a banned phrase in the prompt can teach it to a model that would never have
        produced it. Those rows are enforced but not rendered."""
        rendered = prompt_rules_block(PHRASE_RULES)
        assert rendered == ""
        assert "world class" not in rendered
        # ...but the gate still enforces them.
        assert check_claim_rules("world class", PHRASE_RULES)

    def test_the_gate_and_the_prompt_read_the_SAME_rows(self):
        """The point of the registry: one edit reaches both. A rule that renders must also enforce."""
        both = ATTRIBUTION_RULES + PHRASE_RULES
        assert prompt_rules_block(both)
        assert check_claim_rules("I was assigned to the retrieval platform.", both)
        assert check_claim_rules("industry leading", both)


class TestLoading:
    def test_rules_load_from_a_json_string(self):
        assert check_claim_rules("world class", load_rules(json.dumps(PHRASE_RULES)))

    def test_rules_load_from_a_json_file(self, tmp_path):
        p = tmp_path / "rules.json"
        p.write_text(json.dumps(PHRASE_RULES), encoding="utf-8")
        assert check_claim_rules("world class", load_rules(p))

    @pytest.mark.parametrize("junk", [None, 42, {}, "", []])
    def test_unusable_sources_yield_no_rules_rather_than_raising(self, junk):
        assert check_claim_rules("world class", junk) == []

    def test_malformed_rows_are_skipped_not_fatal(self):
        assert check_claim_rules("world class", ["nonsense", None, 7] + PHRASE_RULES)

    def test_empty_text_yields_nothing(self):
        assert check_claim_rules("", PHRASE_RULES) == []
        assert check_claim_rules("   \n ", PHRASE_RULES) == []


def test_findings_are_hashable_so_callers_can_dedupe():
    found = check_claim_rules("world class", PHRASE_RULES)
    assert isinstance(found[0], RuleFinding)
    assert len({*found, *found}) == 1

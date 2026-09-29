"""Grounding checks (grounding.py): the 79.2% of hallucination spans TC-3/TC-6 can't see."""
from __future__ import annotations

from veritas_gate import TruthChecker
from veritas_gate.grounding import (
    broadened_numeric_ungrounded,
    novel_content_windows,
    ungrounded_entities,
)

EVIDENCE = (
    "Acme Corp shipped the Falcon platform in 2019. The team of 12 engineers cut latency by "
    "shipping a caching layer. Revenue reached 4.5 million dollars that year."
)


class TestBroadenedNumericUngrounded:
    def test_evidenced_number_is_not_flagged(self) -> None:
        assert broadened_numeric_ungrounded("The Falcon platform shipped in 2019.", EVIDENCE) == []

    def test_fabricated_number_is_flagged(self) -> None:
        assert "47" in broadened_numeric_ungrounded("The team shipped 47 updates.", EVIDENCE)

    def test_matches_only_the_unevidenced_token_not_the_whole_sentence(self) -> None:
        out = broadened_numeric_ungrounded("Revenue reached 4.5 million and headcount hit 999 now.",
                                            EVIDENCE)
        assert out == ["999"]          # 4.5 is evidenced; only the fabricated figure survives

    def test_a_figure_ending_a_sentence_in_the_evidence_still_counts_as_grounded(self) -> None:
        """The digit regex is greedy, so a figure that ends a sentence tokenizes WITH its period
        ("...in 2019." -> "2019."). Without stripping that, the same figure mid-sentence in the
        draft ("2019") compares unequal and a perfectly grounded number is reported as fabricated,
        a false positive manufactured entirely by punctuation."""
        assert broadened_numeric_ungrounded("Shipped in 2019 on schedule.",
                                            "The product shipped in 2019.") == []

    def test_an_internal_decimal_point_is_not_stripped(self) -> None:
        """The trailing-punctuation strip must not turn 4.5 into 4: that would silently make two
        different figures compare equal, trading a false positive for a false negative."""
        assert broadened_numeric_ungrounded("Revenue was 4.5 million.", "Revenue was 4 million.") \
            == ["4.5"]


class TestUngroundedEntities:
    def test_evidenced_entity_is_not_flagged(self) -> None:
        assert ungrounded_entities("Acme Corp shipped Falcon.", EVIDENCE) == []

    def test_fabricated_entity_is_flagged(self) -> None:
        assert "Zorbex" in ungrounded_entities("Zorbex Industries acquired the team.", EVIDENCE)

    def test_common_sentence_opener_is_not_flagged_as_an_entity(self) -> None:
        # "The" opens the sentence and is a common opener; must not be treated as a proper noun.
        out = ungrounded_entities("The engineers shipped Falcon on time.", EVIDENCE)
        assert "The" not in out

    def test_non_sentence_initial_stopword_lookalike_is_still_excluded(self) -> None:
        # "In" is a sentence-opener only at position 0; mid-sentence capitalization of a real
        # stopword should never fire regardless of position since stopwords are always excluded.
        out = ungrounded_entities("Shipped it In record time.", EVIDENCE)
        assert "In" not in out

    def test_one_name_repeated_is_one_finding(self) -> None:
        """A fabricated company named five times is one thing to fix: five identical violations
        bury every other finding, the cap claim_rules.py states for a repeated phrase."""
        draft = "Zorbex led the deal. Zorbex grew fast. Zorbex hired many. Zorbex shipped it."
        assert ungrounded_entities(draft, EVIDENCE) == ["Zorbex"]

    def test_distinct_names_are_all_still_reported(self) -> None:
        """Deduplication must collapse repeats of ONE name, not distinct names."""
        out = ungrounded_entities("Zorbex acquired Prendergast. Then Zorbex hired Vashti.",
                                  EVIDENCE)
        assert out == ["Zorbex", "Prendergast", "Vashti"]      # first-appearance order

    def test_a_capitalized_non_opener_adverb_is_a_known_false_positive(self) -> None:
        """Pins a real, priced-in limitation rather than papering over it: this check is a cheap
        capitalization proxy, so a sentence-opening adverb outside the opener lexicon ("Later")
        reads as a proper noun. This is part of why the measured precision is 40.1%, and why the
        three grounding checks ship opt-in and off by default. Change this only by re-running
        benchmark/tune.py, never by adding words after eyeballing one example."""
        assert "Later" in ungrounded_entities("Later Zorbex hired Vashti.", EVIDENCE)


class TestNovelContentWindows:
    def test_fully_evidenced_text_has_no_novel_window(self) -> None:
        assert novel_content_windows(
            "Acme Corp shipped the Falcon platform in 2019 with a caching layer.", EVIDENCE) == []

    def test_wholly_fabricated_passage_is_flagged(self) -> None:
        out = novel_content_windows(
            "Zorbex Dynamics unveiled quantum teleportation devices yesterday in Geneva.", EVIDENCE,
            window=5, threshold=1.0)
        assert out

    def test_short_draft_below_window_size_returns_no_windows(self) -> None:
        assert novel_content_windows("Falcon shipped.", EVIDENCE, window=10) == []

    def test_lower_threshold_catches_more_than_a_strict_one(self) -> None:
        draft = "Acme Corp shipped Zorbex quantum teleportation devices."
        strict = novel_content_windows(draft, EVIDENCE, window=5, threshold=1.0)
        loose = novel_content_windows(draft, EVIDENCE, window=5, threshold=0.6)
        assert len(loose) >= len(strict)

    # _WORD_RE matches letters only, so distinct filler tokens must be letter-only too: "novel0"
    # and "novel1" both tokenize to "novel".
    _FILLER_A = "quark zephyr mango trellis vortex nimbus cobalt saffron gantry plinth widget dovetail"
    _FILLER_B = "obelisk lantern harrow spindle thicket bramble cinder marrow rivet tundra fathom quill"

    def test_one_long_ungrounded_passage_is_one_finding_not_one_per_window(self) -> None:
        """A long ungrounded paragraph is one problem to fix. Reporting it once per sliding window
        (51 near-identical violations for a 60-word passage) buries every other finding, the same
        reasoning claim_rules.py gives for its one-finding-per-clause cap."""
        draft = f"{self._FILLER_A} {self._FILLER_B}"
        out = novel_content_windows(draft, "Nothing related here at all.", window=10, threshold=1.0)
        assert len(out) == 1
        assert len(out[0].split()) == 24          # the merged span covers the whole passage

    def test_two_separated_ungrounded_passages_stay_two_findings(self) -> None:
        """Merging must join only overlapping windows, not collapse genuinely distinct passages."""
        grounded = "acme corp falcon platform engineers latency caching layer revenue dollars year"
        draft = f"{self._FILLER_A} {grounded} {self._FILLER_B}"
        out = novel_content_windows(draft, EVIDENCE, window=10, threshold=1.0)
        assert len(out) == 2
        assert out[0].startswith("quark") and out[1].startswith("obelisk")


class TestWiredIntoTruthChecker:
    def _gate(self, **flags) -> TruthChecker:
        return TruthChecker(experience_evidence=EVIDENCE, **flags)

    def test_all_three_flags_default_off(self) -> None:
        result = self._gate().check(
            "Zorbex Industries shipped 999 quantum teleportation devices in Geneva.")
        types = {v.violation_type for v in result.violations}
        assert types.isdisjoint({"unverified_number", "ungrounded_entity", "novel_content_window"})

    def test_broadened_numeric_flag_fires_when_enabled(self) -> None:
        result = self._gate(enable_broadened_numeric=True).check("Headcount grew to 999.")
        assert any(v.violation_type == "unverified_number" for v in result.violations)

    def test_entity_grounding_flag_fires_when_enabled(self) -> None:
        result = self._gate(enable_entity_grounding=True).check("Zorbex Industries led the deal.")
        assert any(v.violation_type == "ungrounded_entity" for v in result.violations)

    def test_novelty_flag_fires_when_enabled(self) -> None:
        # Needs >= the default window (10 content words) to produce any window at all.
        result = self._gate(enable_novelty_check=True).check(
            "Zorbex Dynamics unveiled quantum teleportation devices yesterday in Geneva to a "
            "stunned crowd of onlookers gathered outside the embassy gates.")
        assert any(v.violation_type == "novel_content_window" for v in result.violations)

    def test_grounding_violations_are_medium_severity_non_blocking(self) -> None:
        result = self._gate(enable_broadened_numeric=True, enable_entity_grounding=True,
                            enable_novelty_check=True).check(
            "Zorbex Industries shipped 999 quantum teleportation devices in Geneva.")
        fired = [v for v in result.violations
                 if v.violation_type in ("unverified_number", "ungrounded_entity",
                                          "novel_content_window")]
        assert fired
        assert all(v.severity == "medium" for v in fired)
        # Non-blocking: severity-medium alone must not flip is_valid False.
        assert result.is_valid

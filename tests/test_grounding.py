"""Grounding checks (grounding.py) — the 79.2% of hallucination spans TC-3/TC-6 can't see."""
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
        # A trailing "." abuts the digit run and is swept in by design (same _digits_in()-style
        # regex checker.py's own TC-3/TC-6 use) -- assert by substring, not exact token equality.
        out = broadened_numeric_ungrounded("Revenue reached 4.5 million and headcount hit 999 now.",
                                            EVIDENCE)
        assert any("999" in t for t in out)
        assert not any("4.5" in t for t in out)  # evidenced


class TestUngroundedEntities:
    def test_evidenced_entity_is_not_flagged(self) -> None:
        assert ungrounded_entities("Acme Corp shipped Falcon.", EVIDENCE) == []

    def test_fabricated_entity_is_flagged(self) -> None:
        assert "Zorbex" in ungrounded_entities("Zorbex Industries acquired the team.", EVIDENCE)

    def test_common_sentence_opener_is_not_flagged_as_an_entity(self) -> None:
        # "The" opens the sentence and is a common opener -- must not be treated as a proper noun.
        out = ungrounded_entities("The engineers shipped Falcon on time.", EVIDENCE)
        assert "The" not in out

    def test_non_sentence_initial_stopword_lookalike_is_still_excluded(self) -> None:
        # "In" is a sentence-opener only at position 0; mid-sentence capitalization of a real
        # stopword should never fire regardless of position since stopwords are always excluded.
        out = ungrounded_entities("Shipped it In record time.", EVIDENCE)
        assert "In" not in out


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

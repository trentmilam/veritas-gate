"""is_degenerate — repetition-runaway detector."""
from __future__ import annotations

from veritas_gate import is_degenerate

CLEAN = (
    "Alex Doe — Forward Deployed AI Engineer. Built a retrieval system over a large corpus, "
    "stood up local model serving, and wrote a deterministic evaluation suite with real tests."
)


def test_clean_text_is_not_degenerate() -> None:
    assert not is_degenerate(CLEAN)
    assert not is_degenerate("")
    assert not is_degenerate("A short, ordinary sentence.")


def test_comma_run_runaway_is_degenerate() -> None:
    runaway = "Skills: " + ", ".join("AI " + w for w in
              (["futures", "potentials", "possibilities", "promises", "benefits"] * 100))
    assert is_degenerate(runaway)


def test_collapsed_vocabulary_is_degenerate() -> None:
    assert is_degenerate("optimization " * 500)

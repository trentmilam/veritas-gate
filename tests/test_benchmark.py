"""Tests for the RAGTruth harness itself.

Two of the benchmark's headline claims are properties of the harness, not of the corpus, so they
are testable without the 36 MB download and run in CI:

    determinism   -- the reported numbers must reproduce exactly, including the error bars
    BLOCKED       -- an undersized sample must yield no quality number at all

Everything here uses synthetic rows. Nothing skips when the corpus is absent, because a test that
quietly skips is how an untested claim ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmark"
sys.path.insert(0, str(BENCH))

import run as harness  # noqa: E402
from metrics import clustered_bootstrap_f1, prf, wilson  # noqa: E402


def _rows(n_clusters: int = 40, per_cluster: int = 6) -> list[dict]:
    """Deterministic synthetic rows with a mix of tp, fp, fn and tn.

    Clusters must be HETEROGENEOUS. An earlier version keyed pred/gold on ``(c + i)`` mod small
    primes, which gave every cluster an identical 1/1/2 tp/fp/fn split -- and F1 is scale-invariant,
    so every bootstrap resample returned the same value and the interval collapsed to a point. The
    seed test below is what caught it, which is the reason that test exists.
    """
    out = []
    for c in range(n_clusters):
        for i in range(per_cluster):
            out.append({
                "cluster": f"src-{c}",
                "pred": (c * 7 + i * i) % 4 == 0,
                "gold": (c * c + i) % 3 != 0,
            })
    return out


class TestDeterminism:
    def test_bootstrap_interval_is_identical_across_calls(self):
        rows = _rows()
        assert clustered_bootstrap_f1(rows, "cluster") == clustered_bootstrap_f1(rows, "cluster")

    def test_bootstrap_interval_survives_row_reordering(self):
        """Clusters are resampled by identity, so input order must not move the interval."""
        rows = _rows()
        assert clustered_bootstrap_f1(rows, "cluster") == clustered_bootstrap_f1(
            list(reversed(rows)), "cluster")

    def test_the_interval_is_computed_from_the_data_not_constant(self):
        """Vacuity guard: reproducibility is trivially satisfiable by returning a fixed pair."""
        lo_a, hi_a = clustered_bootstrap_f1(_rows(), "cluster")
        lo_b, hi_b = clustered_bootstrap_f1(
            [{**r, "gold": not r["gold"]} for r in _rows()], "cluster")
        assert (lo_a, hi_a) != (lo_b, hi_b)
        assert hi_a > lo_a

    def test_the_interval_brackets_the_observed_f1(self):
        rows = _rows()
        observed = prf(sum(1 for r in rows if r["pred"] and r["gold"]),
                       sum(1 for r in rows if r["pred"] and not r["gold"]),
                       sum(1 for r in rows if not r["pred"] and r["gold"]))["f1"]
        lo, hi = clustered_bootstrap_f1(rows, "cluster")
        assert lo <= observed <= hi

    def test_the_reported_bounds_are_stable_under_a_reseed(self):
        """Not a determinism restatement: the percentile bounds land on the same value for a
        DIFFERENT seed, so the published interval is a property of the sample rather than of the
        one seed that happened to be pinned."""
        assert clustered_bootstrap_f1(_rows(), "cluster") == clustered_bootstrap_f1(
            _rows(), "cluster", seed=1)


class TestBlockedPath:
    def test_undersized_sample_reports_blocked_and_no_number(self, monkeypatch, capsys):
        small = [{"source_id": "s0", "response": "x", "split": "test", "labels": []}]
        monkeypatch.setattr(harness, "load", lambda: (small, {"s0": {}}))

        assert harness.main() == 2

        out = capsys.readouterr().out
        assert "BLOCKED" in out
        for forbidden in ("precision", "recall", "F1"):
            assert forbidden not in out, f"a BLOCKED run must not report {forbidden}"

    def test_min_sample_is_a_real_floor(self):
        assert harness.MIN_SAMPLE >= 200


class TestScopeGuards:
    def test_only_the_two_generic_checks_are_scored(self):
        """The whole benchmark's validity rests on no resume-specific rule leaking into the score."""
        assert harness.GENERIC_VIOLATIONS == {"unverified_metric", "unverified_count"}

    def test_evidence_comes_from_source_info_not_the_dataset_name(self):
        """``source`` is only "MARCO"/"CNN/DM"/"Yelp". Grounding against it would score nothing."""
        row = {"source": "CNN/DM", "source_info": "Revenue rose 12% to $4.1 billion."}
        assert harness.evidence_for(row) == "Revenue rose 12% to $4.1 billion."
        assert "CNN/DM" not in harness.evidence_for(row)

    def test_dict_evidence_is_serialized_whole(self):
        """QA and Data2txt carry dicts; every figure the generator saw must reach the bank."""
        row = {"source_info": {"question": "How many?", "passages": ["about 4,200 units"]}}
        evidence = harness.evidence_for(row)
        assert "4,200" in evidence and "How many?" in evidence


class TestPublishedNumbersMatchTheResultsFile:
    """The README publishes figures a reader cannot recompute without the 36 MB corpus. A repo whose
    subject is unsupported claims should not let its own published numbers drift from the machine-
    readable output that produced them -- that is the exact defect this codebase exists to catch.
    """

    @pytest.fixture
    def published(self) -> tuple[dict, str]:
        import json
        root = Path(__file__).resolve().parent.parent
        results = root / "benchmark" / "results.json"
        if not results.exists():
            pytest.fail("benchmark/results.json is committed alongside the README; regenerate with "
                        "`python benchmark/run.py` rather than deleting it")
        return json.loads(results.read_text(encoding="utf-8")), (
            root / "README.md").read_text(encoding="utf-8")

    def test_every_headline_figure_appears_verbatim(self, published):
        results, readme = published
        ceiling = results["digit_span_recall_ceiling"]
        expected = {
            "test responses": f"{results['n']:,}",
            "source clusters": f"{results['source_clusters']:,}",
            "base rate": f"{results['base_rate'] * 100:.1f}%",
            "precision": f"{results['overall']['precision'] * 100:.1f}%",
            "recall": f"{results['overall']['recall'] * 100:.1f}%",
            "F1": f"{results['overall']['f1'] * 100:.1f}%",
            "trivial-classifier F1": f"{results['naive_always_positive_f1'] * 100:.1f}%",
            "annotated spans": f"{ceiling['spans_total']:,}",
            "digit-span ratio": f"{ceiling['ratio'] * 100:.1f}%",
        }
        missing = {k: v for k, v in expected.items() if v not in readme}
        assert not missing, f"README figures drifted from results.json: {missing}"

    def test_the_readme_does_not_claim_to_beat_the_trivial_floor(self, published):
        """Guards the direction of the honest headline, not just the digits."""
        results, readme = published
        beats = results["overall"]["f1"] > results["naive_always_positive_f1"]
        assert not beats, "measurement changed -- the README's 'do not beat' framing needs a rewrite"
        assert "not beat a trivial classifier" in readme.lower()


class TestMetrics:
    def test_wilson_stays_inside_the_unit_interval_at_the_extremes(self):
        """The Wald interval fails exactly here; this is the reason Wilson was chosen."""
        for count, total in ((0, 25), (25, 25), (1, 3)):
            lo, hi = wilson(count, total)
            assert 0.0 <= lo <= hi <= 1.0

    def test_wilson_has_width_at_a_zero_rate(self):
        lo, hi = wilson(0, 25)
        assert hi > lo

    def test_a_detector_that_never_fires_scores_zero_not_undefined(self):
        assert prf(0, 0, 10) == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                 "tp": 0, "fp": 0, "fn": 10}

    @pytest.mark.parametrize("tp,fp,fn,expected", [(1, 1, 1, 0.5), (10, 0, 0, 1.0)])
    def test_f1(self, tp, fp, fn, expected):
        assert prf(tp, fp, fn)["f1"] == pytest.approx(expected)

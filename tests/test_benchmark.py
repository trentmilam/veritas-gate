"""Tests for the RAGTruth harness itself.

Two of the benchmark's headline claims are properties of the harness, not of the corpus, so they
are testable without the 36 MB download and run in CI:

    determinism   -- the reported numbers must reproduce exactly, including the error bars
    BLOCKED       -- an undersized sample must yield no quality number at all

Everything here uses synthetic rows. Nothing skips when the corpus is absent, because a test that
quietly skips is how an untested claim ships.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

import run as harness  # noqa: E402
from metrics import clustered_bootstrap_f1, decision_verdict, prf, wilson  # noqa: E402


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

        assert harness.main(["--reason", "test: undersized sample"]) == 2

        out = capsys.readouterr().out
        assert "BLOCKED" in out
        for forbidden in ("precision", "recall", "F1"):
            assert forbidden not in out, f"a BLOCKED run must not report {forbidden}"

    def test_min_sample_is_a_real_floor(self):
        assert harness.MIN_SAMPLE >= 200


class TestAccessDiscipline:
    """A run against the real test split is deliberately rare and deliberately visible -- see the
    module docstring's "TEST-SET ACCESS DISCIPLINE" section. These pin the two halves of that: the
    gate can't be bypassed, and the fixture path (which never touches the real split) needs none of
    it."""

    def test_missing_reason_blocks_before_touching_the_corpus(self, monkeypatch, capsys):
        def _must_not_be_called():
            raise AssertionError("load() must never run when --reason was not given")
        monkeypatch.setattr(harness, "load", _must_not_be_called)

        assert harness.main([]) == 2   # must not raise -- proves load() was never reached
        out = capsys.readouterr().out
        assert "BLOCKED" in out
        assert "--reason" in out

    def test_a_real_run_appends_one_line_to_the_committed_access_log(self, tmp_path, monkeypatch):
        log_path = tmp_path / "TEST_SET_ACCESS_LOG.jsonl"
        monkeypatch.setattr(harness, "ACCESS_LOG", log_path)
        monkeypatch.setattr(harness, "RESULTS_PATH", tmp_path / "results.json")
        big = [{"source_id": f"s{i}", "response": "x", "split": "test", "labels": []}
               for i in range(harness.MIN_SAMPLE)]
        monkeypatch.setattr(harness, "load", lambda: (big, {f"s{i}": {} for i in range(len(big))}))

        assert harness.main(["--reason", "unit test"]) == 0

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["reason"] == "unit test"
        assert entry["checker_sha256"] == harness._file_hash(*harness.CHECKER_FILES)
        assert "at_utc" in entry

    def test_fixture_mode_needs_no_reason_and_touches_no_real_corpus(self, monkeypatch):
        def _must_not_be_called():
            raise AssertionError("load() (the real corpus) must never run in --fixture mode")
        monkeypatch.setattr(harness, "load", _must_not_be_called)

        assert harness.main(["--fixture"]) == 0   # must not raise


class TestFixture:
    """The committed offline slice CI actually runs against every push -- no network, no test-set
    access. It exists to catch a harness-LOGIC bug (evidence serialization, the violation filter, the
    metric formulas) that a README-text-diff test cannot: something with real corpus shape, scored on
    every push, for free."""

    def test_fixture_files_exist(self):
        fx = BENCH / "fixtures"
        assert (fx / "response_sample.jsonl").exists()
        assert (fx / "source_info_sample.jsonl").exists()
        assert (fx / "expected.json").exists()

    def test_fixture_meets_the_min_sample_floor(self):
        """If this fails, benchmark/fixtures/build_fixture.py's STRIDE needs lowering -- a fixture
        under MIN_SAMPLE would silently stop exercising the real scoring path and only exercise
        BLOCKED, which defeats the whole point of this fixture."""
        responses, _ = harness.load_fixture()
        test = [r for r in responses if r.get("split") == "test"]
        assert len(test) >= harness.MIN_SAMPLE

    def test_fixture_reproduces_its_committed_expected_values(self):
        responses, sources = harness.load_fixture()
        test = [r for r in responses if r.get("split") == "test"]
        out = harness.score(test, responses, sources)
        assert out is not None
        expected = json.loads((BENCH / "fixtures" / "expected.json").read_text(encoding="utf-8"))
        assert out["n"] == expected["n"]
        assert out["overall"]["tp"] == expected["tp"]
        assert out["overall"]["fp"] == expected["fp"]
        assert out["overall"]["fn"] == expected["fn"]
        assert out["digit_span_recall_ceiling"]["ratio"] == pytest.approx(
            expected["digit_ceiling_ratio"])
        assert out["digit_span_recall_ceiling"]["spans_total"] == expected["digit_spans_total"]

    def test_full_pipeline_is_byte_identical_across_two_runs_on_the_fixture(self):
        """The determinism claim tested end-to-end, not just clustered_bootstrap_f1 in isolation."""
        responses, sources = harness.load_fixture()
        test = [r for r in responses if r.get("split") == "test"]
        out1 = harness.score(test, responses, sources)
        out2 = harness.score(test, responses, sources)
        # _timing is wall-clock and deliberately excluded from the determinism claim -- see the
        # comment on score()'s `out["_timing"]` assignment in run.py.
        persisted1 = {k: v for k, v in out1.items() if k != "_timing"}
        persisted2 = {k: v for k, v in out2.items() if k != "_timing"}
        assert json.dumps(persisted1, sort_keys=True) == json.dumps(persisted2, sort_keys=True)

    def test_fixture_mode_never_touches_the_published_results_file(self, tmp_path, monkeypatch):
        decoy = tmp_path / "should_not_be_written.json"
        monkeypatch.setattr(harness, "RESULTS_PATH", decoy)
        assert harness.main(["--fixture"]) == 0
        assert not decoy.exists()


class TestProvenance:
    """The content-hash gate: `checker.py` changing with a stale committed `results.json` must fail
    CI in the SAME PR, not drift silently. This is the direct fix for the harness previously only
    checking README-text agreement with results.json, never results.json agreement with reality."""

    @pytest.fixture
    def results(self) -> dict:
        path = BENCH / "results.json"
        if not path.exists():
            pytest.fail("benchmark/results.json is committed alongside the README; regenerate with "
                        "`python benchmark/run.py --reason ...` rather than deleting it")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_committed_results_json_matches_the_current_checker_source(self, results):
        current = harness._file_hash(*harness.CHECKER_FILES)
        assert results.get("provenance", {}).get("checker_sha256") == current, (
            "checker.py/aliases.py/claim_rules.py changed since benchmark/results.json was last "
            "generated. Re-run `python benchmark/fetch_data.py && python benchmark/run.py --reason "
            "...` and commit the new results.json + README together -- a stale results.json next to "
            "changed detection code is exactly the failure this test exists to catch.")

    def test_results_json_pins_the_exact_corpus_commit_and_checksums(self, results):
        prov = results.get("provenance", {})
        assert prov.get("corpus_commit") == harness.fetch_data.CORPUS_COMMIT
        assert prov.get("corpus_sha256") == harness.fetch_data.CHECKSUMS

    def test_results_json_records_the_pre_registered_decision(self, results):
        assert results.get("decision") in (
            "WORKS", "HIGH_PRECISION_FLAGGER_ONLY", "DOES_NOT_WORK")


class TestDecisionRule:
    """The tiering rule was written BEFORE any new detector was measured against the real test
    split -- these pin the rule's own behaviour, independent of any particular result."""

    def test_matches_the_conclusion_already_published_before_this_rule_existed(self):
        """Today's committed numbers: P 50.0 [36.6,63.4], R 2.7 [1.8,3.9], F1 5.0 [3.3,7.1], naive
        51.8%. Proof this rule was not backed out from an answer: it reproduces the conclusion the
        README already stated in prose before this function existed."""
        assert decision_verdict(0.366, 0.018, 0.033, 0.518) == "DOES_NOT_WORK"

    def test_works_tier(self):
        assert decision_verdict(p_lo=0.65, r_lo=0.55, f_lo=0.60, naive_f1=0.40) == "WORKS"

    def test_flagger_tier_high_precision_low_recall(self):
        assert decision_verdict(p_lo=0.65, r_lo=0.30, f_lo=0.55,
                                naive_f1=0.40) == "HIGH_PRECISION_FLAGGER_ONLY"

    def test_low_precision_alone_is_never_works_even_with_strong_recall_and_f1(self):
        assert decision_verdict(p_lo=0.40, r_lo=0.90, f_lo=0.55, naive_f1=0.30) == "DOES_NOT_WORK"

    def test_f1_at_or_below_the_floor_is_never_works_even_with_strong_precision_and_recall(self):
        assert decision_verdict(p_lo=0.80, r_lo=0.80, f_lo=0.30, naive_f1=0.40) == "DOES_NOT_WORK"


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

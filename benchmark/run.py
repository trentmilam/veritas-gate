"""Measure veritas-gate against RAGTruth, in generic-only mode.

    python benchmark/fetch_data.py                        # once
    python benchmark/run.py --reason "why this run happens"

--reason is required and permanent: this scores the SACRED test split, and every run appends one
line to the committed benchmark/TEST_SET_ACCESS_LOG.jsonl. Iterate freely against the train split
instead with `python benchmark/dev_run.py` -- nothing there needs a reason, because nothing there is
the number that gets published. See "TEST-SET ACCESS DISCIPLINE" below for why this exists.

CI never runs this file with real data. It runs `python benchmark/run.py --fixture` against a small
committed SYNTHETIC fixture (see benchmark/fixtures/) so the harness LOGIC is checked on every push
with no 36 MB download and no test-set access. The fixture is generated, not sampled: RAGTruth is a
derived corpus whose source passages come from CNN/DailyMail, MS MARCO and the Yelp Open Dataset,
each carrying upstream terms that RAGTruth's own MIT licence does not relicense, so no slice of it
is committed here. Regenerate with `python benchmark/fixtures/build_synthetic_fixture.py`.

WHAT THIS DOES AND DOES NOT MEASURE
-----------------------------------
veritas-gate was extracted from a resume-tailoring pipeline. Most of its checks encode
job-application knowledge -- forbidden skill lists, credential tables, employer attribution, a
hardcoded aerospace-employer index, an ATS rubric. Those are meaningless on news summarization and
running them here would produce a number about nothing.

Exactly two of its checks are corpus-generic, meaning their decision logic contains no domain
vocabulary and asks only "is this literal figure present in the supplied evidence":

    TC-3  unverified_metric   a %, $ or multiplier not found among the evidence figures
    TC-6  unverified_count    a count-noun-anchored magnitude not found among the evidence figures

Those two, and only those two, are enabled by default. The checker is constructed with every domain
parameter empty so the resume checks are inert, and predictions are additionally filtered to those
two violation types so nothing else can leak into the score.

THE CEILING, STATED UP FRONT
----------------------------
Both surviving checks are digit matchers. Measured on the corpus: only 20.8% of RAGTruth's 14,289
annotated hallucination spans contain any digit at all. The rest are fabricated names, relations,
entities and claims -- invisible to a digit matcher by construction. Recall against the full
hallucination label is therefore capped near 0.21 no matter how good the implementation is.

That is not a defect to tune away. It is the honest scope of what two numeric checks can do, and
the reason this report is titled "numeric groundedness", not "hallucination detection".

TC-6's count-noun list ships UNMODIFIED. It is resume vocabulary (postings, tests, commits, LOC)
and will barely fire on news text. Extending it by reading RAGTruth would be inventing detector
vocabulary from the evaluation corpus -- the exact in-sample tuning this benchmark exists to avoid.
The near-zero fire rate is accepted as an honest cost.

TEST-SET ACCESS DISCIPLINE
---------------------------
Nothing in a local script can PREVENT someone from re-running this against the test split until a
number looks good. What it can do is make every access visible: a run against the real corpus
requires `--reason "..."`, and that reason -- plus the sha256 of the detector source at the moment
of the run -- is appended to benchmark/TEST_SET_ACCESS_LOG.jsonl, a file that is committed and never
truncated. Six runs while tuning a threshold show up as six dated lines in a PR diff, not as
something a reviewer has no way to see.

The published benchmark/results.json also carries a `provenance.checker_sha256`: a CI test asserts
that hash matches the CURRENT checker source, so a change to checker.py/aliases.py/claim_rules.py
with a stale committed results.json now fails CI in the same PR, instead of silently drifting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_data  # noqa: E402
from metrics import clustered_bootstrap_f1, decision_verdict, pct, prf, wilson  # noqa: E402
from veritas_gate import TruthChecker  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "veritas_gate"
DATA = HERE / "data"
FIXTURES = HERE / "fixtures"
RESULTS_PATH = HERE / "results.json"
ACCESS_LOG = HERE / "TEST_SET_ACCESS_LOG.jsonl"

GENERIC_VIOLATIONS = {"unverified_metric", "unverified_count"}
# The three opt-in grounding.py checks -- broadened numeric, ungrounded entity, and
# content-word novelty window. Off by default in TruthChecker; _grounding_checker() below turns
# all three on so this violation set is exactly what fires.
GROUNDING_VIOLATIONS = {"unverified_number", "ungrounded_entity", "novel_content_window"}
MIN_SAMPLE = 200          # below this the run reports BLOCKED and no quality number

# The exact files that constitute "the detector" for provenance purposes. Anything that changes what
# gets scored belongs in this tuple -- if a new check is added to grounding.py and wired through
# checker.py, add its module here too, or the content-hash gate stops meaning what it claims to.
CHECKER_FILES = (SRC / "checker.py", SRC / "aliases.py", SRC / "claim_rules.py")
GROUNDING_FILES = CHECKER_FILES + (SRC / "grounding.py",)


def _file_hash(*paths: Path) -> str:
    """SHA-256 over the detector source, with line endings normalized to LF first.

    Hashing raw bytes makes this value depend on the CHECKOUT's line-ending policy rather than on
    the source's content: with `.gitattributes`' `text=auto eol=lf`, a Windows working tree that
    predates it still holds CRLF while a fresh Linux CI checkout gets LF, and the two hash
    differently for byte-identical code. That breaks the gate in the exact place it is supposed to
    work -- a CI run would fail the provenance assertion on a source file nobody edited, while the
    real question ("did the detector change?") went unanswered. A CRLF->LF conversion is by
    definition not a detector change, so it must not move this hash.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return h.hexdigest()


def evidence_for(source_row: dict) -> str:
    """The grounding context the human annotators judged against.

    NOT the ``source`` field -- that is only the dataset name ("MARCO", "CNN/DM", "Yelp"). The
    actual context is ``source_info``: a plain string for Summary, and a dict for QA (question +
    passages) and Data2txt (structured business record). Dicts are serialized rather than
    cherry-picked so every figure available to the generating model is in the evidence bank.
    """
    info = source_row.get("source_info")
    if isinstance(info, str):
        return info
    return json.dumps(info, ensure_ascii=False)


def digit_span_ceiling(responses: list) -> tuple[int, int]:
    """(spans containing a digit, total annotated spans) across ``responses``.

    This is the single most important number in the report when only the two digit checks are
    enabled. Both are digit matchers, so a hallucination span with no digit in it is invisible to
    them by construction -- this ratio is the hard ceiling on recall, independent of implementation
    quality. Computed here rather than quoted from a one-off script so the documented command
    reproduces every published figure.
    """
    total = with_digit = 0
    for r in responses:
        for span in r.get("labels") or ():
            total += 1
            if any(ch.isdigit() for ch in span.get("text", "")):
                with_digit += 1
    return with_digit, total


def load() -> tuple[list, dict]:
    """The real, pinned RAGTruth corpus. Re-verifies the checksum on EVERY call, not only at fetch
    time -- a file swapped or corrupted after `fetch_data.py` succeeded once must still be caught
    before it silently produces different numbers."""
    resp_path, src_path = DATA / "response.jsonl", DATA / "source_info.jsonl"
    if not resp_path.exists() or not src_path.exists():
        print("BLOCKED: corpus not present. Run `python benchmark/fetch_data.py` first.")
        raise SystemExit(2)
    for name, path in (("response.jsonl", resp_path), ("source_info.jsonl", src_path)):
        if not fetch_data.verify(name, path):
            print(f"BLOCKED: {name} does not match the pinned checksum for commit "
                  f"{fetch_data.CORPUS_COMMIT[:12]}. Refusing to score against a corpus that cannot "
                  f"be verified. Re-run `python benchmark/fetch_data.py`.")
            raise SystemExit(2)
    responses = [json.loads(l) for l in resp_path.open(encoding="utf-8")]
    sources = {}
    for line in src_path.open(encoding="utf-8"):
        row = json.loads(line)
        sources[row["source_id"]] = row
    return responses, sources


def load_fixture() -> tuple[list, dict]:
    """A small, committed, SYNTHETIC fixture -- see benchmark/fixtures/. Not a slice of the real
    test split and not third-party text: it is generated so the harness logic can be checked
    offline without redistributing corpus source passages. No network, no test-set access;
    regenerated only by `benchmark/fixtures/build_synthetic_fixture.py`, never by CI."""
    responses = [json.loads(l) for l in (FIXTURES / "response_sample.jsonl").open(encoding="utf-8")]
    sources = {}
    for line in (FIXTURES / "source_info_sample.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        sources[row["source_id"]] = row
    return responses, sources


def _default_checker(evidence: str) -> TruthChecker:
    return TruthChecker(experience_evidence=evidence)


def _grounding_checker(evidence: str) -> TruthChecker:
    """TC-3/TC-6 plus the three grounding.py checks, all opt-in flags on.
    ``GROUNDING_VIOLATIONS`` filters scoring to just the three new checks, matching how
    ``_default_checker``'s score is filtered to just TC-3/TC-6 via ``GENERIC_VIOLATIONS``."""
    return TruthChecker(experience_evidence=evidence, enable_broadened_numeric=True,
                        enable_entity_grounding=True, enable_novelty_check=True)


def score(test_rows: list, all_responses: list, sources: dict, *, min_sample: int = MIN_SAMPLE,
         checker_factory=_default_checker, violation_types=None) -> "dict | None":
    """Score ``test_rows`` (already filtered to the split under evaluation) against ``sources``.

    ``all_responses`` is the population ``digit_span_ceiling`` scans -- deliberately not just
    ``test_rows``, matching the original benchmark's behaviour of reporting the ceiling over
    whichever corpus was loaded, not only the scored slice.

    Pure and side-effect-free: no file writes, no stdout. Returns ``None`` (having printed BLOCKED)
    when ``test_rows`` is under ``min_sample`` -- this IS the guard `TestBlockedPath` exercises.
    """
    if len(test_rows) < min_sample:
        print(f"BLOCKED: {len(test_rows)} rows < {min_sample} required. No quality number reported.")
        return None

    violation_types = GENERIC_VIOLATIONS if violation_types is None else violation_types
    rows = []
    check_seconds = 0.0      # gate.check() alone: the steady-state cost per text
    full_seconds = 0.0       # + evidence serialization and checker construction: cold cost per pair
    for r in test_rows:
        src = sources.get(r["source_id"])
        if not src:
            continue
        t_cold = time.perf_counter()
        gate = checker_factory(evidence_for(src))
        t0 = time.perf_counter()
        result = gate.check(r["response"])
        t1 = time.perf_counter()
        check_seconds += t1 - t0
        full_seconds += t1 - t_cold
        fired = [v for v in result.violations if v.violation_type in violation_types]
        rows.append({
            "cluster": r["source_id"],
            "task": src["task_type"],
            "model": r.get("model"),
            "pred": bool(fired),
            "gold": bool(r.get("labels")),
        })

    n = len(rows)
    clusters = len({x["cluster"] for x in rows})
    digit_spans, all_spans = digit_span_ceiling(all_responses)
    gold_pos = sum(1 for x in rows if x["gold"])
    tp = sum(1 for x in rows if x["pred"] and x["gold"])
    fp = sum(1 for x in rows if x["pred"] and not x["gold"])
    fn = sum(1 for x in rows if not x["pred"] and x["gold"])
    overall = prf(tp, fp, fn)

    p_lo, p_hi = wilson(tp, tp + fp) if (tp + fp) else (0.0, 0.0)
    r_lo, r_hi = wilson(tp, tp + fn) if (tp + fn) else (0.0, 0.0)
    f_lo, f_hi = clustered_bootstrap_f1(rows, "cluster")

    base = gold_pos / n if n else 0.0
    naive_f1 = 2 * base / (base + 1.0) if (base + 1.0) else 0.0

    out = {
        "n": n, "source_clusters": clusters, "gold_positive": gold_pos, "base_rate": base,
        "digit_span_recall_ceiling": {
            "spans_with_digit": digit_spans, "spans_total": all_spans,
            "ratio": digit_spans / all_spans if all_spans else 0.0,
        },
        "overall": overall,
        "precision_wilson95": [p_lo, p_hi], "recall_wilson95": [r_lo, r_hi],
        "f1_clustered_bootstrap95": [f_lo, f_hi],
        "naive_always_positive_f1": naive_f1,
        "decision": decision_verdict(p_lo, r_lo, f_lo, naive_f1),
        "checks_enabled": sorted(violation_types),
        "by_task": {t: prf(sum(1 for x in rows if x["task"] == t and x["pred"] and x["gold"]),
                           sum(1 for x in rows if x["task"] == t and x["pred"] and not x["gold"]),
                           sum(1 for x in rows if x["task"] == t and not x["pred"] and x["gold"]))
                    for t in sorted({x["task"] for x in rows})},
    }
    # Deliberately NOT under a persisted top-level key -- wall-clock timing varies between runs by
    # construction, and `results.json`'s determinism claim (byte-identical across runs) depends on
    # every persisted field being a pure function of the input, not of when it happened to run.
    # `main()` strips this key before writing the file; `print_report` reads it separately. An
    # earlier version of this refactor put timing straight into the returned dict and broke exactly
    # this -- caught by the full-pipeline determinism test, not by inspection.
    out["_timing"] = {
        "check_seconds_per_response": check_seconds / n if n else 0.0,
        "full_seconds_per_response": full_seconds / n if n else 0.0,
    }
    return out


def print_report(out: dict, *, title: str = "GENERIC-ONLY MODE (numeric groundedness)",
                 fixture: bool = False) -> None:
    ceiling = out["digit_span_recall_ceiling"]
    overall = out["overall"]
    p_lo, p_hi = out["precision_wilson95"]
    r_lo, r_hi = out["recall_wilson95"]
    f_lo, f_hi = out["f1_clustered_bootstrap95"]

    print("=" * 78)
    corpus_name = "SYNTHETIC FIXTURE" if fixture else "RAGTruth"
    print(f"veritas-gate on {corpus_name} -- {title}")
    print("=" * 78)
    print(f"test responses      : {out['n']} (measured)")
    print(f"source clusters     : {out['source_clusters']} (measured)")
    print(f"hallucinated (gold) : {out['gold_positive']}  base rate {pct(out['base_rate'])} (measured)")
    print(f"checks enabled      : {out['checks_enabled']}")
    if ceiling["spans_total"]:
        print(f"recall ceiling      : {pct(ceiling['ratio'])} of {ceiling['spans_total']} annotated "
              f"spans contain a digit (measured)")
    print()
    print("RESULT (response-level, measured)")
    print(f"  fired on          : {overall['tp'] + overall['fp']} responses")
    print(f"  precision         : {pct(overall['precision'])}   Wilson 95% [{pct(p_lo)}, {pct(p_hi)}]")
    print(f"  recall            : {pct(overall['recall'])}   Wilson 95% [{pct(r_lo)}, {pct(r_hi)}]")
    print(f"  F1                : {pct(overall['f1'])}   source-clustered bootstrap 95% "
          f"[{pct(f_lo)}, {pct(f_hi)}]")
    print(f"  tp/fp/fn          : {overall['tp']}/{overall['fp']}/{overall['fn']}")
    print()
    print("MANDATORY FLOOR (computed)")
    print(f"  always-positive   : precision {pct(out['base_rate'])}  recall 100.0%  "
          f"F1 {pct(out['naive_always_positive_f1'])}")
    print(f"  -> DECISION: {out['decision']}  (pre-registered rule, scored on the LOWER confidence "
          f"bound -- see benchmark/metrics.py::decision_verdict)")
    print()
    timing = out.get("_timing")
    if timing:
        print(f"LATENCY (measured, this machine -- excluded from results.json on purpose, so that "
              f"file stays byte-identical across runs)")
        print(f"  gate.check() only : {timing['check_seconds_per_response'] * 1000:.3f} ms/response "
              f"(steady state: one checker reused across texts)")
        print(f"  + build & serialize: {timing['full_seconds_per_response'] * 1000:.3f} ms/response "
              f"(cold: a fresh checker and evidence bank per response -- what this harness actually "
              f"does)")
        print()
    print("BY TASK TYPE (measured)")
    for task, s in sorted(out["by_task"].items()):
        print(f"  {task:10} P {pct(s['precision']):>6}  R {pct(s['recall']):>6}  F1 {pct(s['f1']):>6}")


def _print_published_baselines() -> None:
    print("PUBLISHED BASELINES (RAGTruth paper, ACL 2024, Table 5 -- cited, not reproduced here)")
    for name, p, r_, f in (("Prompt GPT-3.5-turbo", 37.1, 92.3, 52.9),
                           ("Prompt GPT-4-turbo", 46.9, 97.9, 63.4),
                           ("SelfCheckGPT GPT-3.5", 49.7, 71.9, 58.8),
                           ("Finetuned Llama-2-13B", 76.9, 80.7, 78.7)):
        print(f"  {name:22} P {p:5.1f}  R {r_:5.1f}  F1 {f:5.1f}")


def _append_access_log(reason: str, checker_files: tuple = CHECKER_FILES) -> None:
    entry = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "checker_sha256": _file_hash(*checker_files),
        "reason": reason,
    }
    with ACCESS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _parse_args(argv: "list[str] | None") -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reason", default=None,
                   help="Why this run touches the real, sacred RAGTruth test split. Required unless "
                        "--fixture. Appended to a permanent, committed TEST_SET_ACCESS_LOG.jsonl.")
    p.add_argument("--fixture", action="store_true",
                   help="Score the committed offline fixture slice instead of the real corpus. No "
                        "network, no --reason, never writes results.json.")
    p.add_argument("--detector", choices=("generic", "grounding"), default="generic",
                   help="'generic' (default): TC-3/TC-6 digit checks only, writes results.json. "
                        "'grounding': the three grounding.py checks, writes "
                        "results_grounding.json -- a separate file so the generic-only baseline is "
                        "never silently overwritten by a different detector's numbers.")
    return p.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    grounding = args.detector == "grounding"
    checker_factory = _grounding_checker if grounding else _default_checker
    violation_types = GROUNDING_VIOLATIONS if grounding else GENERIC_VIOLATIONS
    checker_files = GROUNDING_FILES if grounding else CHECKER_FILES
    results_path = HERE / "results_grounding.json" if grounding else RESULTS_PATH
    title = "GROUNDING CHECKS" if grounding else "GENERIC-ONLY MODE (numeric groundedness)"

    if args.fixture:
        responses, sources = load_fixture()
    else:
        if not args.reason:
            print("BLOCKED: the real RAGTruth test split is scored deliberately rarely -- see "
                  "\"TEST-SET ACCESS DISCIPLINE\" in this file's module docstring. Pass "
                  "--reason \"...\" stating why this run is happening; it is appended to "
                  "benchmark/TEST_SET_ACCESS_LOG.jsonl, a permanent, committed record. Iterate "
                  "against the train split instead with `python benchmark/dev_run.py`.")
            return 2
        responses, sources = load()

    test = [r for r in responses if r.get("split") == "test"]
    out = score(test, responses, sources, checker_factory=checker_factory,
               violation_types=violation_types)
    if out is None:
        return 2

    print_report(out, title=title, fixture=args.fixture)
    print()
    _print_published_baselines()

    if not args.fixture:
        out["provenance"] = {
            "checker_sha256": _file_hash(*checker_files),
            "corpus_commit": fetch_data.CORPUS_COMMIT,
            "corpus_sha256": dict(fetch_data.CHECKSUMS),
        }
        persisted = {k: v for k, v in out.items() if k != "_timing"}
        results_path.write_text(json.dumps(persisted, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
        print(f"\nwrote benchmark/{results_path.name}")
        _append_access_log(args.reason, checker_files)
        print(f"logged this run to {ACCESS_LOG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

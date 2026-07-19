"""Measure veritas-gate against RAGTruth, in generic-only mode.

    python benchmark/fetch_data.py     # once
    python benchmark/run.py

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

Those two, and only those two, are enabled. The checker is constructed with every domain parameter
empty so the resume checks are inert, and predictions are additionally filtered to those two
violation types so nothing else can leak into the score.

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
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import clustered_bootstrap_f1, pct, prf, wilson  # noqa: E402
from veritas_gate import TruthChecker  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
GENERIC_VIOLATIONS = {"unverified_metric", "unverified_count"}
MIN_SAMPLE = 200          # below this the run reports BLOCKED and no quality number


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
    """(spans containing a digit, total annotated spans) across the whole corpus.

    This is the single most important number in the report. Both enabled checks are digit matchers,
    so a hallucination span with no digit in it is invisible to them by construction -- this ratio
    is the hard ceiling on recall, independent of implementation quality. It is computed here rather
    than quoted from a one-off script so the documented command reproduces every published figure.
    """
    total = with_digit = 0
    for r in responses:
        for span in r.get("labels") or ():
            total += 1
            if any(ch.isdigit() for ch in span.get("text", "")):
                with_digit += 1
    return with_digit, total


def load() -> tuple[list, dict]:
    resp_path, src_path = DATA / "response.jsonl", DATA / "source_info.jsonl"
    if not resp_path.exists() or not src_path.exists():
        print("BLOCKED: corpus not present. Run `python benchmark/fetch_data.py` first.")
        raise SystemExit(2)
    responses = [json.loads(l) for l in resp_path.open(encoding="utf-8")]
    sources = {}
    for line in src_path.open(encoding="utf-8"):
        row = json.loads(line)
        sources[row["source_id"]] = row
    return responses, sources


def main() -> int:
    responses, sources = load()
    test = [r for r in responses if r.get("split") == "test"]

    if len(test) < MIN_SAMPLE:
        print(f"BLOCKED: {len(test)} test rows < {MIN_SAMPLE} required. No quality number reported.")
        return 2

    rows = []
    check_seconds = 0.0      # gate.check() alone: the steady-state cost per text
    full_seconds = 0.0       # + evidence serialization and checker construction: cold cost per pair
    for r in test:
        src = sources.get(r["source_id"])
        if not src:
            continue
        t_cold = time.perf_counter()
        gate = TruthChecker(experience_evidence=evidence_for(src))
        t0 = time.perf_counter()
        result = gate.check(r["response"])
        t1 = time.perf_counter()
        check_seconds += t1 - t0
        full_seconds += t1 - t_cold
        fired = [v for v in result.violations if v.violation_type in GENERIC_VIOLATIONS]
        rows.append({
            "cluster": r["source_id"],
            "task": src["task_type"],
            "model": r.get("model"),
            "pred": bool(fired),
            "gold": bool(r.get("labels")),
        })

    n = len(rows)
    clusters = len({x["cluster"] for x in rows})
    digit_spans, all_spans = digit_span_ceiling(responses)
    gold_pos = sum(1 for x in rows if x["gold"])
    tp = sum(1 for x in rows if x["pred"] and x["gold"])
    fp = sum(1 for x in rows if x["pred"] and not x["gold"])
    fn = sum(1 for x in rows if not x["pred"] and x["gold"])
    overall = prf(tp, fp, fn)

    p_lo, p_hi = wilson(tp, tp + fp) if (tp + fp) else (0.0, 0.0)
    r_lo, r_hi = wilson(tp, tp + fn) if (tp + fn) else (0.0, 0.0)
    f_lo, f_hi = clustered_bootstrap_f1(rows, "cluster")

    base = gold_pos / n
    naive_f1 = 2 * base / (base + 1.0)

    print("=" * 78)
    print("veritas-gate on RAGTruth -- GENERIC-ONLY MODE (numeric groundedness)")
    print("=" * 78)
    print(f"test responses      : {n} (measured)")
    print(f"source clusters     : {clusters} (measured)")
    print(f"hallucinated (gold) : {gold_pos}  base rate {pct(base)} (measured)")
    print(f"checks enabled      : {sorted(GENERIC_VIOLATIONS)}")
    print(f"recall ceiling      : {pct(digit_spans / all_spans)} of {all_spans} annotated spans "
          f"contain a digit (measured) -- both checks are digit matchers, so this caps recall")
    print()
    print("RESULT (response-level, measured)")
    print(f"  fired on          : {tp + fp} responses")
    print(f"  precision         : {pct(overall['precision'])}   Wilson 95% [{pct(p_lo)}, {pct(p_hi)}]")
    print(f"  recall            : {pct(overall['recall'])}   Wilson 95% [{pct(r_lo)}, {pct(r_hi)}]")
    print(f"  F1                : {pct(overall['f1'])}   source-clustered bootstrap 95% "
          f"[{pct(f_lo)}, {pct(f_hi)}]")
    print(f"  tp/fp/fn          : {tp}/{fp}/{fn}")
    print()
    print("MANDATORY FLOOR (computed)")
    print(f"  always-positive   : precision {pct(base)}  recall 100.0%  F1 {pct(naive_f1)}")
    verdict = "BEATS" if overall["f1"] > naive_f1 else "DOES NOT BEAT"
    print(f"  -> this detector {verdict} the trivial classifier on F1")
    print()
    print("PUBLISHED BASELINES (RAGTruth paper, ACL 2024, Table 5 -- cited, not reproduced here)")
    for name, p, r_, f in (("Prompt GPT-3.5-turbo", 37.1, 92.3, 52.9),
                           ("Prompt GPT-4-turbo", 46.9, 97.9, 63.4),
                           ("SelfCheckGPT GPT-3.5", 49.7, 71.9, 58.8),
                           ("Finetuned Llama-2-13B", 76.9, 80.7, 78.7)):
        print(f"  {name:22} P {p:5.1f}  R {r_:5.1f}  F1 {f:5.1f}")
    print()
    print("LATENCY (measured, this machine -- deliberately NOT written to results.json, which must"
          " stay byte-identical across runs)")
    print(f"  gate.check() only : {check_seconds * 1000 / n:.3f} ms/response  (steady state: one "
          f"checker reused across texts)")
    print(f"  + build & serialize: {full_seconds * 1000 / n:.3f} ms/response  (cold: a fresh "
          f"checker and evidence bank per response -- what this harness actually does)")
    print()
    print("BY TASK TYPE (measured)")
    for task in sorted({x["task"] for x in rows}):
        sub = [x for x in rows if x["task"] == task]
        s = prf(sum(1 for x in sub if x["pred"] and x["gold"]),
                sum(1 for x in sub if x["pred"] and not x["gold"]),
                sum(1 for x in sub if not x["pred"] and x["gold"]))
        print(f"  {task:10} n={len(sub):5}  P {pct(s['precision']):>6}  R {pct(s['recall']):>6}  "
              f"F1 {pct(s['f1']):>6}")

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
        "checks_enabled": sorted(GENERIC_VIOLATIONS),
        "by_task": {t: prf(sum(1 for x in rows if x["task"] == t and x["pred"] and x["gold"]),
                           sum(1 for x in rows if x["task"] == t and x["pred"] and not x["gold"]),
                           sum(1 for x in rows if x["task"] == t and not x["pred"] and x["gold"]))
                    for t in sorted({x["task"] for x in rows})},
    }
    (Path(__file__).resolve().parent / "results.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\nwrote benchmark/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

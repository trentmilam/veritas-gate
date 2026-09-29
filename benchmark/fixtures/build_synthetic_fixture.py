"""Generate the committed offline CI fixture from synthetic data.

The fixture exists so CI can check the benchmark harness LOGIC on every push without network
access. It is NOT the published measurement: `benchmark/run.py --fixture` never writes
results.json (see its argparse help). The headline numbers come only from the full RAGTruth
corpus fetched by `benchmark/fetch_data.py` and scored by the scheduled benchmark-refresh
workflow.

Because the fixture only has to exercise the harness, it does not need to contain real corpus
text, and it should not. RAGTruth is a derived corpus whose source passages come from
CNN/DailyMail, MS MARCO and the Yelp Open Dataset, each carrying its own upstream terms that
RAGTruth's repository-level MIT licence does not relicense. Committing a verbatim slice of
those passages into this repository would redistribute third-party content this project has no
right to redistribute. So the committed fixture is generated here instead: same record shapes,
same task types, same scoring code paths, entirely invented content.

To rebuild the committed fixture:

    python benchmark/fixtures/build_synthetic_fixture.py

To score the REAL corpus (the actual benchmark), fetch it first; it is never committed:

    python benchmark/fetch_data.py
    python benchmark/run.py --reason "..."

Deterministic: a fixed seed, so regenerating always reproduces the same bytes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

SEED = 20260825
N_SOURCES = 378
N_RESPONSES = 386          # must stay >= run.MIN_SAMPLE (200) or the run reports BLOCKED

# A wholly invented domain. No real company, product, person or publication appears here.
UNITS = ["kPa", "degC", "rpm", "hours", "cycles", "mm", "kg", "litres"]
PARTS = ["Kestrel manifold", "VX-7 pressure transducer", "Larkspur seal ring",
         "Tamarin flow divider", "Bracken igniter housing", "Osprey feed line",
         "Wren throttle body", "Marlin heat exchanger", "Sable check valve",
         "Ferrous bellows coupling"]
SITES = ["Bay 4", "Bay 11", "Cell A", "Cell C", "Stand 2", "Stand 9"]
PROGRAMS = ["Northwind", "Halyard", "Cobalt Ridge", "Aldergate", "Pinemark"]
VERBS = ["completed", "was withdrawn from", "passed", "was re-run through", "was deferred from"]


def _summary_source(rng: random.Random, i: int) -> tuple[str, str]:
    part, site, prog = rng.choice(PARTS), rng.choice(SITES), rng.choice(PROGRAMS)
    cycles, press, hours = rng.randint(120, 9800), rng.randint(80, 4200), rng.randint(2, 480)
    body = (
        f"Qualification note {2000 + i}: the {part} on programme {prog} {rng.choice(VERBS)} "
        f"acceptance testing in {site}. The article accumulated {cycles} cycles at a working "
        f"pressure of {press} kPa over {hours} hours of stand time. No anomaly was recorded "
        f"against the seal interface. The responsible engineer signed the article off for "
        f"integration pending a witness review. A follow-on soak of {hours * 2} hours was "
        f"scheduled but not executed within this reporting period."
    )
    prompt = "Summarise the qualification note in two sentences."
    return body, prompt


def _qa_source(rng: random.Random, i: int) -> tuple[dict, str]:
    part, prog = rng.choice(PARTS), rng.choice(PROGRAMS)
    limit, actual = rng.randint(200, 900), rng.randint(200, 900)
    passages = [
        f"The {part} is rated to a proof pressure of {limit} kPa on programme {prog}.",
        f"During the most recent run the article saw {actual} kPa at the inlet flange.",
        f"Operators log every excursion above {limit} kPa in the stand deviation record.",
    ]
    q = f"What is the proof pressure rating of the {part}?"
    return {"question": q, "passages": passages}, q


def _data2txt_source(rng: random.Random, i: int) -> tuple[dict, str]:
    part = rng.choice(PARTS)
    rec = {
        "article": part,
        "site": rng.choice(SITES),
        "programme": rng.choice(PROGRAMS),
        "cycles_completed": rng.randint(50, 5000),
        "torque_setting_nm": rng.randint(4, 260),
        "inspections_passed": rng.randint(1, 40),
        "status": rng.choice(["accepted", "quarantined", "reworked"]),
    }
    prompt = "Write a short status paragraph describing this test record."
    return rec, prompt


def build_sources(rng: random.Random) -> list[dict]:
    rows, kinds = [], (["Summary"] * 126) + (["QA"] * 126) + (["Data2txt"] * 126)
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds[:N_SOURCES]):
        sid = f"syn-{i:05d}"
        if kind == "Summary":
            info, prompt = _summary_source(rng, i)
            source = "SyntheticNotes"
        elif kind == "QA":
            info, prompt = _qa_source(rng, i)
            source = "SyntheticQA"
        else:
            info, prompt = _data2txt_source(rng, i)
            source = "SyntheticRecords"
        rows.append({"source_id": sid, "task_type": kind, "source": source,
                     "prompt": prompt, "source_info": info})
    return rows


def _numbers_in(info) -> list[str]:
    """Every digit run that genuinely appears in the source, so a grounded claim can cite one."""
    import re
    return re.findall(r"\d+", json.dumps(info))


# Count nouns the TC-6 gate actually anchors on (checker._COUNT_NOUNS). A magnitude only counts
# as "significant" when it is >= 1000, carries a thousands separator, or takes a '+'/'k' suffix.
COUNT_NOUNS = ["tests", "records", "units", "documents", "files", "engines"]


def _labelled(text: str, span: str, rng: random.Random) -> dict:
    start = text.index(span)
    return {"start": start, "end": start + len(span), "text": span,
            "meta": "synthetic: figure absent from source",
            "label_type": rng.choice(["Evident Conflict", "Evident Baseless Info"]),
            "implicit_true": False, "due_to_null": False}


def build_responses(rng: random.Random, sources: list[dict]) -> list[dict]:
    """Mix grounded and invented numeric claims so every scoring branch is exercised.

    A fixture where the checker never fires proves the harness RUNS but not that it
    DISCRIMINATES: tp/fp would both be 0 and the regression test would pass even if the
    detector were deleted. So the mix is built deliberately to hit all four branches:

      A  ungrounded significant count + labelled  -> checker fires on a real hallucination (tp)
      B  ungrounded significant count, unlabelled -> checker fires on a clean response  (fp)
      C  labelled hallucination that is NOT count-noun-anchored -> checker stays silent  (fn)
      D  grounded figure, unlabelled              -> checker correctly stays silent (true neg)

    Branch C is the majority here, mirroring the real corpus: TC-3/TC-6 are resume-domain
    digit matchers, so most annotated hallucinations in prose fall straight through them. That
    is the finding the benchmark exists to report, not a defect in the fixture.
    """
    rows = []
    for i in range(N_RESPONSES):
        src = sources[i % len(sources)]
        nums = [n for n in _numbers_in(src["source_info"]) if len(n) >= 2]
        grounded = rng.choice(nums) if nums else "12"
        noun = rng.choice(COUNT_NOUNS)
        labels: list[dict] = []
        branch = i % 12

        if branch in (0, 3):                                    # A: tp candidates
            bogus = f"{rng.randint(11, 89)},{rng.randint(100, 999)}"
            text = (f"The article cleared {bogus} {noun} during the follow-on sequence and was "
                    f"signed off by the reviewing engineer without further comment.")
            labels.append(_labelled(text, bogus, rng))
        elif branch == 6:                                       # B: fp candidates
            bogus = f"{rng.randint(11, 89)},{rng.randint(100, 999)}"
            text = (f"The article cleared {bogus} {noun} during the follow-on sequence, which "
                    f"the stand log records as a nominal result.")
        elif branch in (1, 4, 7, 9):                             # C: fn candidates
            bogus = str(rng.randint(100000, 999999))
            text = (f"The article recorded {grounded} against the logged value and then "
                    f"sustained {bogus} {rng.choice(UNITS)} during the follow-on run, which "
                    f"the reviewing engineer accepted without further comment.")
            labels.append(_labelled(text, bogus, rng))
        else:                                                    # D: true negatives
            text = (f"The article recorded {grounded} against the logged value and completed "
                    f"the scheduled sequence with no deviation raised against it.")

        rows.append({
            "id": f"synres-{i:05d}", "source_id": src["source_id"], "response": text,
            "labels": labels, "split": "test", "model": f"synthetic-model-{i % 6}",
            "quality": "good", "temperature": 0.7,
        })
    return rows


def main() -> int:
    rng = random.Random(SEED)
    sources = build_sources(rng)
    responses = build_responses(rng, sources)

    (OUT / "source_info_sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sources), encoding="utf-8")
    (OUT / "response_sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in responses), encoding="utf-8")

    labelled = sum(1 for r in responses if r["labels"])
    print(f"wrote {len(sources)} synthetic sources and {len(responses)} synthetic responses "
          f"({labelled} carrying a labelled hallucination span)")
    print("now regenerate expected.json:  python benchmark/run.py --fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

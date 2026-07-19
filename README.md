# veritas-gate

**A deterministic, domain-configured gate for LLM-generated text, and an honest measurement of how far that gets you.**

![ci](https://github.com/trentmilam/veritas-gate/actions/workflows/ci.yml/badge.svg)

`veritas-gate` is a small, zero-dependency library that gates LLM output with deterministic rules
instead of an LLM judge. You supply an evidence bank and a set of things the subject may not claim;
it flags claims that contradict them, scores quality with a no-LLM rubric, and catches
repetition-runaway that passes both honesty checks and length floors.

It is fast, reproducible, and auditable. It is **not** a general hallucination detector, and this
README includes the benchmark that shows exactly where the line is.

```python
from veritas_gate import TruthChecker, rubric_score, is_degenerate

gate = TruthChecker(
    experience_evidence="Built a RAG system over a 35,000+ doc corpus that cut research time 85%.",
    forbidden_skills=["kubernetes", "lora", "fine-tuning"],   # things the subject has NOT done
    forbidden_claims="cloud-native\nfull-stack",
)

gate.check("Built a RAG pipeline over a 35,000+ document corpus.").is_valid        # True
gate.check("Expert in Kubernetes and LoRA fine-tuning.").is_valid                   # False (fabrication)
gate.check("I have no experience with Kubernetes.").is_valid                        # True (honest omission)

is_degenerate("AI futures, AI potentials, AI possibilities, " * 100)               # True
```

Run the full demo: `python -m veritas_gate.example`

## Measured against RAGTruth

Most tools in this space assert that rules beat an LLM judge and stop there. This one was measured
against [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., ACL 2024), a corpus of
human-annotated LLM responses labeled for hallucination against their source passages.

Reproduce it:

```bash
python benchmark/fetch_data.py   # 36 MB, not vendored
python benchmark/run.py
```

**What was measured.** Only the two checks whose decision logic contains no domain vocabulary:
`unverified_metric` (a %/$/multiplier absent from the evidence) and `unverified_count` (a
count-anchored magnitude absent from the evidence). Every résumé-specific rule was disabled, and
predictions were filtered to those two violation types so nothing else could leak into the score.

**Result** on the 2,700-response test split, 450 source clusters, 34.9% base rate:

| | precision | recall | F1 |
|---|---|---|---|
| veritas-gate, generic-only | 50.0% `[36.6, 63.4]` | 2.7% `[1.8, 3.9]` | **5.0%** `[3.3, 7.1]` |
| always-say-hallucinated | 34.9% | 100% | **51.8%** |
| Prompt GPT-3.5-turbo † | 37.1% | 92.3% | 52.9% |
| Prompt GPT-4-turbo † | 46.9% | 97.9% | 63.4% |
| Finetuned Llama-2-13B † | 76.9% | 80.7% | 78.7% |

† published in the RAGTruth paper, Table 5. Cited, not reproduced here.

Precision and recall intervals are Wilson 95%. The F1 interval is a bootstrap that resamples whole
source clusters, because six model responses share each source passage and a per-response interval
would assume an independence the corpus does not have.

**The generic checks do not beat a trivial classifier.** F1 5.0% against a 51.8% floor. That is the
honest headline and it is not going to be tuned away, because the reason is structural: both
surviving checks are digit matchers, and only 20.8% of RAGTruth's 14,289 annotated hallucination
spans contain a digit at all. The rest are fabricated names, relations and entities. Recall is
capped near 0.21 by construction.

The one number that holds up is precision: when it fires, it is right 50% of the time against a
34.9% base rate. The interval's lower bound is 36.6%, so even that is weak evidence of lift.

**Speed and determinism do hold.** Across three runs of the full test split: 0.288–0.298 ms per
response building a fresh checker and evidence bank each time, and 0.122–0.127 ms for `check()`
alone once a checker is reused. `results.json` is byte-identical across runs, timing excluded from
it on purpose. At a judge latency of 1–3 s that is three to four orders of magnitude cheaper, with
no drift. That part is real, and it is the honest reason to reach for rules: as a cheap
deterministic pre-filter inside a domain you have configured, not as a replacement for semantic
verification.

**An aside worth its own line.** The RAGTruth paper's prompt-GPT-3.5 baseline scores F1 52.9%. The
trivial always-say-hallucinated classifier scores 51.8% on the same split. A widely-cited
LLM-as-judge baseline beats "always answer yes" by 1.1 points. Whatever else this benchmark shows,
it is worth knowing what these numbers are being compared against.

**What was not measured.** The résumé-specific rules (forbidden skills, credentials, employer
attribution, the entity index, `rubric_score`) have no ground truth here and were excluded rather
than scored on a corpus they were not built for. Span-level localization was not attempted; the gate
returns a verdict, not character offsets.

## What it checks

**`TruthChecker`** — deterministic verification against a configured evidence bank:
- **Forbidden and over-claim phrases** — substring-robust, whitespace- and hyphen-normalized.
- **Not-claimable skills** — word-boundary matched, with an **honest-omission whitelist** so
  *"I have no experience with X"* is a disclosure, not a claim.
- **Unverified impact metrics and counts** — a `%`/`$`/`×`/large-count in the draft that is not in
  the evidence is flagged. These are the two checks the benchmark above measures.
- **Misattribution** (optional) — flags a personal-project signature under an employer block;
  config-driven, inert unless you supply the markers.
- **Credentials and named entities** — asserted but not evidenced, flagged.

Note what this list is not: it does not parse arbitrary prose into claims and verify each one. It
enforces the constraints you configure. The benchmark exists because that distinction matters and is
easy to blur.

**`rubric_score`** — a deterministic 0–100 quality score with no LLM judge, built on keyword
alignment without stuffing, title and intent alignment, real-metric density, and parseable
structure.

**`is_degenerate`** — catches repetition-runaway output (collapsed vocabulary, or a long run of
comma-items sharing a leading word) that passes both the honesty checks and any length floor.

**`check_claim_rules`** — a declarative registry where one row is both the rule the gate enforces
and the instruction the prompt renders, so the two cannot drift apart. See below.

## The claim registry, and where declarative rules stop working

When a pipeline keeps producing the same class of wrong claim, the usual fix is another hand-written
detector. Do that seventeen times and each nuance lives in two places that drift: the prompt re-types
the fact as an instruction, the checker encodes it as a regex. A policy written in a spec file
reaches the model immediately, because the model reads prose — and stays invisible to the gate until
it leaks into production and someone writes another regex.

```python
from veritas_gate import check_claim_rules, prompt_rules_block

rules = [{
    "id": "initiative-attribution",
    "violation_type": "attribution_error",
    "severity": "high",
    "prompt_instruction": "The retrieval platform was self-initiated; the migration was assigned.",
    "subjects": {
        "self_initiated": {"any_of": ["retrieval platform"]},
        "assigned_work":  {"any_of": ["migration workstream"]},
    },
    "forbid": [{
        "subject": "self_initiated",
        "within_sentence": ["was assigned", "tasked with"],
        "message": "the retrieval platform was self-initiated, not assigned",
    }],
}]

# TRUE sentence covering both subjects -> passes
check_claim_rules("I initiated the retrieval platform and was assigned to the migration "
                  "workstream.", rules)                                          # []
# the term attached to the wrong subject -> fires
check_claim_rules("I was assigned to the retrieval platform.", rules)            # 1 finding

prompt_rules_block(rules)   # the same row, rendered for the system prompt
```

**Nearest-marker attribution** is the mechanism worth stealing. One real sentence often covers
several subjects legitimately, so banning a term whenever the sentence *mentions* a subject would
flag correct writing. Co-occurrence is the wrong test; ownership is the right one. Each forbidden
term attaches to whichever subject's marker sits nearest it.

**Now the part most rules engines leave out.** The original goal was to retire every hand-written
detector into rows here. That was tried, measured against 9,963 real generated documents (36.7M
characters), and **rejected**. Rules here match literal lower-cased substrings with no word
boundaries, and at scale that is not a subtle problem:

| what happened | measured |
|---|---|
| a two-character term (`rl`) as a registry row vs. the equivalent regex | fired on **3,410 of 9,963 documents** vs **107** — a 31.9× blast radius, 3,328 pure false positives, because `rl` sits inside *world, girl, early, hourly, quarterly* |
| `done`, `grow`, `initial` as rows | match inside *abandoned, condone, undone*; *outgrew*; *uninitialized* |
| a single-word company-name ban | matched a different real company containing it as a substring |
| translating one numeric check into rows | **112 false negatives** on the exact incident class that check existed to catch |

So a rule belongs in the registry when its terms are multi-word and cannot occur inside a larger
word. A rule belongs in code when it needs word boundaries, cross-sentence state, occurrence
counting, open-ended numeric comparison, negative lookbehind, or the surrounding document. Adding a
boundary-sensitive rule to a registry like this does not make the gate stricter — it makes it wrong
in both directions at once, over-firing on substrings while under-firing on the real pattern.

That boundary is not a limitation to fix later. It is the finding.

## Design notes

- **Blocklist → allowlist tradeoff.** Skill claims are gated by a not-claimable blocklist with an
  omission whitelist; the honest "no experience with X" sentence must pass while the affirmative
  "expert in X" must fail. Per-clause detection prevents an honest sentence from whitelisting an
  affirmative claim elsewhere on the same line.
- **Calibrating anti-stuffing.** Aggregate keyword density wrongly penalizes a concise skills list,
  so the penalty keys on per-term over-repetition instead; only the excess is docked.
- **"Quantified" means a real impact metric.** Counting "any digit" rewards version numbers and
  years; the rubric counts only `%`/`$`/`×`/thousands magnitudes.
- **Catching runaway that passes everything else.** A repetition loop is truth-valid and long, so it
  needs its own detector: unique-token ratio plus a leading-word run check.
- **Vocabulary was not tuned on the benchmark.** The count-noun list ships unmodified even though it
  is résumé vocabulary that barely fires on news text. Extending it by reading RAGTruth would be
  inventing detector vocabulary from the evaluation corpus, which is the failure this measurement
  exists to avoid. The near-zero fire rate is the honest cost.

## Install and test

```bash
pip install -e ".[dev]"
pytest -q
```

Zero runtime dependencies; tests are pure-Python and offline. CI runs the suite on Python 3.10–3.12.

## License

MIT.

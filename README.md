# veritas-gate

**Deterministic truth-gating and quality scoring for LLM-generated text — a fast, reproducible alternative to LLM-as-judge.**

![ci](https://github.com/trentmilam/veritas-gate/actions/workflows/ci.yml/badge.svg)

LLM-as-judge is non-deterministic, slow, and expensive to run on every generation. `veritas-gate`
is a small, **zero-dependency** library that gates LLM output with **deterministic rules** instead:
it traces every claim to an evidence bank (blocking fabrication and over-claiming), scores quality
with a no-LLM rubric, and catches repetition-runaway that slips past both honesty checks and length
floors. It's built for production pipelines that need an **auditable, repeatable** gate — not a vibe
check — and as a reference for how to do **evals engineering** without an LLM judge.

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

## What it checks

**`TruthChecker`** — deterministic claim → evidence verification:
- **Fabrication / unsupported claims** — every claim must trace to the evidence bank.
- **Forbidden & over-claim phrases** — substring-robust, whitespace-normalized.
- **Not-claimable skills** — word-boundary matched, with an **honest-omission whitelist** so
  *"I have no experience with X"* is a disclosure, not a claim.
- **Unverified impact metrics & counts** — a `%`/`$`/`×`/large-count in the draft that isn't in the
  evidence is flagged (provenance, not just presence).
- **Misattribution** (optional) — flags a self-directed/personal-project signature listed under an
  employer block; config-driven, inert unless you supply the markers.
- **Credentials & named entities** — asserted but not evidenced → flagged.

**`rubric_score`** — a deterministic 0–100 quality score (no LLM judge), built on the levers that
actually move screening outcomes:
- **keyword alignment** to the target without **stuffing** (over-repetition is penalized),
- **title / intent alignment** (highest-weight real signal),
- **real-metric density** (impact figures, not "any digit" / years / versions),
- **parseable structure** (standard headers, no column-grid layouts).

**`is_degenerate`** — catches repetition-runaway output (collapsed vocabulary, or a long run of
comma-items sharing a leading word) that passes both the honesty checks and any length floor.

## Why deterministic?

- **Reproducible** — same input, same verdict, every time. No temperature, no judge drift.
- **Fast & free** — pure Python standard library, no model call, runs inline in a pipeline.
- **Auditable** — every violation names the rule and a concrete fix, so you can explain *why* a
  generation was gated (and write a regression test for it).

It is **not** a semantic-understanding oracle — it's a fast first line of defense that catches the
failure modes LLM pipelines hit most (fabrication, over-claiming, keyword-stuffing, runaway), which
you'd otherwise pay an LLM judge to catch non-deterministically.

## Design notes

A few decisions worth calling out (the interesting part):

- **Blocklist → allowlist tradeoff.** Skill claims are gated by a not-claimable blocklist with an
  omission whitelist; the honest "no experience with X" sentence must pass while the affirmative
  "expert in X" must fail. Per-clause omission detection prevents a separate honest sentence on the
  same line from whitelisting an affirmative claim elsewhere on it.
- **Calibrating anti-stuffing.** Aggregate keyword *density* wrongly penalizes a concise skills list
  (legitimately keyword-dense), so the penalty keys on **per-term over-repetition** instead — a
  central term may appear a few times; only the excess is docked.
- **"Quantified" means a real impact metric.** Counting "any digit" rewards version numbers, years,
  and "3 years"; the rubric counts only `%`/`$`/`×`/thousands/`k·m·b` magnitudes.
- **Catching runaway that passes everything else.** A repetition loop is truth-valid (no false
  claims) *and* long (passes any min-length floor), so it needs its own detector — unique-token
  ratio plus a leading-word run check.

## Install & test

```bash
pip install -e ".[dev]"
pytest -q
```

Zero runtime dependencies; tests are pure-Python and offline. CI runs the suite on Python 3.10–3.12
(`.github/workflows/ci.yml`).

## License

MIT.

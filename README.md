# veritas-gate

Deterministic, domain-configured gate for LLM-generated text.

Writeup: [trentmilam.dev/work/veritas-gate](https://trentmilam.dev/work/veritas-gate/)

![ci](https://github.com/trentmilam/veritas-gate/actions/workflows/ci.yml/badge.svg)

- Zero dependencies
- Rules instead of an LLM judge
- Input: an evidence bank and a list of things the subject may not claim
- Output: flagged contradictions, a no-LLM rubric score, repetition-runaway detection
- Not a general hallucination detector

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

Demo: `python -m veritas_gate.example`

## Benchmark: RAGTruth

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., ACL 2024): human-annotated LLM responses labeled for hallucination.

```bash
python benchmark/fetch_data.py   # 36 MB, fetched to a git-ignored dir, not vendored
python benchmark/run.py
```

| | what it is | what it measures | writes `results.json`? |
|---|---|---|---|
| `python benchmark/run.py` | RAGTruth evaluation, fetched corpus | generic-grounding performance: every number below | yes |
| `python benchmark/run.py --fixture` | offline run, committed **synthetic** fixture | loader, harness and scoring branches only | **no, never** |

- Fixture built by `benchmark/fixtures/build_synthetic_fixture.py`, no corpus text
- Corpus not vendored: see [benchmark/DATA-PROVENANCE.md](benchmark/DATA-PROVENANCE.md)

Scored:
- `unverified_metric`: a %/$/multiplier absent from the evidence
- `unverified_count`: a count-anchored magnitude absent from the evidence
- Résumé-specific rules disabled
- Predictions filtered to those two violation types

2,700-response test split, 450 source clusters, 34.9% base rate:

| | precision | recall | F1 |
|---|---|---|---|
| veritas-gate, generic-only | 50.0% `[36.6, 63.4]` | 2.7% `[1.8, 3.9]` | **5.0%** `[3.3, 7.1]` |
| always-say-hallucinated | 34.9% | 100% | **51.8%** |
| Prompt GPT-3.5-turbo † | 37.1% | 92.3% | 52.9% |
| Prompt GPT-4-turbo † | 46.9% | 97.9% | 63.4% |
| Finetuned Llama-2-13B † | 76.9% | 80.7% | 78.7% |

† RAGTruth paper, Table 5. Cited, not reproduced.

- Precision and recall intervals: Wilson 95%
- F1 interval: bootstrap over whole source clusters
- Generic checks do not beat a trivial classifier
- Both checks are digit matchers
- 20.8% of RAGTruth's 14,289 annotated hallucination spans contain a digit
- Recall capped near 0.21
- Speed, three runs of the full test split: 0.288-0.298 ms per response with a fresh checker and evidence bank, 0.122-0.127 ms for `check()` on a reused checker
- `results.json` byte-identical across runs, timing excluded
- vs 1-3 s LLM judge: 3-4 orders of magnitude cheaper
- Use: cheap deterministic pre-filter inside a configured domain

Not measured:
- Résumé-specific rules (forbidden skills, credentials, employer attribution, entity index, `rubric_score`)
- Span-level localization

## Grounding checks

`src/veritas_gate/grounding.py`. Opt-in, off by default:
`TruthChecker(..., enable_broadened_numeric=True, enable_entity_grounding=True, enable_novelty_check=True)`

- `unverified_number`: every digit token in the draft absent from the evidence
- `ungrounded_entity`: a capitalized token the evidence never mentions, sentence-openers excluded
- `novel_content_window`: a sliding window of stopword-filtered content words absent from the evidence. Window 10 tokens, 60% novel; train-split grid search over window sizes 3/5/7/10 and thresholds 0.6/0.8/1.0

```bash
python benchmark/fetch_data.py           # once, if not already done
python benchmark/tune.py                 # train-only CV, writes nothing
python benchmark/run.py --detector grounding --reason "..."
```

Same test split, ensemble vote-of-one across all three checks:

| | precision | recall | F1 |
|---|---|---|---|
| grounding ensemble | 40.1% `[38.1, 42.2]` | 96.0% `[94.5, 97.1]` | **56.6%** `[54.0, 59.2]` |
| always-say-hallucinated | 34.9% | 100% | **51.8%** |

- F1 lower bound 54.0%, trivial floor 51.8%
- Verdict: **DOES_NOT_WORK** (pre-registered rule in `benchmark/metrics.py`)
- `WORKS` and `HIGH_PRECISION_FLAGGER_ONLY` need a precision lower bound of at least 60%; this one is 38.1%
- Fires on 2,255 of 2,700 responses (84%)

Per task:

| task | n | naive F1 | ensemble F1 | verdict |
|---|---|---|---|---|
| Data2txt | 900 | 78.3% | 78.3% | fires on 100% of responses: ties the floor exactly, no signal |
| QA | 900 | 30.2% | 34.5% | real margin over the floor |
| Summary | 900 | 37.0% | 41.6% | real margin over the floor |

- Data2txt: recall 100.0%, precision 64.3% (equals the task base rate)
- QA and Summary beat their floors by 4.3 and 4.6 points
- Use: pre-filter that routes to a human or a judge model

## What it checks

`TruthChecker`, against a configured evidence bank:
- Forbidden and over-claim phrases: substring-robust, whitespace- and hyphen-normalized
- Not-claimable skills: word-boundary matched, honest-omission whitelist (*"I have no experience with X"*)
- Unverified impact metrics and counts: `%`/`$`/`×`/large-count not in the evidence
- Misattribution (optional): personal-project signature under an employer block; inert without markers
- Credentials and named entities: asserted but not evidenced
- Grounding checks (optional): broadened numeric, ungrounded entity, content-word novelty
- Claim registry (optional): `claim_rules=[...]`, enforced at check time

Enforces configured constraints only; does not parse prose into claims.

- `rubric_score`: deterministic 0-100 quality score. Keyword alignment without stuffing, title and intent alignment, real-metric density, parseable structure
- `is_degenerate`: repetition-runaway output (collapsed vocabulary, or a long run of comma-items sharing a leading word)
- `check_claim_rules`: declarative registry. One row is both the enforced rule and the rendered prompt instruction. Standalone, or via `TruthChecker(claim_rules=[...])`

## Claim registry

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

- Nearest-marker attribution: each forbidden term attaches to the subject whose marker sits nearest it
- Rules match literal lower-cased substrings, no word boundaries
- False positives: short term inside a longer word
- False negatives: rule translated out of code loses its boundary logic

Private-corpus run (2026, 9,963 documents, 36.7M chars; not vendored):
- `rl` fired on 3,410 of 9,963 documents as a registry row, 107 for the word-boundary regex (31.9×)
- `done`/`grow`/`initial` matched inside *abandoned*/*outgrew*/*uninitialized*
- A single-word company-name ban matched an unrelated company containing it
- Translating one numeric check into rows produced 112 false negatives

Public reproduction on the RAGTruth test split, `benchmark/registry_blast_radius.py`:

```bash
python benchmark/fetch_data.py                # once
python benchmark/registry_blast_radius.py      # public mode, writes registry_blast_radius.json
```

- `rl`: 0 word-boundary matches, **598 of 2,700 documents** by bare substring, 100% false positives
- Matched inside: airline, beverly, clearly, disorderly, earlier, girl, nearly, orlando, world
- `--root <path>`: private mode on your own documents, aggregate counts only, never the default

- Registry: multi-word terms
- Code: word boundaries, cross-sentence state, occurrence counting, open-ended numeric comparison, negative lookbehind, document context

## Design notes

- Skill claims: not-claimable blocklist with an omission whitelist, per-clause detection
- Anti-stuffing: per-term over-repetition penalty, only the excess is docked
- "Quantified": `%`/`$`/`×`/thousands magnitudes only, not version numbers or years
- Runaway detection: unique-token ratio plus a leading-word run check
- Count-noun list unmodified; barely fires on news text

## Install and test

```bash
pip install -e ".[dev]"
pytest -q
```

Zero runtime dependencies. Tests are pure-Python and offline. CI: Python 3.10-3.12.

## License

MIT.

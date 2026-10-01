# Benchmark data provenance

## Contents

- Committed: benchmark code, scoring harness, published results files, generated **synthetic** fixture (`fixtures/build_synthetic_fixture.py` and its output)
- Not committed: the RAGTruth corpus. `benchmark/data/` is git-ignored

```
python benchmark/fetch_data.py          # ~36 MB, pinned commit, SHA-256 verified
python benchmark/run.py --reason "..."  # scores the fetched corpus
```

## Two things run under `benchmark/`

| | source data | what it measures | writes `results.json`? |
|---|---|---|---|
| `python benchmark/run.py` | the RAGTruth corpus fetched into `benchmark/data/` | generic-grounding performance: every published number | yes |
| `python benchmark/run.py --fixture` | the committed synthetic fixture | loader, harness and scoring branches only | **no, never** |

- Fixture runs in CI on every push, no download; test split untouched
- Fixture `tp/fp/fn` are properties of generated data
- `--fixture` never writes `results.json`, labels its output `SYNTHETIC FIXTURE`, never produces `provenance.corpus_commit`
- Enforced by tests in `tests/test_benchmark.py`
- Every published benchmark number comes from the fetched corpus

## Upstream terms

RAGTruth is MIT-licensed (Copyright 2023 Particle Media). The licence covers the corpus authors' annotations and code. The corpus is derived from CNN/DailyMail, MS MARCO, the Yelp Open Dataset, and news sources.

| upstream source | terms that govern the passages |
|---|---|
| CNN/DailyMail | article text remains the publishers' copyright; the Apache-2.0 and MIT licences in that lineage cover the collection and processing scripts |
| Yelp Open Dataset | Yelp Dataset Terms of Use: academic use, revocable, and restricting redistribution |
| MS MARCO | non-commercial research use, "without extending any license or other intellectual property rights" |
| news sources | publisher copyright |

- Corpus fetched locally, not redistributed
- Obtain it from Particle Media directly, under their terms
- `fetch_data.py` states this at run time

## Reproducibility

`fetch_data.py` pins a commit and verifies SHA-256 over the served bytes, aborting on mismatch:

```
CORPUS_COMMIT = c103204b9ce28d6bbad859304bf30de72b8ed8fe
```

- `benchmark/run.py` re-verifies the checksums on every load
- `results.json` records `provenance.corpus_commit` and a hash of the detector source

## Citation

Niu et al., *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented
Language Models*, ACL 2024.
<https://arxiv.org/abs/2401.00396> · <https://github.com/ParticleMedia/RAGTruth> · MIT,
Copyright (c) 2023 Particle Media.

# Benchmark data provenance

## What this repository contains

**Committed:** the benchmark code, the scoring harness, the published results files, and a
generated **synthetic** fixture (`fixtures/build_synthetic_fixture.py` and its output).

**Not committed:** the RAGTruth corpus. `benchmark/data/` is git-ignored.

This repository does not redistribute the corpus. It is fetched on demand:

```
python benchmark/fetch_data.py          # ~36 MB, pinned commit, SHA-256 verified
python benchmark/run.py --reason "..."  # scores the fetched corpus
```

## Two things run under `benchmark/`

| | source data | what it measures | writes `results.json`? |
|---|---|---|---|
| `python benchmark/run.py` | the RAGTruth corpus fetched into `benchmark/data/` | generic-grounding performance — every published number | yes |
| `python benchmark/run.py --fixture` | the committed synthetic fixture | loader, harness and scoring branches only | **no, never** |

The fixture exists so CI can exercise the harness on every push without a 36 MB download and
without touching the test split. Its `tp/fp/fn` are properties of generated data and are not a
measurement of anything external. The `--fixture` path never writes `results.json`, labels its
output `SYNTHETIC FIXTURE`, and cannot produce the `provenance.corpus_commit` block that a real
run records — tests in `tests/test_benchmark.py` enforce each of those.

Every published benchmark number comes from the fetched corpus.

## Upstream terms

RAGTruth is MIT-licensed (Copyright 2023 Particle Media), and that licence covers the corpus
authors' own work: their annotations and code. RAGTruth is a derived corpus — it incorporates
passages originating from CNN/DailyMail, MS MARCO, the Yelp Open Dataset, and news sources.

| upstream source | terms that govern the passages |
|---|---|
| CNN/DailyMail | article text remains the publishers' copyright; the Apache-2.0 and MIT licences in that lineage cover the collection and processing scripts |
| Yelp Open Dataset | Yelp Dataset Terms of Use — academic use, revocable, and restricting redistribution |
| MS MARCO | non-commercial research use, "without extending any license or other intellectual property rights" |
| news sources | publisher copyright |

Because those upstream materials carry their own terms, the corpus is fetched locally rather
than redistributed here, and the committed fixture is generated data. Anyone reproducing the
benchmark obtains the corpus from Particle Media directly, under whatever terms apply to them.
`fetch_data.py` states this at run time.

## Reproducibility

`fetch_data.py` pins an explicit commit rather than a branch, and verifies SHA-256 over the exact
bytes served there, aborting on any mismatch:

```
CORPUS_COMMIT = c103204b9ce28d6bbad859304bf30de72b8ed8fe
```

`benchmark/run.py` re-verifies the same checksums on every load, so a swapped or truncated corpus
fails loudly instead of silently changing a published number. `results.json` records
`provenance.corpus_commit` alongside a hash of the detector source, so a results file can always
be tied to the corpus revision and the code that produced it.

## Citation

Niu et al., *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented
Language Models*, ACL 2024.
<https://arxiv.org/abs/2401.00396> · <https://github.com/ParticleMedia/RAGTruth> · MIT,
Copyright (c) 2023 Particle Media.

"""Metrics for the RAGTruth benchmark, with the interval discipline the claim requires.

This file makes two decisions that need explaining before you change them:

**Wilson, not Wald.** The naive normal-approximation interval extends outside [0, 1], collapses to
zero width exactly when the rate hits 0% or 100%, and its true coverage oscillates well below
nominal at small n. Wilson does none of that.

**Clustered bootstrap, not just Wilson.** RAGTruth pairs every source passage with six model
responses, so the 2,700 test responses are not 2,700 independent trials; they are 450 source
clusters. A response-level Wilson interval silently assumes independence and reports a band that is
too narrow. We report both: Wilson for comparability with published per-response numbers, and a
source-clustered bootstrap that resamples whole clusters, whose independence assumption actually
holds for this data.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict


def wilson(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if total <= 0:
        return (0.0, 0.0)
    p = count / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall, F1. A zero denominator yields 0.0 rather than an exception: a detector
    that never fires has precision 0, not undefined-and-therefore-excused."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def clustered_bootstrap_f1(rows: list, cluster_key: str, n_boot: int = 2000,
                           seed: int = 20260719) -> tuple[float, float]:
    """95% F1 interval that respects the source clustering.

    ``rows`` are dicts carrying ``pred`` (bool), ``gold`` (bool) and ``cluster_key``. Whole clusters
    are resampled with replacement, so the interval reflects the real number of independent units
    (source passages) rather than the inflated response count.

    The seed is fixed: this benchmark's headline claim is determinism, so its own error bars must
    reproduce exactly across runs.
    """
    by_cluster: dict = defaultdict(list)
    for r in rows:
        by_cluster[r[cluster_key]].append(r)
    clusters = list(by_cluster.values())
    if not clusters:
        return (0.0, 0.0)

    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(len(clusters)):
            for r in clusters[rng.randrange(len(clusters))]:
                if r["pred"] and r["gold"]:
                    tp += 1
                elif r["pred"]:
                    fp += 1
                elif r["gold"]:
                    fn += 1
        stats.append(prf(tp, fp, fn)["f1"])
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return (lo, hi)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# Pre-registered 2026-08-08, before any new detector was measured against the real test split;
# see the design doc this benchmark's grounding checks were built from. Fixed thresholds, chosen
# independently of any observed outcome, so a result cannot be scored by a rule quietly picked to
# fit it. Every comparison uses the LOWER confidence bound, never the point estimate, specifically
# so a lucky small-sample run cannot cross a tier by noise alone.
_WORKS_MIN_RECALL = 0.50
_WORKS_MIN_PRECISION = 0.60


def decision_verdict(p_lo: float, r_lo: float, f_lo: float, naive_f1: float) -> str:
    """Classify a measured result into one of three tiers.

    WORKS                        : f_lo > naive_f1 AND r_lo >= 0.50 AND p_lo >= 0.60. Trustworthy
                                     as a standalone detector.
    HIGH_PRECISION_FLAGGER_ONLY  : p_lo >= 0.60 AND f_lo > naive_f1 AND r_lo < 0.50. Right when it
                                     fires, but silent on too much of the problem to rely on its
                                     silence; usable as a pre-filter ahead of a judge or a human,
                                     not as a replacement for one.
    DOES_NOT_WORK                : otherwise. This also covers "F1's upper bound can't clear the
                                     floor" without taking the upper bound as an argument: f_lo <=
                                     f_hi always, so f_lo failing to clear naive_f1 already implies
                                     it, and there is no comparison this function makes against f_hi
                                     directly.

    Sanity check against the number that already existed when this rule was written (precision
    50.0% [36.6, 63.4], recall 2.7% [1.8, 3.9], F1 5.0% [3.3, 7.1] vs a 51.8% naive floor):
    ``decision_verdict(0.366, 0.018, 0.033, 0.518) == "DOES_NOT_WORK"``, matching the conclusion
    that was already published, proving this rule was not backed out from an answer.
    """
    if f_lo > naive_f1 and r_lo >= _WORKS_MIN_RECALL and p_lo >= _WORKS_MIN_PRECISION:
        return "WORKS"
    if p_lo >= _WORKS_MIN_PRECISION and f_lo > naive_f1 and r_lo < _WORKS_MIN_RECALL:
        return "HIGH_PRECISION_FLAGGER_ONLY"
    return "DOES_NOT_WORK"

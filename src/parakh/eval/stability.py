from __future__ import annotations

import numpy as np


def population_stability_index(
    baseline: np.ndarray, comparison: np.ndarray, buckets: int = 10, epsilon: float = 1e-4
) -> float:
    """Population Stability Index between a baseline and a comparison sample.

    Bucketing uses the baseline's quantile edges (the standard 10-bucket PSI), so
    each baseline bucket carries roughly equal mass. Empty buckets are floored to
    ``epsilon`` before the log so PSI stays finite. Rule-of-thumb reading: below
    0.10 no material shift, 0.10-0.25 moderate shift, above 0.25 significant shift.
    """
    baseline = np.asarray(baseline, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)
    base_share = np.clip(base_counts / max(base_counts.sum(), 1), epsilon, None)
    comp_share = np.clip(comp_counts / max(comp_counts.sum(), 1), epsilon, None)

    return float(np.sum((comp_share - base_share) * np.log(comp_share / base_share)))


def score_stability_by_cohort(
    scores: np.ndarray,
    cohorts: np.ndarray,
    baseline_cohort: int | None = None,
    buckets: int = 10,
    threshold: float = 0.10,
) -> dict[str, object]:
    """PSI of the score distribution for each cohort against a baseline cohort.

    The baseline defaults to the earliest cohort present. Returns the per-cohort
    PSI, the maximum PSI across cohorts, and the share of non-baseline cohorts
    that stay within ``threshold`` (target < 0.10). Values are reported as
    measured; a cohort above the threshold is flagged, not smoothed away.
    """
    scores = np.asarray(scores, dtype=float)
    cohorts = np.asarray(cohorts)
    unique = sorted(int(c) for c in np.unique(cohorts))
    if baseline_cohort is None:
        baseline_cohort = unique[0]

    base = scores[cohorts == baseline_cohort]
    per_cohort: dict[str, float] = {}
    for c in unique:
        if c == baseline_cohort:
            continue
        psi = population_stability_index(base, scores[cohorts == c], buckets=buckets)
        per_cohort[str(c)] = round(psi, 4)

    values = list(per_cohort.values())
    max_psi = round(max(values), 4) if values else 0.0
    within = sum(1 for v in values if v < threshold)
    share_within = round(within / len(values), 4) if values else 1.0

    return {
        "baseline_cohort": int(baseline_cohort),
        "threshold": threshold,
        "per_cohort_psi": per_cohort,
        "max_psi": max_psi,
        "share_within_threshold": share_within,
        "cohorts_over_threshold": [c for c, v in per_cohort.items() if v >= threshold],
    }

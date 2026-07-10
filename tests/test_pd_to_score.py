from __future__ import annotations

import numpy as np

from parakh.scoring.model import (
    ODDS_REFERENCE,
    SCORE_CEILING,
    SCORE_FLOOR,
    SCORE_REFERENCE,
    pd_to_score,
)


def test_high_pd_maps_to_low_score():
    assert pd_to_score(0.9) < pd_to_score(0.1)


def test_score_is_monotone_decreasing_in_pd():
    grid = np.linspace(0.01, 0.99, 50)
    scores = np.array([int(pd_to_score(p)) for p in grid])
    assert np.all(np.diff(scores) <= 0)


def test_reference_odds_anchor():
    # PD at the reference 3:1 good-to-bad odds (PD = 1 / (1 + odds)) maps to the
    # documented SCORE_REFERENCE anchor.
    reference_pd = 1.0 / (1.0 + ODDS_REFERENCE)
    assert int(pd_to_score(reference_pd)) == SCORE_REFERENCE


def test_pd_above_069_no_longer_collapses_to_floor():
    # The previous linear 100 - 130 * PD collapsed every PD >= 0.69 to the floor.
    # The log-odds mapping keeps PD 0.69 clearly above the floor.
    assert int(pd_to_score(0.69)) > SCORE_FLOOR


def test_scores_stay_within_display_band():
    for p in (1e-6, 0.001, 0.5, 0.999, 1 - 1e-6):
        value = int(pd_to_score(p))
        assert SCORE_FLOOR <= value <= SCORE_CEILING


def test_vectorized_input_matches_scalar():
    grid = np.array([0.1, 0.3, 0.6])
    vectorized = pd_to_score(grid)
    scalar = np.array([int(pd_to_score(p)) for p in grid])
    assert np.array_equal(vectorized, scalar)

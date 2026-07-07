from parakh.monitoring.trajectory import HORIZON, WATCH_THRESHOLD, simulate


def test_healthy_firm_stays_flat_and_unflagged():
    traj = simulate(88, {"gst_turnover_decline_3m": -0.05, "epfo_headcount_trend": 0.06, "xf_gst_bank_gap": 0.05})
    assert len(traj.scores) == HORIZON
    assert traj.flag_month is None
    assert not traj.deteriorating


def test_stressed_firm_slides_and_flags():
    traj = simulate(46, {"gst_turnover_decline_3m": 0.12, "epfo_headcount_trend": -0.08, "xf_gst_bank_gap": 0.38})
    assert traj.deteriorating
    assert traj.scores[0] > traj.scores[-1]
    assert traj.flag_month is not None
    assert traj.scores[traj.flag_month - 1] < WATCH_THRESHOLD

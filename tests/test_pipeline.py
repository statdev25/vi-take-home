"""Unit tests for the score-layer logic (pure functions, no data needed).

These guard the behaviour we rely on: divertibility sinks late-funnel patients and
decays with recency, the hard outreach rules force reach to zero, BMI parses from Z68
codes, and calibration metrics are sane. Training is exercised by `uv run main.py train`.
"""

import numpy as np
import pandas as pd

from assignment import pipeline as P


def _row(**kw):
    base = dict(imaging_flag=0, pt_referral_prior=0, injection_flag=0,
                surgical_consult_flag=0, days_since_first_lbp=0, recent_outreach=0,
                opt_out_flag=0, decline_block=0, completion_block=0)
    base.update(kw)
    return pd.DataFrame([base])


def test_divertibility_sinks_late_funnel():
    early = P.divertibility(_row())[0]                          # LBP only, just entered
    imaging = P.divertibility(_row(imaging_flag=1))[0]
    referral = P.divertibility(_row(pt_referral_prior=1))[0]
    consult = P.divertibility(_row(surgical_consult_flag=1))[0]
    assert early > imaging > referral > consult                 # deeper funnel -> lower
    assert consult < 0.1                                        # surgical consult is pushed down
    assert 0 < consult <= early <= 1.0


def test_divertibility_recency_decays():
    fresh = P.divertibility(_row(days_since_first_lbp=0))[0]
    old = P.divertibility(_row(days_since_first_lbp=1000))[0]
    assert fresh > old
    assert old >= 0.29                                          # floored ~0.3


def test_reach_hard_rules_force_zero():
    assert P.reach(_row(opt_out_flag=1))[0] == 0.0
    assert P.reach(_row(decline_block=1))[0] == 0.0
    assert P.reach(_row(completion_block=1))[0] == 0.0
    assert P.reach(_row(recent_outreach=0))[0] == 1.0           # fresh, eligible
    assert P.reach(_row(recent_outreach=3))[0] == 0.25          # fatigue discount


def test_z68_bmi_parsing():
    assert P._z68_bmi("Z68.26") == 26.0
    assert P._z68_bmi("Z68.41") == 42.0
    assert P._z68_bmi("Z68.1") == 19.0


def test_calib_metrics_perfect_is_well_calibrated():
    # predictions that exactly match observed frequencies -> slope ~1, ECE ~0
    rng = np.random.RandomState(0)
    p = np.clip(rng.beta(2, 30, 5000), 1e-4, 0.99)
    y = (rng.uniform(size=p.size) < p).astype(int)
    ece, slope, _ = P._calib_metrics(y, p)
    assert ece < 0.05
    assert 0.6 < slope < 1.4

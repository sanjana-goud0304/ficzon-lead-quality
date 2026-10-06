"""Leakage tests for out-of-time scoring (synthetic data only)."""
import numpy as np
import pandas as pd

from ficzon.features import FEATURE_SETS
from ficzon.scoring import SCORE_SETS, build_scores

CFG = {"split": {"train_end": "2018-08-31 23:59:59", "val_end": "2018-09-30 23:59:59"},
       "min_category_count": 5, "random_seed": 42}


def frozen():
    sets = {n: {"columns": FEATURE_SETS[n], "model": "Logistic regression", "params": {"C": 1.0}, "threshold": 0.5}
            for n in SCORE_SETS}
    return {"cv_months": ["2018-07", "2018-08", "2018-09"], "sets": sets}


def synth(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    created = pd.Series(pd.date_range("2018-05-01", "2018-11-10", periods=n))
    loc = rng.choice(["Bangalore", "Other Locations", "Chennai"], n)
    y = (rng.random(n) < np.where(loc == "Bangalore", 0.6, 0.25)).astype(int)
    return pd.DataFrame({
        "lead_id": np.arange(1, n + 1), "Created": created, "target_high": y, "Source_clean": rng.choice(["Call", "Web"], n),
        "created_hour_bucket": rng.choice(["09-12", "13-16"], n), "created_dow": rng.choice(["Monday", "Friday"], n),
        "Product": rng.choice(["P1", "P2"], n), "Location": loc, "Delivery_Mode": rng.choice(["Mode-1", "Mode-5"], n),
        "email_valid": rng.integers(0, 2, n)})


def test_sources_and_coverage():
    df = synth()
    out = build_scores(df, frozen(), CFG)
    assert len(out) == len(df) and out["lead_id"].is_unique
    m = out.merge(df[["lead_id", "Created"]], on="lead_id")
    assert (m.loc[m.Created < "2018-07-01", "score_source"] == "none").all()          # nothing earlier to train on
    assert (m.loc[(m.Created >= "2018-07-01") & (m.Created < "2018-10-01"), "score_source"] == "cv_oof").all()
    assert (m.loc[m.Created >= "2018-10-01", "score_source"] == "final_model").all()
    scored = m[m.score_source != "none"]
    assert scored[[f"score_{n}" for n in SCORE_SETS]].notna().all().all()
    assert scored[[f"score_{n}" for n in SCORE_SETS]].stack().between(0, 1).all()


def test_scores_do_not_depend_on_a_leads_own_or_later_labels():
    """Flip every label from August onward: July scores must not move; August scores may only change via earlier months (they must not)."""
    df = synth()
    base = build_scores(df, frozen(), CFG).set_index("lead_id")
    flipped = df.copy()
    late = flipped.Created >= "2018-08-01"
    flipped.loc[late, "target_high"] = 1 - flipped.loc[late, "target_high"]
    alt = build_scores(flipped, frozen(), CFG).set_index("lead_id")
    created = df.set_index("lead_id")["Created"]
    july = created[(created >= "2018-07-01") & (created < "2018-08-01")].index
    aug = created[(created >= "2018-08-01") & (created < "2018-09-01")].index
    col = "score_post_contact"
    np.testing.assert_allclose(base.loc[july, col], alt.loc[july, col])      # trained before August
    np.testing.assert_allclose(base.loc[aug, col], alt.loc[aug, col])        # trained on Jul and earlier only


def test_test_scores_ignore_test_labels():
    df = synth()
    base = build_scores(df, frozen(), CFG).set_index("lead_id")
    flipped = df.copy()
    test = flipped.Created >= "2018-10-01"
    flipped.loc[test, "target_high"] = 1 - flipped.loc[test, "target_high"]
    alt = build_scores(flipped, frozen(), CFG).set_index("lead_id")
    ids = df.loc[test, "lead_id"]
    np.testing.assert_allclose(base.loc[ids, "score_post_contact"], alt.loc[ids, "score_post_contact"])

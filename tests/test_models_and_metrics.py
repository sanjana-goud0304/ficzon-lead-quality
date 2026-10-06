"""Tests for baselines, metrics and end-to-end model pipelines (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from ficzon.evaluate import bootstrap_ci, evaluate_scores, topk_metrics
from ficzon.features import FEATURE_SETS
from ficzon.models import SourceRateBaseline, build_baselines, build_candidates


def synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.choice(["Call", "Web", "Referral"], n, p=[0.5, 0.4, 0.1])
    base = pd.Series(src).map({"Call": 0.3, "Web": 0.35, "Referral": 0.9}).to_numpy()
    y = (rng.random(n) < base).astype(int)
    df = pd.DataFrame({
        "Source_clean": src,
        "created_hour_bucket": rng.choice(["00-08", "09-12", "13-16"], n),
        "created_dow": rng.choice(["Monday", "Tuesday", "Sunday"], n),
        "Product": rng.choice(["P1", "P2", None], n),
        "Location": rng.choice(["Bangalore", "Other Locations", None], n),
        "Delivery_Mode": rng.choice(["Mode-1", "Mode-5"], n),
        "email_valid": rng.integers(0, 2, n),
    })
    return df, y


def test_source_rate_baseline_smooths_and_handles_unseen():
    df, y = synth()
    m = SourceRateBaseline().fit(df, y)
    p = m.predict_proba(df)[:, 1]
    assert p[df.Source_clean == "Referral"].mean() > p[df.Source_clean == "Call"].mean()
    new = pd.DataFrame({"Source_clean": ["Never Seen", None]})
    assert np.allclose(m.predict_proba(new)[:, 1], m.prior_)


def test_topk_perfect_ranking_and_random_ties():
    y = np.array([1] * 20 + [0] * 80)
    perfect = topk_metrics(y, y.astype(float), 0.2)
    assert perfect["precision_at_k"] == 1.0 and perfect["recall_at_k"] == 1.0 and perfect["lift_at_k"] == pytest.approx(5.0)
    const = topk_metrics(np.tile([1, 0, 0, 0, 0], 400), np.zeros(2000), 0.2)   # constant score -> ~ prevalence
    assert const["lift_at_k"] == pytest.approx(1.0, abs=0.2)


def test_threshold_metrics_confusion_matrix_and_f1():
    from ficzon.evaluate import threshold_metrics

    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert m["flag_rate"] == pytest.approx(3 / 8)
    none_flagged = threshold_metrics(y, s, 2.0)
    assert none_flagged["tp"] == 0 and none_flagged["f1"] == 0.0 and np.isnan(none_flagged["precision"])


def test_bootstrap_interval_contains_point_estimate():
    from sklearn.metrics import roc_auc_score
    df, y = synth()
    score = np.random.default_rng(1).random(len(y)) + y * 0.3
    lo, hi = bootstrap_ci(y, score, roc_auc_score)
    assert lo <= roc_auc_score(y, score) <= hi


@pytest.mark.parametrize("fset", list(FEATURE_SETS))
def test_all_candidates_fit_and_predict_on_both_feature_sets(fset):
    df, y = synth()
    cols = FEATURE_SETS[fset]
    cols = [c for c in cols if c in df.columns]
    tr, va = df.iloc[:400], df.iloc[400:]
    for name, pipe in build_candidates(cols, min_count=5).items():
        pipe.fit(tr[cols], y[:400])
        p = pipe.predict_proba(va[cols])[:, 1]
        assert p.shape == (200,) and np.isfinite(p).all(), name
        assert set(evaluate_scores(y[400:], p)) >= {"roc_auc", "pr_auc", "lift_at_k"}


def test_baselines_run():
    df, y = synth()
    for name, pipe in build_baselines().items():
        pipe.fit(df.iloc[:400], y[:400])
        assert pipe.predict_proba(df.iloc[400:]).shape == (200, 2), name

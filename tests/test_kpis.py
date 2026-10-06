"""KPI definition tests (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from ficzon.kpis import lead_kpis, status_dimension, topk_kpis

CFG = {"target": {"high_potential": ["CONVERTED", "Potential"], "low_potential": ["Junk Lead", "Not Responding", "Open", "LOST"]},
       "kpi_definitions": {"converted": ["CONVERTED"], "no_engagement": ["Junk Lead", "Not Responding"], "unresolved": ["Open"]}}


def test_status_dimension_flags_follow_config():
    d = status_dimension(CFG).set_index("status")
    assert d.loc["Potential", "lead_category"] == "High Potential" and d.loc["Open", "lead_category"] == "Low Potential"
    assert d.loc["CONVERTED", "is_converted"] == 1 and d.loc["Potential", "is_converted"] == 0
    assert d.loc["Junk Lead", "is_no_engagement"] == 1 and d.loc["Open", "is_unresolved"] == 1
    assert d.index.is_unique


def test_lead_kpis_hand_computed():
    leads = pd.DataFrame({"status": ["CONVERTED"] * 2 + ["Potential"] * 3 + ["Junk Lead"] * 3 + ["Open"] + ["LOST"]})
    k = lead_kpis(leads, status_dimension(CFG))
    assert k["total_leads"] == 10 and k["high_potential_leads"] == 5
    assert k["high_potential_rate"] == pytest.approx(0.5) and k["conversion_rate"] == pytest.approx(0.2)
    assert k["no_engagement_rate"] == pytest.approx(0.3) and k["unresolved_share"] == pytest.approx(0.1)
    # Wilson 95% interval for 5/10 is a known value: (0.2366, 0.7634)
    assert k["high_rate_wilson_lower"] == pytest.approx(0.2366, abs=1e-3) and k["high_rate_wilson_upper"] == pytest.approx(0.7634, abs=1e-3)


def test_lead_kpis_rejects_unmapped_status_and_handles_empty():
    with pytest.raises(ValueError):
        lead_kpis(pd.DataFrame({"status": ["Mystery"]}), status_dimension(CFG))
    empty = lead_kpis(pd.DataFrame({"status": []}), status_dimension(CFG))
    assert empty["total_leads"] == 0 and np.isnan(empty["high_potential_rate"])


def test_topk_is_deterministic_with_ties_and_rounds_half_up():
    s = pd.DataFrame({"lead_id": [5, 4, 3, 2, 1], "is_high": [0, 1, 1, 0, 1], "score": [0.9, 0.9, 0.9, 0.1, 0.1]})
    k = topk_kpis(s, "score", 0.5)                       # 0.5 * 5 = 2.5 -> k = 3 (half up)
    assert k["top_k"] == 3
    # ties at 0.9 are all inside the top 3; high among them: ids 4 and 3 -> 2 of 3
    assert k["high_in_top_k"] == 2 and k["precision_at_top"] == pytest.approx(2 / 3)
    assert k["recall_at_top"] == pytest.approx(2 / 3) and k["lift_at_top"] == pytest.approx((2 / 3) / (3 / 5))
    tie = topk_kpis(pd.DataFrame({"lead_id": [2, 1], "is_high": [1, 0], "score": [0.5, 0.5]}), "score", 0.5)
    assert tie["top_k"] == 1 and tie["high_in_top_k"] == 0   # tie broken by lead_id ascending -> lead 1 first


def test_topk_ignores_unscored_leads():
    s = pd.DataFrame({"lead_id": [1, 2, 3, 4], "is_high": [1, 0, 1, 0], "score": [0.8, 0.2, np.nan, np.nan]})
    k = topk_kpis(s, "score", 0.5)
    assert k["scored_leads"] == 2 and k["top_k"] == 1 and k["precision_at_top"] == 1.0

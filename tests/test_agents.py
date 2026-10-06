"""Synthetic checks for the segment-adjusted agent comparison."""
import numpy as np
import pandas as pd

from ficzon.agents import adjusted_agent_gaps


def make(n_per=300, seed=0, lift_c=0.0):
    """Agents A, B, C handle an 'easy' (70% High) and a 'hard' (20% High) segment.
    A is routed mostly to easy, B mostly to hard, C evenly. C gets +lift_c on every lead."""
    rng = np.random.default_rng(seed)
    rows = []
    for agent, share_easy, lift in [("A", 0.9, 0.0), ("B", 0.1, 0.0), ("C", 0.5, lift_c)]:
        for seg, base in [("easy", 0.7), ("hard", 0.2)]:
            n = int(n_per * (share_easy if seg == "easy" else 1 - share_easy)) + 40
            y = (rng.random(n) < min(base + lift, 0.99)).astype(int)
            rows.append(pd.DataFrame({"Sales_Agent": agent, "seg": seg, "target_high": y}))
    return pd.concat(rows, ignore_index=True)


def test_routing_difference_disappears_after_adjustment_when_skill_is_equal():
    df = make(lift_c=0.0)
    raw = df.groupby("Sales_Agent").target_high.mean()
    assert raw["A"] - raw["B"] > 0.2                       # raw rates differ a lot purely because of routing
    out = adjusted_agent_gaps(df, ["seg"]).set_index("agent")
    assert out["gap_pp"].abs().max() < 8                   # ...but all adjusted gaps are near zero


def test_genuinely_better_agent_has_the_highest_gap():
    out = adjusted_agent_gaps(make(lift_c=0.25), ["seg"]).set_index("agent")
    assert out["gap_pp"].idxmax() == "C" and out.loc["C", "gap_pp"] > 10
    assert out.loc["C", "gap_lo"] > 0                      # interval excludes zero


def test_strong_agent_lowers_everyone_elses_gap():
    """Peer-relative gaps shift when a strong agent is in the pool; dropping them removes that shift."""
    df = make(lift_c=0.25)
    with_c = adjusted_agent_gaps(df, ["seg"]).set_index("agent")
    without_c = adjusted_agent_gaps(df[df.Sales_Agent != "C"], ["seg"]).set_index("agent")
    assert with_c.loc["A", "gap_pp"] < without_c.loc["A", "gap_pp"] - 10
    assert abs(without_c.loc["A", "gap_pp"]) < 8


def test_thin_strata_are_excluded_not_guessed():
    df = pd.DataFrame({"Sales_Agent": ["A"] * 50 + ["B"] * 5, "seg": ["x"] * 55, "target_high": [1, 0] * 27 + [1]})
    out = adjusted_agent_gaps(df, ["seg"], min_peer_leads=20).set_index("agent")
    assert out.loc["A", "n_comparable"] == 0 and np.isnan(out.loc["A", "gap_pp"])   # only 5 peer leads
    assert out.loc["B", "n_comparable"] == 5 and np.isnan(out.loc["B", "gap_pp"])   # too few to estimate


def test_missing_agent_rows_are_ignored():
    df = make(); df.loc[:10, "Sales_Agent"] = None
    out = adjusted_agent_gaps(df, ["seg"])
    assert out.n_leads.sum() == df.Sales_Agent.notna().sum()

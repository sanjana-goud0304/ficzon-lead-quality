"""Segment-adjusted comparison of sales agents.

For each lead, the comparison rate is the High rate of OTHER agents' leads in the same
stratum (leave-agent-out), so an agent is never compared with their own results. An agent's
adjusted gap is (observed High rate) - (expected rate given the strata of their leads).
Leads in strata with too few other-agent leads are excluded and counted, not guessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjusted_agent_gaps(df: pd.DataFrame, strata: list[str], agent: str = "Sales_Agent", target: str = "target_high",
                        min_peer_leads: int = 20, n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    d = df.dropna(subset=[agent]).copy()
    g_all = d.groupby(strata)[target].agg(S="sum", N="size")
    g_agent = d.groupby(strata + [agent])[target].agg(s="sum", n="size")
    d = d.join(g_all, on=strata).join(g_agent, on=strata + [agent])
    d["peer_n"] = d["N"] - d["n"]
    d["expected"] = np.where(d["peer_n"] >= min_peer_leads, (d["S"] - d["s"]) / d["peer_n"].where(d["peer_n"] > 0), np.nan)
    rng = np.random.default_rng(seed)
    rows = []
    for a, sub in d.groupby(agent):
        comp = sub.dropna(subset=["expected"])
        row = {"agent": a, "n_leads": len(sub), "n_comparable": len(comp), "comparable_%": len(comp) / len(sub) * 100,
               "unadjusted_high_%": sub[target].mean() * 100}
        if len(comp) >= 30:
            y, e = comp[target].to_numpy(), comp["expected"].to_numpy()
            gap = (y - e).mean() * 100
            boots = [(y[b] - e[b]).mean() * 100 for b in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.update({"observed_%": y.mean() * 100, "expected_%": e.mean() * 100, "gap_pp": gap, "gap_lo": lo, "gap_hi": hi})
        else:
            row.update({"observed_%": np.nan, "expected_%": np.nan, "gap_pp": np.nan, "gap_lo": np.nan, "gap_hi": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_leads", ascending=False).reset_index(drop=True)

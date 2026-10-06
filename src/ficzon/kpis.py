"""Business KPI definitions in pandas. The Power BI DAX measures mirror these exactly, and
`scripts/export_powerbi.py` uses this module to publish expected values for checking the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eda import wilson


def status_dimension(cfg: dict) -> pd.DataFrame:
    """One row per status with its category and KPI flags, generated from configs/config.yaml."""
    high = set(cfg["target"]["high_potential"])
    kd = cfg["kpi_definitions"]
    statuses = list(cfg["target"]["high_potential"]) + list(cfg["target"]["low_potential"])
    return pd.DataFrame({
        "status": statuses,
        "lead_category": ["High Potential" if s in high else "Low Potential" for s in statuses],
        "is_converted": [int(s in kd["converted"]) for s in statuses],
        "is_no_engagement": [int(s in kd["no_engagement"]) for s in statuses],
        "is_unresolved": [int(s in kd["unresolved"]) for s in statuses],
        "sort_order": range(1, len(statuses) + 1),
    })


def lead_kpis(leads: pd.DataFrame, dim: pd.DataFrame) -> dict[str, float]:
    """KPIs for a set of leads (needs a `status` column). Rates are fractions in [0, 1]."""
    d = leads[["status"]].merge(dim, on="status", how="left")
    if d["lead_category"].isna().any():
        raise ValueError("Unmapped status values: " + ", ".join(sorted(d.loc[d.lead_category.isna(), "status"].unique())))
    n = len(d)
    high = int((d["lead_category"] == "High Potential").sum())
    out = {"total_leads": n, "high_potential_leads": high,
           "high_potential_rate": high / n if n else np.nan,
           "converted_leads": int(d["is_converted"].sum()),
           "conversion_rate": d["is_converted"].sum() / n if n else np.nan,
           "no_engagement_rate": d["is_no_engagement"].sum() / n if n else np.nan,
           "unresolved_share": d["is_unresolved"].sum() / n if n else np.nan}
    if n:
        lo, hi = wilson(np.array([high]), np.array([n]))
        out["high_rate_wilson_lower"], out["high_rate_wilson_upper"] = float(lo[0]), float(hi[0])
    else:
        out["high_rate_wilson_lower"] = out["high_rate_wilson_upper"] = np.nan
    return out


def topk_kpis(scored: pd.DataFrame, score_col: str, frac: float = 0.2) -> dict[str, float]:
    """Precision / recall / lift when the top `frac` of scored leads are worked first.

    Deterministic: sort by score descending, then lead_id ascending; k = floor(frac * n + 0.5)
    (round half up, matching DAX ROUND). `scored` needs lead_id, is_high (0/1) and the score column.
    """
    s = scored.dropna(subset=[score_col])
    n = len(s)
    k = int(np.floor(frac * n + 0.5))
    top = s.sort_values([score_col, "lead_id"], ascending=[False, True]).head(k)
    high_all, high_top = int(s["is_high"].sum()), int(top["is_high"].sum())
    base = high_all / n if n else np.nan
    precision = high_top / k if k else np.nan
    return {"scored_leads": n, "top_k": k, "high_in_top_k": high_top, "precision_at_top": precision,
            "recall_at_top": high_top / high_all if high_all else np.nan, "base_rate": base,
            "lift_at_top": precision / base if base else np.nan}

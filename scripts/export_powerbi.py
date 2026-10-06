"""Phase 8: export PII-free tables for the Power BI model, plus expected KPI values to validate the DAX.

Writes (powerbi/data/ is gitignored because it is row-level data derived from customer records):
  powerbi/data/fact_leads.csv    one row per lead, no Mobile / EMAIL / contact key
  powerbi/data/dim_status.csv    status -> High/Low Potential and KPI flags (generated from configs/config.yaml)
  powerbi/data/fact_scores.csv   out-of-time model scores only (cv_oof for Jun-Sep, final_model for Oct-Nov)
  powerbi/data/agent_gaps.csv    segment-adjusted agent gaps (static, from Python)
and (aggregates only, safe to commit):
  powerbi/expected_kpis.csv      pandas values for every KPI/slice, to compare with the report's cards and tables

Usage:  python scripts/export_powerbi.py
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ficzon import eda  # noqa: E402
from ficzon.agents import adjusted_agent_gaps  # noqa: E402
from ficzon.cleaning import clean_leads  # noqa: E402
from ficzon.data import REPO_ROOT, assert_no_pii, load_config  # noqa: E402
from ficzon.features import add_features  # noqa: E402
from ficzon.kpis import lead_kpis, status_dimension, topk_kpis  # noqa: E402
from ficzon.scoring import SCORE_SETS, build_scores  # noqa: E402
from ficzon.splits import temporal_masks  # noqa: E402

warnings.filterwarnings("ignore")
MISSING = "(missing)"
MIN_SLICE = 30


def main() -> None:
    cfg = load_config()
    frozen = json.loads((REPO_ROOT / "reports" / "phase6_frozen_config.json").read_text(encoding="utf-8"))
    with closing(sqlite3.connect(str(REPO_ROOT / cfg['paths']['sqlite_db']))) as con:
        raw = pd.read_sql_query("SELECT * FROM leads", con, parse_dates=["Created"])
    df = add_features(clean_leads(raw, cfg)).sort_values(["Created", "lead_id"]).reset_index(drop=True)
    masks = temporal_masks(df["Created"], cfg)
    df["split"] = np.select([masks["train"], masks["val"], masks["test"]], ["train", "validation", "test"], default="unassigned")
    if (df["split"] == "unassigned").any():
        raise ValueError("Some leads fall outside the train/validation/test masks")

    dim = status_dimension(cfg)
    unknown = set(df["Status"]) - set(dim["status"])
    if unknown:
        raise ValueError(f"Statuses missing from dim_status: {sorted(unknown)}")

    fact = pd.DataFrame({
        "lead_id": df["lead_id"], "created_at": df["Created"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "created_date": df["Created"].dt.strftime("%Y-%m-%d"), "month_start": df["Created"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d"),
        "source_raw": df["Source"].fillna(MISSING), "source_group": df["Source_clean"].fillna(MISSING),
        "product": df["Product"].fillna(MISSING), "location": df["Location"].fillna(MISSING),
        "delivery_mode": df["Delivery_Mode"], "sales_agent": df["Sales_Agent"].fillna(MISSING),
        "status": df["Status"], "split": df["split"]})

    sc = build_scores(df, frozen, cfg)
    fact_scores = sc[sc["score_source"] != "none"].merge(fact[["lead_id", "split"]], on="lead_id")
    fact_scores = fact_scores[["lead_id", "split", "score_source"] + [f"score_{n}" for n in SCORE_SETS]]
    for c in [f"score_{n}" for n in SCORE_SETS]:
        fact_scores[c] = fact_scores[c].round(6)

    dev = df[df["Created"] <= pd.Timestamp(cfg["split"]["val_end"])].copy()
    dev["month"] = dev["Created"].dt.strftime("%Y-%m")
    dev["Location_g"] = eda.group_levels(dev["Location"], 30); dev["Mode_g"] = eda.group_levels(dev["Delivery_Mode"], 30)
    gaps = adjusted_agent_gaps(dev, ["Mode_g", "Location_g", "month"]).round(2)
    gaps.insert(1, "basis", "Apr-Sep 2018, Delivery_Mode x Location x month strata")

    out_dir = REPO_ROOT / "powerbi" / "data"; out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in [("fact_leads", fact), ("dim_status", dim), ("fact_scores", fact_scores), ("agent_gaps", gaps)]:
        assert_no_pii(t)
        t.to_csv(out_dir / f"{name}.csv", index=False)

    # ---- expected KPI values (the same definitions the DAX implements) ----
    f = fact.rename(columns={"status": "status"})
    rows = []
    def add(scope: str, label: str, leads: pd.DataFrame) -> None:
        for k, v in lead_kpis(leads, dim).items():
            rows.append({"scope": scope, "slice": label, "metric": k, "value": v})
    add("overall", "All leads", f)
    for m, g in f.groupby(f["month_start"].str[:7]): add("month", m, g)
    for dim_col, scope in [("source_group", "source_group"), ("delivery_mode", "delivery_mode"), ("location", "location"), ("sales_agent", "sales_agent"), ("product", "product")]:
        for v, g in f.groupby(dim_col):
            if len(g) >= MIN_SLICE: add(scope, str(v), g)
    # rolling 3-month High rate (pooled counts over the current and two previous calendar months, as in DAX DATESINPERIOD)
    mon = f.assign(month=f["month_start"].str[:7]).merge(dim, on="status").groupby("month").agg(leads=("lead_id", "size"), high=("lead_category", lambda s: (s == "High Potential").sum()))
    for m, v in (mon.high.rolling(3, min_periods=1).sum() / mon.leads.rolling(3, min_periods=1).sum()).items():
        rows.append({"scope": "month", "slice": m, "metric": "high_rate_rolling_3m", "value": v})
    # model KPIs on out-of-time scores
    sc_full = fact_scores.merge(df[["lead_id", "target_high"]].rename(columns={"target_high": "is_high"}), on="lead_id")
    for scope_name, sub in [("test_months_final_model", sc_full[sc_full.score_source == "final_model"]), ("all_scored_cv_oof_plus_test", sc_full)]:
        for n in SCORE_SETS:
            for k, v in topk_kpis(sub, f"score_{n}", 0.2).items():
                rows.append({"scope": f"model:{scope_name}", "slice": n, "metric": k, "value": v})
    exp = pd.DataFrame(rows)
    exp["value"] = exp["value"].astype(float).round(6)
    exp.to_csv(REPO_ROOT / "powerbi" / "expected_kpis.csv", index=False)

    print(f"fact_leads {len(fact):,} rows | fact_scores {len(fact_scores):,} rows | dim_status {len(dim)} rows | agent_gaps {len(gaps)} rows | expected_kpis {len(exp):,} values")
    ov = exp[(exp.scope == "overall")].set_index("metric").value
    print(ov.round(4).to_string())
    print(exp[exp.scope.str.startswith("model:test")].pivot_table(index="metric", columns="slice", values="value").round(3).to_string())


if __name__ == "__main__":
    main()

"""Every named query in sql/*.sql must run on a synthetic `leads` table and return sensible aggregates."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ficzon.sqlutil import parse_queries

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def synthetic_db():
    rng = np.random.default_rng(1)
    n = 400
    status = rng.choice(["CONVERTED", "Potential", "Junk Lead", "Not Responding", "Open", "LOST"], n, p=[.1, .2, .25, .2, .05, .2])
    df = pd.DataFrame({
        "lead_id": np.arange(1, n + 1),
        "Created": pd.date_range("2018-05-01", periods=n, freq="9h").strftime("%Y-%m-%d %H:%M:%S"),
        "Product_ID": rng.choice([15.0, 18.0, np.nan], n), "Source": rng.choice(["Call", "Website", None], n),
        "Sales_Agent": rng.choice(["Sales-Agent-4", "Sales-Agent-9"], n), "Location": rng.choice(["Bangalore", "Other Locations"], n),
        "Delivery_Mode": rng.choice(["Mode-1", "Mode-5"], n), "Status": status})
    df["target_high"] = df.Status.isin(["CONVERTED", "Potential"]).astype(int)
    con = sqlite3.connect(":memory:")
    df.to_sql("leads", con, index=False)
    return con, df


@pytest.mark.parametrize("sql_file", ["02_business_analysis.sql"])
def test_every_named_query_runs(sql_file):
    con, _ = synthetic_db()
    queries = parse_queries((SQL_DIR / sql_file).read_text(encoding="utf-8"))
    assert len(queries) >= 8 and len({n for n, _ in queries}) == len(queries)
    for name, q in queries:
        out = pd.read_sql_query(q, con)
        assert len(out) >= 1, name


def test_kpi_overview_matches_pandas():
    con, df = synthetic_db()
    q = dict(parse_queries((SQL_DIR / "02_business_analysis.sql").read_text(encoding="utf-8")))["kpi_overview"]
    r = pd.read_sql_query(q, con).iloc[0]
    assert r.total_leads == len(df) and r.high_potential_leads == df.target_high.sum()
    assert r.converted_leads == (df.Status == "CONVERTED").sum()
    assert r.no_engagement_rate_pct == pytest.approx(100 * df.Status.isin(["Junk Lead", "Not Responding"]).mean(), abs=0.01)


def test_rolling_window_uses_three_months_of_counts():
    con, df = synthetic_db()
    q = dict(parse_queries((SQL_DIR / "02_business_analysis.sql").read_text(encoding="utf-8")))["monthly_kpis_with_rolling_3m"]
    r = pd.read_sql_query(q, con).sort_values("month").reset_index(drop=True)
    df["month"] = df.Created.str[:7]
    g = df.groupby("month").agg(n=("lead_id", "size"), h=("target_high", "sum")).sort_index()
    expected = (g.h.rolling(3, min_periods=1).sum() / g.n.rolling(3, min_periods=1).sum() * 100).round(1).to_numpy()
    np.testing.assert_allclose(r.high_rate_rolling_3m_pct.to_numpy(), expected)


def test_extract_template_never_selects_raw_pii():
    text = (SQL_DIR / "00_mysql_extract_template.sql").read_text(encoding="utf-8")
    select_part = text.split("SELECT", 1)[1]
    # EMAIL and Mobile may only appear inside the hash and the two derived flags, never as bare selected columns
    for line in select_part.splitlines():
        stripped = line.strip().rstrip(",")
        assert stripped not in {"EMAIL", "Mobile"}
    assert "password" not in text.lower().replace("$db_password", "")

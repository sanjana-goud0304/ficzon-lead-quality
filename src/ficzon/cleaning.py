"""Cleaning: standardize labels and types without dropping any row."""
from __future__ import annotations

import numpy as np
import pandas as pd

STRING_COLS = ["Source", "Sales_Agent", "Location", "Delivery_Mode", "Status"]


def clean_leads(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a cleaned copy. Guarantees the row count is unchanged.

    - trims whitespace in text columns; empty strings become missing
    - consolidates Source labels via cfg['source_map'] (kept in config, not code)
    - turns Product_ID (float because of NaN) into a categorical label 'P<id>'
    - parses Created if it is still a string
    """
    n_before = len(df)
    out = df.copy()
    for col in STRING_COLS:
        if col in out.columns:
            s = out[col].astype("object")
            s = s.where(s.isna(), s.astype(str).str.strip())
            out[col] = s.replace({"": np.nan})
    out["Source_clean"] = out["Source"].replace(cfg.get("source_map", {}))
    out["Product"] = out["Product_ID"].map(lambda v: f"P{int(v)}" if pd.notna(v) else np.nan)
    if not np.issubdtype(out["Created"].dtype, np.datetime64):
        out["Created"] = pd.to_datetime(out["Created"])
    assert len(out) == n_before, "cleaning must not drop rows"
    return out

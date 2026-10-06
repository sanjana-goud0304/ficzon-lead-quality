"""Time-ordered train / validation / test split."""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_masks(created: pd.Series, cfg: dict) -> dict[str, pd.Series]:
    """Boolean masks for train, val, test by creation time. They are disjoint and cover every row."""
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    if not train_end < val_end:
        raise ValueError("train_end must be earlier than val_end")
    train = created <= train_end
    val = (created > train_end) & (created <= val_end)
    test = created > val_end
    masks = {"train": train, "val": val, "test": test}
    total = train.astype(int) + val.astype(int) + test.astype(int)
    assert (total == 1).all(), "split masks must be disjoint and exhaustive"
    return masks


def expanding_month_folds(created: pd.Series, val_months: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: for each month M in val_months, train on every row created BEFORE
    the first day of M and validate on rows created in M. Returns positional index arrays.

    No fold ever trains on a row from the same month as, or later than, its validation rows.
    """
    month = created.dt.strftime("%Y-%m").to_numpy()
    start = created.dt.to_period("M").dt.start_time.to_numpy()
    folds = []
    for m in val_months:
        val_idx = np.flatnonzero(month == m)
        if len(val_idx) == 0:
            raise ValueError(f"No rows in validation month {m}")
        first_day = pd.Period(m).start_time.to_datetime64()
        train_idx = np.flatnonzero(created.to_numpy() < first_day)
        if len(train_idx) == 0:
            raise ValueError(f"No training rows before {m}")
        folds.append((train_idx, val_idx))
    return folds

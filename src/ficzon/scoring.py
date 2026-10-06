"""Out-of-time lead scores for the frozen models.

Every score is produced by a model that never saw the lead's own month or any later month:
  * Jun-Sep 2018 leads: expanding-window cross-validation predictions ("cv_oof")
  * Oct 2018 onward (test): the final model refit on all train+validation rows ("final_model")
  * Apr-May 2018 leads: no score (nothing earlier to train on)
In-sample scores are never produced, so a dashboard built on them cannot overstate performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import build_estimator
from .splits import expanding_month_folds, temporal_masks

SCORE_SETS = ["creation_time", "assignment_time", "post_contact"]


def build_scores(df: pd.DataFrame, frozen: dict, cfg: dict) -> pd.DataFrame:
    """df must be cleaned + featured and contain lead_id, Created, target_high. Returns one row per lead."""
    df = df.sort_values(["Created", "lead_id"]).reset_index(drop=True)
    masks = temporal_masks(df["Created"], cfg)
    dev_idx = np.flatnonzero((masks["train"] | masks["val"]).to_numpy())
    test_idx = np.flatnonzero(masks["test"].to_numpy())
    dev = df.iloc[dev_idx].reset_index(drop=True)
    folds = expanding_month_folds(dev["Created"], frozen["cv_months"])
    ydev = dev["target_high"].to_numpy()
    min_count, seed = cfg["min_category_count"], cfg["random_seed"]

    out = pd.DataFrame({"lead_id": df["lead_id"].to_numpy(), "score_source": "none"})
    for name in SCORE_SETS:
        spec = frozen["sets"][name]
        cols = spec["columns"]
        scores = np.full(len(df), np.nan)
        for tr, va in folds:
            pipe = build_estimator(spec["model"], spec["params"], cols, min_count, seed).fit(dev.iloc[tr][cols], ydev[tr])
            scores[dev_idx[va]] = pipe.predict_proba(dev.iloc[va][cols])[:, 1]
        final = build_estimator(spec["model"], spec["params"], cols, min_count, seed).fit(dev[cols], ydev)
        scores[test_idx] = final.predict_proba(df.iloc[test_idx][cols])[:, 1]
        out[f"score_{name}"] = scores
    scored = out[[f"score_{n}" for n in SCORE_SETS]].notna().all(axis=1).to_numpy()
    src = np.where(scored, "cv_oof", "none").astype(object)
    src[test_idx] = "final_model"
    out["score_source"] = src
    return out

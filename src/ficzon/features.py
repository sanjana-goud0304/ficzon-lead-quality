"""Leakage-aware feature engineering and the sklearn preprocessor.

Feature tiers
-------------
Tier A  - believed to be known when the lead is created (source, creation time).
Tier B  - groups whose timing or routing is NOT confirmed by the project owner
          (product, location, delivery mode, sales agent, completeness flags,
          prior contact history). They are screened in the leakage audit and the
          final model is reported with and without them.

All fitting steps (imputation, encoding, scaling) live in the sklearn pipeline and
are fit on training rows only. Nothing here looks at the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TIER_A = ["Source_clean", "created_hour_bucket", "created_dow"]
TIER_B_GROUPS = {
    "Product": ["Product"],
    "Location": ["Location"],
    "Delivery_Mode": ["Delivery_Mode"],
    "Sales_Agent": ["Sales_Agent"],
    "email_mobile_flags": ["email_valid", "mobile_missing"],
    "field_missing_flags": ["product_missing", "source_missing", "agent_missing", "location_missing"],
    "contact_history": ["prior_contact_leads"],
}
CATEGORICAL = ["Source_clean", "created_hour_bucket", "created_dow", "Product", "Location", "Delivery_Mode", "Sales_Agent"]
NUMERIC = ["prior_contact_leads"]
BINARY = ["email_valid", "mobile_missing", "product_missing", "source_missing", "agent_missing", "location_missing"]
ALL_FEATURES = TIER_A + [c for cols in TIER_B_GROUPS.values() for c in cols]

# Modelling feature sets decided from the Phase 4 audit (reports/phase4_leakage_audit.md).
FEATURE_SETS = {
    # Source and creation time: believed available when the lead is created.
    "creation_time": list(TIER_A),
    # Adds Delivery_Mode and Location. The owner says agents were routed on these, so they exist
    # when the lead is assigned; whether the customer enters them is NOT confirmed.
    "assignment_time": TIER_A + ["Delivery_Mode", "Location"],
    # Adds Product and email_valid, treated as possibly rep-entered after contact until confirmed.
    "post_contact": TIER_A + ["Delivery_Mode", "Location", "Product", "email_valid"],
}
# Sensitivity only: the same sets plus Sales_Agent. Agent assignment was partly segmented by
# mode/location (Phase 4 check) and the outcome label is agent-entered, so these are never headline results.
HANDLER_AWARE_SETS = {f"{name}+agent": cols + ["Sales_Agent"] for name, cols in FEATURE_SETS.items()}
EXCLUDED_FEATURES = {
    "mobile_missing": "Pure time artifact: 100% missing through May, 0% from July.",
    "prior_contact_leads": "Time proxy (rank correlation with time ~0.38); identifiers are shared placeholders, so it counts identifier frequency, not contact history.",
    "product_missing / source_missing / agent_missing / location_missing": "No ranking signal (AUC ~0.5); each flag covers under 1% of rows.",
    "Sales_Agent": "Leads look routed by product/location/mode and assignment may follow engagement; keep only as a labelled sensitivity run.",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add creation-time and completeness features to a cleaned frame.

    prior_contact_leads counts EARLIER leads from the same hashed contact key
    (strictly before this lead in time order), so it never uses future rows.
    Contacts with no key get 0.
    """
    out = df.copy()
    hour = out["Created"].dt.hour
    out["created_hour_bucket"] = pd.cut(hour, [-1, 8, 12, 16, 20, 24], labels=["00-08", "09-12", "13-16", "17-20", "21-23"]).astype(object)
    out["created_dow"] = out["Created"].dt.day_name()
    out["product_missing"] = out["Product"].isna().astype(int)
    out["source_missing"] = out["Source"].isna().astype(int)
    out["agent_missing"] = out["Sales_Agent"].isna().astype(int)
    out["location_missing"] = out["Location"].isna().astype(int)
    order = out.sort_values(["Created", "lead_id"]).index
    ck = out.loc[order, "contact_key"]
    prior = ck.groupby(ck).cumcount().where(ck.notna(), 0)
    out["prior_contact_leads"] = prior.reindex(out.index).astype(float)
    return out


def make_preprocessor(columns: list[str], min_count: int = 30, dense: bool = False) -> ColumnTransformer:
    """ColumnTransformer for a subset of ALL_FEATURES. Unfitted; fit inside a Pipeline.

    dense=True returns a dense matrix (needed by tree models that reject sparse input).
    """
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c in NUMERIC]
    binary = [c for c in columns if c in BINARY]
    unknown = set(columns) - set(cat) - set(num) - set(binary)
    if unknown:
        raise ValueError(f"Unknown feature columns: {sorted(unknown)}")
    parts = []
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_count,
                                     sparse_output=not dense)),
        ]), cat))
    if num:
        parts.append(("num", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), num))
    if binary:
        parts.append(("bin", "passthrough", binary))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0 if dense else 0.3)

"""Tests use small synthetic frames only - no real customer data."""
import numpy as np
import pandas as pd
import pytest

from ficzon.cleaning import clean_leads
from ficzon.features import ALL_FEATURES, add_features, make_preprocessor
from ficzon.splits import temporal_masks

CFG = {
    "source_map": {"Live Chat -PPC": "Live Chat-PPC", "Existing Customer": "Existing Client"},
    "split": {"train_end": "2018-08-31 23:59:59", "val_end": "2018-09-30 23:59:59"},
}


def make_raw(n=12):
    created = pd.date_range("2018-08-28 09:00", periods=n, freq="1D")
    return pd.DataFrame({
        "lead_id": range(1, n + 1),
        "Created": created,
        "Product_ID": [18.0, np.nan] * (n // 2),
        "Source": [" Call ", "Live Chat -PPC", "Existing Customer", None] * (n // 4),
        "Sales_Agent": ["Sales-Agent-1", None] * (n // 2),
        "Location": ["Bangalore", " "] * (n // 2),
        "Delivery_Mode": ["Mode-1"] * n,
        "Status": ["CONVERTED"] * n,
        "contact_key": ["a", "a", "b", None] * (n // 4),
        "mobile_missing": [0, 1] * (n // 2),
        "email_valid": [1, 0] * (n // 2),
    })


def test_cleaning_keeps_every_row_and_fixes_labels():
    raw = make_raw()
    out = clean_leads(raw, CFG)
    assert len(out) == len(raw)
    assert set(out["Source_clean"].dropna()) == {"Call", "Live Chat-PPC", "Existing Client"}
    assert out["Location"].isna().sum() == 6          # blank strings become missing, not dropped
    assert out["Product"].dropna().unique().tolist() == ["P18"]


def test_prior_contact_count_uses_only_earlier_rows():
    out = add_features(clean_leads(make_raw(), CFG))
    a_rows = out[out["contact_key"] == "a"].sort_values("Created")
    assert a_rows["prior_contact_leads"].tolist() == list(range(len(a_rows)))
    assert (out.loc[out["contact_key"].isna(), "prior_contact_leads"] == 0).all()


def test_prior_contact_count_ignores_input_row_order():
    out1 = add_features(clean_leads(make_raw(), CFG)).set_index("lead_id")["prior_contact_leads"]
    shuffled = make_raw().sample(frac=1, random_state=0)
    out2 = add_features(clean_leads(shuffled, CFG)).set_index("lead_id")["prior_contact_leads"]
    pd.testing.assert_series_equal(out1.sort_index(), out2.sort_index())


def test_features_do_not_include_target_or_pii_columns():
    forbidden = {"Status", "target_high", "Mobile", "EMAIL", "contact_key", "Created"}
    assert forbidden.isdisjoint(ALL_FEATURES)


def test_split_is_disjoint_exhaustive_and_ordered():
    created = pd.Series(pd.date_range("2018-08-01", "2018-11-10", freq="6h"))
    m = temporal_masks(created, CFG)
    assert (m["train"].astype(int) + m["val"].astype(int) + m["test"].astype(int) == 1).all()
    assert created[m["train"]].max() < created[m["val"]].min()
    assert created[m["val"]].max() < created[m["test"]].min()


def test_split_rejects_bad_boundaries():
    bad = {"split": {"train_end": "2018-10-01", "val_end": "2018-09-01"}}
    with pytest.raises(ValueError):
        temporal_masks(pd.Series(pd.date_range("2018-08-01", periods=3)), bad)


def test_preprocessor_fits_on_train_only_and_handles_unseen_levels():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = add_features(clean_leads(make_raw(), CFG))
    y = np.array([0, 1] * 6)
    pipe = Pipeline([("prep", make_preprocessor(ALL_FEATURES, min_count=1)), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(df.iloc[:8][ALL_FEATURES], y[:8])
    new = df.iloc[8:][ALL_FEATURES].copy()
    new["Source_clean"] = "Brand New Source"            # unseen level must not crash
    assert pipe.predict_proba(new).shape == (4, 2)


def test_preprocessor_rejects_unknown_columns():
    with pytest.raises(ValueError):
        make_preprocessor(["Status"])


def test_modelling_feature_sets_exclude_flagged_features_and_pii():
    from ficzon.features import EXCLUDED_FEATURES, FEATURE_SETS

    banned = {"mobile_missing", "prior_contact_leads", "Sales_Agent", "product_missing", "source_missing",
              "agent_missing", "location_missing", "Status", "target_high", "contact_key", "Mobile", "EMAIL"}
    for name, cols in FEATURE_SETS.items():
        assert banned.isdisjoint(cols), f"{name} contains an excluded feature"
        make_preprocessor(cols)                       # every listed column must be buildable
    assert FEATURE_SETS["creation_time"] == FEATURE_SETS["post_contact"][:3]
    assert set(FEATURE_SETS["creation_time"]) < set(FEATURE_SETS["assignment_time"]) < set(FEATURE_SETS["post_contact"])
    assert EXCLUDED_FEATURES


def test_handler_aware_sets_only_add_sales_agent():
    from ficzon.features import FEATURE_SETS, HANDLER_AWARE_SETS

    assert set(HANDLER_AWARE_SETS) == {f"{k}+agent" for k in FEATURE_SETS}
    for name, cols in HANDLER_AWARE_SETS.items():
        base = FEATURE_SETS[name.removesuffix("+agent")]
        assert set(cols) - set(base) == {"Sales_Agent"}
        make_preprocessor(cols)


def test_assert_no_pii_blocks_columns_and_values():
    from ficzon.data import assert_no_pii

    assert_no_pii(pd.DataFrame({"lead_id": [1], "Source": ["Call"], "created_at": ["2018-05-01 09:00:00"]}))
    with pytest.raises(ValueError):
        assert_no_pii(pd.DataFrame({"EMAIL": ["x"]}))
    with pytest.raises(ValueError):
        assert_no_pii(pd.DataFrame({"note": ["reach me at a@b.com"]}))
    with pytest.raises(ValueError):
        assert_no_pii(pd.DataFrame({"note": ["call 919876543210"]}))


@pytest.mark.parametrize("sep,enc", [(";", "cp1252"), (",", "utf-8-sig"), (",", "utf-8"), ("\t", "utf-8")])
def test_load_raw_autodetects_delimiter_and_encoding(tmp_path, sep, enc):
    from ficzon.data import EXPECTED_COLUMNS, load_raw

    rows = [["28-04-2018 07:54", 25, "Website", "9", "a@b.com", "Sales-Agent-3", "Chennai", "Mode-1", "CONVERTED"]]
    p = tmp_path / "x.csv"
    p.write_text(sep.join(EXPECTED_COLUMNS) + "\n" + "\n".join(sep.join(map(str, r)) for r in rows) + "\n", encoding=enc)
    cfg = {"csv": {"sep": ";", "encoding": "cp1252"}, "paths": {"raw_csv": "x"}}
    assert list(load_raw(p, cfg).columns) == EXPECTED_COLUMNS


def test_load_raw_error_names_the_problem(tmp_path):
    from ficzon.data import load_raw

    p = tmp_path / "bad.csv"; p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Could not find the expected columns"):
        load_raw(p, {"csv": {"sep": ";", "encoding": "cp1252"}, "paths": {"raw_csv": "x"}})


def test_parse_created_handles_alternate_formats():
    from ficzon.data import parse_created

    a = parse_created(pd.Series(["28-04-2018 07:54"])); b = parse_created(pd.Series(["28/04/2018 07:54"]))
    assert a[0] == b[0] == pd.Timestamp("2018-04-28 07:54")

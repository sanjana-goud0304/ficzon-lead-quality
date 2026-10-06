"""Tests for the time-ordered CV splitter and parameterized estimators (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from ficzon.models import PARAM_GRIDS, build_estimator
from ficzon.splits import expanding_month_folds


def make_created():
    return pd.Series(pd.date_range("2018-04-28", "2018-09-30", freq="7h"))


def test_expanding_folds_never_train_on_validation_month_or_later():
    created = make_created()
    folds = expanding_month_folds(created, ["2018-06", "2018-07", "2018-09"])
    assert len(folds) == 3
    for (tr, va), month in zip(folds, ["2018-06", "2018-07", "2018-09"]):
        assert set(tr).isdisjoint(va)
        assert created.iloc[tr].max() < created.iloc[va].min()
        assert (created.iloc[va].dt.strftime("%Y-%m") == month).all()


def test_expanding_folds_grow_and_cover_whole_month():
    created = make_created()
    folds = expanding_month_folds(created, ["2018-06", "2018-07", "2018-08"])
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    full_july = (created.dt.strftime("%Y-%m") == "2018-07").sum()
    assert len(folds[1][1]) == full_july


def test_expanding_folds_reject_empty_months():
    with pytest.raises(ValueError):
        expanding_month_folds(make_created(), ["2019-01"])
    with pytest.raises(ValueError):
        expanding_month_folds(make_created(), ["2018-04"])      # nothing before the first lead


@pytest.mark.parametrize("model", list(PARAM_GRIDS))
def test_build_estimator_applies_params_without_mutating_defaults(model):
    cols = ["Source_clean", "Delivery_Mode"]
    params = {k: v[0] for k, v in PARAM_GRIDS[model].items()}
    pipe = build_estimator(model, params, cols)
    for k, v in params.items():
        assert pipe.get_params()[f"clf__{k}"] == v
    other = build_estimator(model, {k: v[-1] for k, v in PARAM_GRIDS[model].items()}, cols)
    assert other.get_params() != pipe.get_params()

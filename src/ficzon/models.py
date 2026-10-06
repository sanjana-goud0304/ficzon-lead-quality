"""Baselines and candidate classifiers, all as sklearn Pipelines.

Hyperparameters here are fixed, reasonable defaults. Tuning happens in Phase 6 with
time-ordered cross-validation on train + validation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


class SourceRateBaseline(ClassifierMixin, BaseEstimator):
    """Transparent rule: score a lead by the (smoothed) training High rate of its Source.

    The smoothing pulls small sources toward the overall rate, so a source with a handful
    of leads cannot produce an extreme score. Unseen or missing sources get the overall rate.
    """

    def __init__(self, column: str = "Source_clean", smoothing: float = 20.0):
        self.column = column
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        g = pd.DataFrame({"k": X[self.column].astype("object").fillna("Unknown").to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        self.rates_ = ((g["sum"] + self.smoothing * self.prior_) / (g["count"] + self.smoothing)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame):
        keys = X[self.column].astype("object").fillna("Unknown")
        p = keys.map(self.rates_).fillna(self.prior_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_baselines() -> dict[str, Pipeline]:
    return {
        "Majority class (prior)": Pipeline([("clf", DummyClassifier(strategy="prior"))]),
        "Source-rate rule": Pipeline([("clf", SourceRateBaseline())]),
    }


def build_candidates(columns: list[str], min_count: int = 30, seed: int = 42) -> dict[str, Pipeline]:
    """Candidate models over one feature set. The preprocessor is part of each pipeline."""
    def prep():
        return make_preprocessor(columns, min_count, dense=True)

    return {
        "Logistic regression": Pipeline([("prep", prep()),
                                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random forest": Pipeline([("prep", prep()),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
                                                                  n_jobs=-1, random_state=seed))]),
        "Gradient boosting": Pipeline([("prep", prep()),
                                       ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=3,
                                                                              min_samples_leaf=40, l2_regularization=1.0,
                                                                              random_state=seed))]),
    }


# Small, deliberately coarse grids: the data is one noisy 6-month window, so a fine search would only fit noise.
PARAM_GRIDS = {
    "Logistic regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "Random forest": {"min_samples_leaf": [10, 30], "max_depth": [4, 8]},
    "Gradient boosting": {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 250]},
}


def build_estimator(name: str, params: dict, columns: list[str], min_count: int = 30, seed: int = 42) -> Pipeline:
    """One candidate pipeline with `params` applied to its classifier step."""
    pipe = build_candidates(columns, min_count, seed)[name]
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe

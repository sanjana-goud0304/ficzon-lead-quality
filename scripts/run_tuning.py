"""Phase 6, step 1: time-ordered CV and tuning on TRAIN + VALIDATION months only.

The SQL query itself excludes test months, so no test row is ever loaded here. The
chosen models, hyperparameters and decision thresholds are written to
reports/phase6_frozen_config.json; scripts/run_final_eval.py then reads that file
and evaluates the test months exactly once.

Usage:  python scripts/run_tuning.py
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
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ficzon import eda  # noqa: E402
from ficzon.cleaning import clean_leads  # noqa: E402
from ficzon.data import REPO_ROOT, load_config  # noqa: E402
from ficzon.features import FEATURE_SETS, HANDLER_AWARE_SETS, add_features  # noqa: E402
from ficzon.models import PARAM_GRIDS, SourceRateBaseline, build_estimator  # noqa: E402
from ficzon.splits import expanding_month_folds  # noqa: E402

warnings.filterwarnings("ignore")
CV_MONTHS = ["2018-06", "2018-07", "2018-08", "2018-09"]
SIMPLICITY_ORDER = ["Logistic regression", "Gradient boosting", "Random forest"]
PARSIMONY_TOLERANCE = 0.01          # prefer the simplest model within this much mean CV PR-AUC of the best


def load_dev(cfg: dict) -> pd.DataFrame:
    q = "SELECT * FROM leads WHERE Created <= ?"
    with closing(sqlite3.connect(str(REPO_ROOT / cfg['paths']['sqlite_db']))) as con:
        df = pd.read_sql_query(q, con, params=[cfg["split"]["val_end"]], parse_dates=["Created"])
    assert df["Created"].max() <= pd.Timestamp(cfg["split"]["val_end"]), "test rows must not be loaded in tuning"
    return add_features(clean_leads(df, cfg)).sort_values(["Created", "lead_id"]).reset_index(drop=True)


def cv_scores(make_pipe, X: pd.DataFrame, y: np.ndarray, folds):
    """Fit on each expanding window, score the next month. Returns per-fold metrics and out-of-fold scores."""
    oof = np.full(len(y), np.nan)
    rows = []
    for (tr, va), month in zip(folds, CV_MONTHS):
        pipe = make_pipe().fit(X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[va])[:, 1]
        oof[va] = s
        rows.append({"month": month, "n_train": len(tr), "n_val": len(va), "base_rate": float(y[va].mean()),
                     "roc_auc": roc_auc_score(y[va], s), "pr_auc": average_precision_score(y[va], s)})
    return rows, oof


def f1_best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximizing F1 on pooled out-of-fold scores (fixed before the test months are touched)."""
    cands = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def main() -> None:
    cfg = load_config(); seed = cfg["random_seed"]; min_count = cfg["min_category_count"]
    df = load_dev(cfg)
    y = df["target_high"].to_numpy()
    folds = expanding_month_folds(df["Created"], CV_MONTHS)

    grid_rows, frozen = [], {"cv_months": CV_MONTHS, "parsimony_tolerance": PARSIMONY_TOLERANCE, "sets": {}}
    fold_detail = {}

    for fset, cols in FEATURE_SETS.items():
        X = df[cols]
        results = []
        for model, grid in PARAM_GRIDS.items():
            for params in ParameterGrid(grid):
                rows, _ = cv_scores(lambda: build_estimator(model, params, cols, min_count, seed), X, y, folds)
                r = pd.DataFrame(rows)
                results.append({"feature_set": fset, "model": model, "params": json.dumps(params, sort_keys=True),
                                "cv_pr_auc": r.pr_auc.mean(), "cv_roc_auc": r.roc_auc.mean(),
                                "pr_auc_sd": r.pr_auc.std(ddof=0), "worst_fold_pr_auc": r.pr_auc.min()})
        res = pd.DataFrame(results); grid_rows.append(res)

        best_per_model = res.loc[res.groupby("model").cv_pr_auc.idxmax()].set_index("model")
        top = best_per_model.cv_pr_auc.max()
        eligible = [m for m in SIMPLICITY_ORDER if best_per_model.loc[m, "cv_pr_auc"] >= top - PARSIMONY_TOLERANCE]
        chosen = eligible[0]
        params = json.loads(best_per_model.loc[chosen, "params"])
        rows, oof = cv_scores(lambda: build_estimator(chosen, params, cols, min_count, seed), X, y, folds)
        mask = ~np.isnan(oof)
        thr = f1_best_threshold(y[mask], oof[mask])
        fold_detail[fset] = pd.DataFrame(rows)
        frozen["sets"][fset] = {"columns": cols, "model": chosen, "params": params, "threshold": thr,
                                "cv_pr_auc": float(pd.DataFrame(rows).pr_auc.mean()), "cv_roc_auc": float(pd.DataFrame(rows).roc_auc.mean()),
                                "best_per_model": {m: float(v) for m, v in best_per_model.cv_pr_auc.items()},
                                "pooled_oof_flag_rate_at_threshold": float((oof[mask] >= thr).mean())}

    # Handler-aware sensitivity: reuse the base set's model and parameters (no separate tuning), own threshold.
    frozen["handler_aware"] = {}
    for hname, hcols in HANDLER_AWARE_SETS.items():
        base = frozen["sets"][hname.removesuffix("+agent")]
        rows, oof = cv_scores(lambda: build_estimator(base["model"], base["params"], hcols, min_count, seed), df[hcols], y, folds)
        mask = ~np.isnan(oof); r = pd.DataFrame(rows)
        frozen["handler_aware"][hname] = {"columns": hcols, "model": base["model"], "params": base["params"],
                                          "threshold": f1_best_threshold(y[mask], oof[mask]),
                                          "cv_pr_auc": float(r.pr_auc.mean()), "cv_roc_auc": float(r.roc_auc.mean())}
        fold_detail[hname] = r

    # Baselines under the same CV (for context)
    def src_rule():
        return SourceRateBaseline()
    rows, _ = cv_scores(src_rule, df[["Source_clean"]].rename(columns={}), y, folds)
    base_r = pd.DataFrame(rows)
    frozen["baselines_cv"] = {"random_pr_auc_mean_of_fold_base_rates": float(base_r.base_rate.mean()),
                              "source_rule_cv_pr_auc": float(base_r.pr_auc.mean()), "source_rule_cv_roc_auc": float(base_r.roc_auc.mean())}
    fold_detail["Source-rate rule"] = base_r

    (REPO_ROOT / "reports" / "phase6_frozen_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    grid = pd.concat(grid_rows); grid.to_csv(REPO_ROOT / "reports" / "phase6_cv_grid.csv", index=False)

    # ---- report ----
    md = ["# Phase 6a: time-ordered cross-validation and tuning", "",
          "_Generated by `scripts/run_tuning.py`. Train + validation months only (Apr-Sep 2018); test months were never loaded._", "",
          f"Expanding-window CV: train on all earlier months, validate on {', '.join(CV_MONTHS)} ({len(CV_MONTHS)} folds). "
          f"Selection metric: mean PR-AUC across folds. Parsimony rule: choose the simplest model class whose best mean PR-AUC is within "
          f"{PARSIMONY_TOLERANCE} of the overall best (order: logistic regression, gradient boosting, random forest). "
          "Thresholds maximize F1 on pooled out-of-fold scores and are frozen before the test months are touched.", ""]
    for fset, f in frozen["sets"].items():
        md += [f"## {fset}", "", f"Features: {', '.join(f['columns'])}", "",
               f"Chosen: **{f['model']}**, params {json.dumps(f['params'])}, mean CV PR-AUC {f['cv_pr_auc']:.3f}, "
               f"mean CV ROC-AUC {f['cv_roc_auc']:.3f}, threshold {f['threshold']:.3f} "
               f"(flags {f['pooled_oof_flag_rate_at_threshold']*100:.0f}% of out-of-fold leads)", "",
               "Best mean CV PR-AUC by model class: " + ", ".join(f"{m} {v:.3f}" for m, v in f["best_per_model"].items()), "",
               fold_detail[fset].round(3).to_markdown(index=False), ""]
    md += ["## Handler-aware sensitivity (adds Sales_Agent; same model and parameters as the base set)", "",
           pd.DataFrame([{"feature_set": k, "model": v["model"], "cv_pr_auc": round(v["cv_pr_auc"], 3), "cv_roc_auc": round(v["cv_roc_auc"], 3)}
                         for k, v in frozen["handler_aware"].items()]).to_markdown(index=False), "",
           "## Baselines under the same CV", "",
           f"- Random scorer PR-AUC (mean of fold base rates): {frozen['baselines_cv']['random_pr_auc_mean_of_fold_base_rates']:.3f}",
           f"- Source-rate rule: PR-AUC {frozen['baselines_cv']['source_rule_cv_pr_auc']:.3f}, ROC-AUC {frozen['baselines_cv']['source_rule_cv_roc_auc']:.3f}", "",
           "Full grid: `reports/phase6_cv_grid.csv`.", ""]
    (REPO_ROOT / "reports" / "phase6a_tuning.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figure: PR-AUC by CV fold ----
    eda.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8)); xs = np.arange(len(CV_MONTHS))
    for (name, col), off in zip([("creation_time", eda.GRAY), ("assignment_time", eda.ORANGE), ("post_contact", eda.BLUE)], [-0.12, 0, 0.12]):
        d = fold_detail[name]
        ax.plot(xs + off, d.pr_auc, marker="o", ms=6, lw=2, color=col, label=f"{name} ({frozen['sets'][name]['model']})")
    ax.plot(xs, fold_detail["Source-rate rule"].pr_auc, marker="s", ms=5, lw=1.5, color=eda.INK_2, ls=":", label="Source-rate rule")
    ax.plot(xs, fold_detail["post_contact"].base_rate, color=eda.INK_2, lw=1.2, ls="--", label="Random scorer (fold base rate)")
    ax.set_xticks(xs); ax.set_xticklabels([f"{m[5:]}/{m[2:4]}" for m in CV_MONTHS]); ax.set_xlabel("Validation month (trained on all earlier months)")
    ax.set_ylabel("PR-AUC (High Potential)"); ax.set_title("Cross-validated PR-AUC by month, selected models"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(REPO_ROOT / "reports/figures/fig16_cv_pr_auc_by_fold.png", bbox_inches="tight"); plt.close(fig)

    for fset, f in frozen["sets"].items():
        print(f"{fset:16s} {f['model']:20s} {json.dumps(f['params'])} cvPR={f['cv_pr_auc']:.3f} cvROC={f['cv_roc_auc']:.3f} thr={f['threshold']:.3f}")
    print({k: round(v["cv_pr_auc"], 3) for k, v in frozen["handler_aware"].items()}, frozen["baselines_cv"])


if __name__ == "__main__":
    main()

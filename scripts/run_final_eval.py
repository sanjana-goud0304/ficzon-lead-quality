"""Phase 6, step 2: single final evaluation on the TEST months (October 2018 onward).

Reads reports/phase6_frozen_config.json (written by run_tuning.py from train+validation
only), refits each frozen configuration on all train+validation rows, and scores the test
months once. Nothing here feeds back into model or threshold selection.

Usage:  python scripts/run_final_eval.py
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ficzon import eda  # noqa: E402
from ficzon.cleaning import clean_leads  # noqa: E402
from ficzon.data import REPO_ROOT, load_config  # noqa: E402
from ficzon.evaluate import evaluate_scores, threshold_metrics, topk_flags  # noqa: E402
from ficzon.features import add_features  # noqa: E402
from ficzon.models import SourceRateBaseline, build_estimator  # noqa: E402
from ficzon.splits import temporal_masks  # noqa: E402

warnings.filterwarnings("ignore")
TOP_FRAC = 0.2


def main() -> None:
    cfg = load_config(); seed = cfg["random_seed"]; min_count = cfg["min_category_count"]
    frozen = json.loads((REPO_ROOT / "reports" / "phase6_frozen_config.json").read_text(encoding="utf-8"))
    with closing(sqlite3.connect(str(REPO_ROOT / cfg['paths']['sqlite_db']))) as con:
        df = pd.read_sql_query("SELECT * FROM leads", con, parse_dates=["Created"])
    df = add_features(clean_leads(df, cfg)).sort_values(["Created", "lead_id"]).reset_index(drop=True)
    masks = temporal_masks(df["Created"], cfg)
    dev, test = df[masks["train"] | masks["val"]].copy(), df[masks["test"]].copy()
    assert dev["Created"].max() < test["Created"].min()
    ydev, ytest = dev["target_high"].to_numpy(), test["target_high"].to_numpy()
    high = set(cfg["target"]["high_potential"])

    # ---- fit once, score once ----
    specs = {**frozen["sets"], **frozen["handler_aware"]}
    scores, models = {}, {}
    for name, sp in specs.items():
        pipe = build_estimator(sp["model"], sp["params"], sp["columns"], min_count, seed).fit(dev[sp["columns"]], ydev)
        models[name], scores[name] = pipe, pipe.predict_proba(test[sp["columns"]])[:, 1]
    rule = SourceRateBaseline().fit(dev[["Source_clean"]], ydev)
    scores["Source-rate rule"] = rule.predict_proba(test[["Source_clean"]])[:, 1]
    scores["Majority class"] = np.full(len(test), ydev.mean())

    # ---- metrics ----
    rows = []
    for name, s in scores.items():
        m = evaluate_scores(ytest, s, TOP_FRAC, seed)
        thr = specs.get(name, {}).get("threshold")
        t = threshold_metrics(ytest, s, thr) if thr is not None else {k: np.nan for k in ["tn", "fp", "fn", "tp", "precision", "recall", "f1", "flag_rate"]}
        rows.append({"model": name, "algorithm": specs[name]["model"] if name in specs else "-", "threshold": thr, **m, **t})
    res = pd.DataFrame(rows)
    res.to_csv(REPO_ROOT / "reports" / "phase6_test_results.csv", index=False)

    primary = ["Majority class", "Source-rate rule", "creation_time", "assignment_time", "post_contact"]
    handler = [k for k in scores if k.endswith("+agent")]
    r = res.set_index("model")
    fmt = lambda x: f"{x:.3f}"

    def table1(names):
        out = pd.DataFrame({
            "model": names, "algorithm": [r.loc[n, "algorithm"] for n in names],
            "ROC-AUC (95% CI)": [f"{r.loc[n,'roc_auc']:.3f} ({r.loc[n,'roc_lo']:.3f}-{r.loc[n,'roc_hi']:.3f})" for n in names],
            "PR-AUC (95% CI)": [f"{r.loc[n,'pr_auc']:.3f} ({r.loc[n,'pr_lo']:.3f}-{r.loc[n,'pr_hi']:.3f})" for n in names]})
        return out

    def table_thr(names):
        return pd.DataFrame({"model": names, "threshold": [r.loc[n, "threshold"] for n in names], "flagged %": [r.loc[n, "flag_rate"] * 100 for n in names],
                             "precision": [r.loc[n, "precision"] for n in names], "recall": [r.loc[n, "recall"] for n in names], "F1": [r.loc[n, "f1"] for n in names],
                             "TP": [r.loc[n, "tp"] for n in names], "FP": [r.loc[n, "fp"] for n in names], "FN": [r.loc[n, "fn"] for n in names],
                             "TN": [r.loc[n, "tn"] for n in names]})

    def table_top(names):
        return pd.DataFrame({"model": names, "precision@top20%": [r.loc[n, "precision_at_k"] for n in names],
                             "recall@top20%": [r.loc[n, "recall_at_k"] for n in names], "lift@top20%": [r.loc[n, "lift_at_k"] for n in names]})

    # top-20% confusion matrices
    top_cm = {n: threshold_metrics(ytest, topk_flags(scores[n], TOP_FRAC, seed), 0.5) for n in ["creation_time", "assignment_time", "post_contact"]}

    # ---- sensitivity: alternative target definitions (frozen configs, labels remapped for fit and score) ----
    alts = {"Open mapped to High": lambda s: s.isin(high | {"Open"}),
            "Long Term mapped to Low": lambda s: s.isin(high - {"Long Term"}),
            "CONVERTED only": lambda s: s.isin({"CONVERTED"})}
    sens = []
    for alt, fn in alts.items():
        ya_dev, ya_test = fn(dev["Status"]).astype(int).to_numpy(), fn(test["Status"]).astype(int).to_numpy()
        for name in ["creation_time", "assignment_time", "post_contact"]:
            sp = specs[name]
            p = build_estimator(sp["model"], sp["params"], sp["columns"], min_count, seed).fit(dev[sp["columns"]], ya_dev)
            s = p.predict_proba(test[sp["columns"]])[:, 1]
            from ficzon.evaluate import topk_metrics
            tk = topk_metrics(ya_test, s, TOP_FRAC, seed)
            sens.append({"target definition": alt, "feature_set": name, "test base rate %": ya_test.mean() * 100,
                         "ROC-AUC": roc_auc_score(ya_test, s), "PR-AUC": average_precision_score(ya_test, s), "lift@top20%": tk["lift_at_k"]})
    sens = pd.DataFrame(sens)

    # ---- per test month ----
    test_month = test["Created"].dt.strftime("%Y-%m").to_numpy()
    pm = []
    for mth in sorted(set(test_month)):
        sel = test_month == mth
        for name in ["creation_time", "assignment_time", "post_contact", "Source-rate rule"]:
            pm.append({"month": mth, "n": int(sel.sum()), "base_rate %": ytest[sel].mean() * 100, "model": name,
                       "ROC-AUC": roc_auc_score(ytest[sel], scores[name][sel]), "PR-AUC": average_precision_score(ytest[sel], scores[name][sel])})
    pm = pd.DataFrame(pm)

    # ---- permutation importance (interpretation only; scored on test, nothing is selected from it) ----
    imps = {}
    for name in ["creation_time", "assignment_time", "post_contact"]:
        cols = specs[name]["columns"]
        pi = permutation_importance(models[name], test[cols], ytest, scoring="average_precision", n_repeats=15, random_state=seed)
        imps[name] = pd.DataFrame({"feature": cols, "pr_auc_drop": pi.importances_mean, "sd": pi.importances_std}).sort_values("pr_auc_drop", ascending=False)

    # ---- report ----
    flag_oof = {k: frozen["sets"][k]["pooled_oof_flag_rate_at_threshold"] * 100 for k in frozen["sets"]}
    md = ["# Phase 6b: final test evaluation", "",
          "_Generated by `scripts/run_final_eval.py`. Models and thresholds were frozen from train+validation (Apr-Sep 2018) in "
          "`reports/phase6_frozen_config.json` before the test months were loaded. Each model was refit once on all train+validation "
          "rows and scored once on the test months._", "",
          f"- Train+validation: {len(dev):,} leads ({dev.Created.min():%Y-%m-%d} to {dev.Created.max():%Y-%m-%d}), High rate {ydev.mean()*100:.1f}%",
          f"- Test: {len(test):,} leads ({test.Created.min():%Y-%m-%d} to {test.Created.max():%Y-%m-%d}), High rate {ytest.mean()*100:.1f}% "
          f"(a random scorer's PR-AUC and precision both equal this)",
          f"- Share of test leads still 'Open' (mapped to Low): {(test.Status=='Open').mean()*100:.1f}%", "",
          "## 1. Ranking quality (threshold-free)", "", table1(primary).to_markdown(index=False), "",
          "Intervals are 95% bootstrap over test rows. The three feature sets are a ladder by when each field is known; "
          "`assignment_time` and `post_contact` rest on timing assumptions not yet confirmed by the project owner.", "",
          "## 2. Classification at the pre-specified F1-optimal threshold", "",
          table_thr(["creation_time", "assignment_time", "post_contact"]).round(3).to_markdown(index=False), "",
          "The threshold maximized F1 on out-of-fold scores during tuning. For a weak ranker that rule drifts toward flagging nearly "
          f"everything: out-of-fold flag rates were " + ", ".join(f"{k} {v:.0f}%" for k, v in flag_oof.items()) + ". "
          "Where the flagged share is high, precision stays close to the base rate and the table says little about usefulness; use section 3 instead.", "",
          "## 3. Working only the top 20% of leads by score", "", table_top(["Source-rate rule", "creation_time", "assignment_time", "post_contact"]).round(3).to_markdown(index=False), "",
          "Confusion matrices when exactly the top 20% by score are flagged High:", "",
          pd.DataFrame([{"model": n, **{k: int(v) for k, v in m.items() if k in ["tp", "fp", "fn", "tn"]}, "precision": round(m["precision"], 3), "recall": round(m["recall"], 3)}
                        for n, m in top_cm.items()]).to_markdown(index=False), "",
          "## 4. Handler-aware sensitivity (adds Sales_Agent)", "", table1(handler).to_markdown(index=False), "",
          "Agent routing was partly segmented by mode and location and the Status label is agent-entered, so this is a sensitivity run only.", "",
          "## 5. Sensitivity to the target definition", "", sens.round(3).to_markdown(index=False), "",
          "Same frozen model settings, with labels remapped for both fitting and scoring. No re-tuning.", "",
          "## 6. Stability across the two test months", "", pm.round(3).to_markdown(index=False), "",
          "November is a partial month (to 14 Nov) with many unresolved leads.", "",
          "## 7. Permutation importance on test (drop in PR-AUC when a column is shuffled)", ""]
    for name, d in imps.items():
        md += [f"**{name}**", "", d.round(4).to_markdown(index=False), ""]

    # dynamic reading
    cs, asg, pc, sr = r.loc["creation_time"], r.loc["assignment_time"], r.loc["post_contact"], r.loc["Source-rate rule"]
    pc_vs_asg = "overlap" if max(asg.roc_lo, pc.roc_lo) <= min(asg.roc_hi, pc.roc_hi) else "do not overlap"
    cs_vs_rule = "overlap" if cs.roc_lo <= sr.roc_hi and sr.roc_lo <= cs.roc_hi else "do not overlap"
    md += ["## Reading the results", "",
           f"- creation_time ROC-AUC {cs.roc_auc:.3f} vs Source-rate rule {sr.roc_auc:.3f}; intervals {cs_vs_rule}.",
           f"- assignment_time ROC-AUC {asg.roc_auc:.3f} vs post_contact {pc.roc_auc:.3f}; intervals {pc_vs_asg}.",
           f"- Top-20% lift: creation_time {cs.lift_at_k:.2f}, assignment_time {asg.lift_at_k:.2f}, post_contact {pc.lift_at_k:.2f}, Source-rate rule {sr.lift_at_k:.2f}.",
           f"- Test base rate {ytest.mean()*100:.1f}% vs {ydev.mean()*100:.1f}% in train+validation: the label mix shifted, which moves threshold-based metrics.", ""]
    (REPO_ROOT / "reports" / "phase6b_final_evaluation.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures ----
    eda.apply_style()
    colors = {"creation_time": eda.GRAY, "assignment_time": eda.ORANGE, "post_contact": eda.BLUE, "Source-rate rule": eda.INK_2}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, col in colors.items():
        fpr, tpr, _ = roc_curve(ytest, scores[name]); axes[0].plot(fpr, tpr, color=col, lw=2, ls=":" if name == "Source-rate rule" else "-",
                                                                   label=f"{name} (AUC {r.loc[name,'roc_auc']:.2f})")
        pr, rc, _ = precision_recall_curve(ytest, scores[name]); axes[1].plot(rc, pr, color=col, lw=2, ls=":" if name == "Source-rate rule" else "-",
                                                                              label=f"{name} (AP {r.loc[name,'pr_auc']:.2f})")
    axes[0].plot([0, 1], [0, 1], color=eda.GRAY, ls="--", lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC curve, test months")
    axes[1].axhline(ytest.mean(), color=eda.GRAY, ls="--", lw=1, label=f"Random ({ytest.mean():.2f})"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve, test months")
    for a in axes: a.legend(frameon=False, fontsize=8.5, loc="lower right" if a is axes[0] else "upper right")
    fig.savefig(REPO_ROOT / "reports/figures/fig17_roc_pr_test.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for j, name in enumerate(["creation_time", "assignment_time", "post_contact"]):
        for i, (label, m) in enumerate([(f"F1 threshold {specs[name]['threshold']:.2f}", threshold_metrics(ytest, scores[name], specs[name]["threshold"])),
                                        ("Top 20% flagged", top_cm[name])]):
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]); pct = cm / cm.sum(axis=1, keepdims=True) * 100
            ax = axes[i, j]; ax.imshow(pct, cmap=eda.SEQ, vmin=0, vmax=100); ax.grid(False)
            for a_ in range(2):
                for b_ in range(2):
                    ax.text(b_, a_, f"{cm[a_, b_]:,}\n({pct[a_, b_]:.0f}%)", ha="center", va="center", fontsize=9, color="white" if pct[a_, b_] > 55 else eda.INK)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Low", "High"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name}\n{label}", fontsize=10)
    fig.suptitle("Confusion matrices on test months (cell % = share of each actual class)", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(REPO_ROOT / "reports/figures/fig18_confusion_matrices_test.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, col in colors.items():
        order = np.random.default_rng(seed).permutation(len(ytest)); o = order[np.argsort(-scores[name][order], kind="stable")]
        gain = np.cumsum(ytest[o]) / ytest.sum(); ax.plot(np.arange(1, len(o) + 1) / len(o) * 100, gain * 100, color=col, lw=2, ls=":" if name == "Source-rate rule" else "-", label=name)
    ax.plot([0, 100], [0, 100], color=eda.GRAY, ls="--", lw=1, label="Random order"); ax.axvline(20, color=eda.GRAY, lw=0.8)
    ax.set_xlabel("Share of leads worked, highest score first (%)"); ax.set_ylabel("Share of High Potential leads reached (%)")
    ax.set_title("Cumulative gains on test months"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig(REPO_ROOT / "reports/figures/fig19_cumulative_gains_test.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=False)
    for ax, (name, d) in zip(axes, imps.items()):
        d = d.sort_values("pr_auc_drop"); ax.barh(d.feature, d.pr_auc_drop, xerr=d.sd, color=colors[name], height=0.6, ecolor=eda.INK_2, zorder=2)
        ax.axvline(0, color=eda.GRAY, lw=1); ax.set_title(name, fontsize=11); ax.grid(axis="y", visible=False); ax.set_xlabel("Drop in PR-AUC")
    fig.suptitle("Permutation importance on test months", fontsize=12, fontweight="bold", x=0.02, ha="left"); fig.tight_layout()
    fig.savefig(REPO_ROOT / "reports/figures/fig20_permutation_importance.png", bbox_inches="tight"); plt.close(fig)

    pd.set_option("display.width", 220)
    print(table1(primary + handler).to_string(index=False)); print(table_thr(["creation_time", "assignment_time", "post_contact"]).round(3).to_string(index=False))
    print(table_top(["Source-rate rule", "creation_time", "assignment_time", "post_contact"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

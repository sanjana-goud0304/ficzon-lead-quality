# Model card: lead-quality classifier (portfolio project)

**Task.** Predict High vs Low Potential for a lead. **Not for production use**: built from one 2018 export, evaluated on one window, no deployment pipeline.

**Data.** 7,422 leads, Apr-Nov 2018. Time-ordered split: train to 31 Aug (4,936), validation Sep (1,103), test 1 Oct to 14 Nov (1,383).

**Models** (scikit-learn Pipelines; all preprocessing fitted inside the pipeline on training rows only):
| Set | Features | Algorithm (chosen by expanding-window CV, PR-AUC, simplest within 0.01) |
|---|---|---|
| creation_time | Source, hour bucket, weekday | Logistic regression |
| assignment_time | + Delivery_Mode, Location | Random forest (depth 4, leaf 10) |
| post_contact | + Product, email_valid | Logistic regression (C=10) |

**Performance (held-out test, 1,383 leads, base rate 45.6%).** ROC-AUC 0.559 / 0.764 / 0.795 and top-20% lift 1.21 / 1.63 / 1.74 respectively; source-rate rule 0.545 / 1.18. Details and intervals: `reports/phase6b_final_evaluation.md`. No claim is made about revenue.

**Intended use.** Ranking leads for prioritization in a shadow-mode pilot (`docs/phase7_recommendations_and_pilot.md`).

**Limitations.** (1) When Location, Delivery_Mode, Product and email are captured is unconfirmed; if after contact, the two stronger models partly restate the outcome. (2) Labels mature with lead age, so test-month labels are less settled. (3) Intervals ignore clustering of leads sharing placeholder contacts, so they are too narrow. (4) Status is agent-entered. (5) Single test window.

**Fairness and privacy.** No PII features; Sales_Agent excluded from the main models. Review scores by location and product before any use; scores should prioritize effort, not deny service.

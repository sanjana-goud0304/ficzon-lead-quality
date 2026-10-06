# Phase 6: cross-validation, tuning and final evaluation (summary)

Detail and tables live in the generated reports; this page summarizes them. Every number here appears in one of:
`phase6a_tuning.md` (CV), `phase6b_final_evaluation.md` (test), `phase6c_agent_analysis.md` (agents), `phase6_test_results.csv`.

## Protocol
- **Split:** train + validation = leads created Apr-Sep 2018 (6,039); test = 1 Oct to 14 Nov 2018 (1,383). Test High rate is 45.6%, versus 36.7% before.
- **CV:** expanding window, validating on Jun, Jul, Aug, Sep (each trained on all earlier months). Selection metric: mean PR-AUC. Simplest model class within 0.01 of the best wins.
- **Freeze, then test:** models, hyperparameters and F1-optimal thresholds were written to `phase6_frozen_config.json` from train+validation only. The tuning script never loads test rows (its SQL excludes them). The test months were scored once, after refitting on all train+validation rows.

## Feature ladder (by when each field is believed known)
| Set | Columns | Timing status |
|---|---|---|
| `creation_time` | Source, creation hour bucket, weekday | Believed known at creation |
| `assignment_time` | + Delivery_Mode, Location | Owner says agents were routed on these; whether the customer enters them is unconfirmed |
| `post_contact` | + Product, email_valid | Treated as possibly rep-entered after contact |

## Results on the test months (1,383 leads, base rate 45.6%)
| Model | Algorithm | ROC-AUC (95% CI) | PR-AUC (95% CI) | Lift in top 20% |
|---|---|---|---|---|
| Majority class | - | 0.500 | 0.456 | - |
| Source-rate rule | - | 0.545 (0.516-0.572) | 0.510 (0.474-0.540) | 1.18 |
| creation_time | Logistic regression | 0.559 (0.526-0.590) | 0.542 (0.504-0.583) | 1.21 |
| assignment_time | Random forest | 0.764 (0.739-0.789) | 0.713 (0.673-0.747) | 1.63 |
| post_contact | Logistic regression | 0.795 (0.772-0.817) | 0.748 (0.707-0.781) | 1.74 |

Cross-validated (Jun-Sep) PR-AUC for comparison: creation_time 0.495, assignment_time 0.618, post_contact 0.661, source rule 0.468, random 0.375.

**Classification metrics** (precision, recall, F1, confusion matrices) are in `phase6b_final_evaluation.md` and `figures/fig18_confusion_matrices_test.png`, at two operating points:
- *F1-optimal threshold (frozen before test):* assignment_time flags 57% of leads with precision 0.634, recall 0.797, F1 0.706; post_contact flags 65% with precision 0.612, recall 0.873, F1 0.720. For creation_time this rule degenerates: it flags 99.5% of leads (precision 0.455, recall 0.994), which is just "call everyone" and says nothing about usefulness.
- *Top 20% by score:* precision 0.744 and recall 0.327 (assignment_time); precision 0.791 and recall 0.348 (post_contact); precision 0.552 and recall 0.243 (creation_time).

## What the results support
1. **Creation-time information alone is weak.** Its ROC-AUC interval overlaps the one-column source rule's, and its top-20% lift is 1.21 against 1.18.
2. **Location, Product and Delivery_Mode carry the signal.** On test, shuffling Location costs 0.177 PR-AUC in assignment_time and 0.106 in post_contact; Product costs 0.097 in post_contact. Delivery_Mode adds nothing once Location and Product are present.
3. **post_contact vs assignment_time:** 0.795 vs 0.764 ROC-AUC, intervals overlap. Not a clear difference.
4. **The results hold across both test months** (post_contact ROC-AUC 0.766 in Oct, 0.865 in the partial Nov) and across target definitions (ROC-AUC 0.77-0.78 for post_contact when Open is mapped to High or Long Term to Low).
5. **Narrower target, bigger lift:** when High means CONVERTED only (5.8% of test leads), the top 20% by score reaches lift 3.00 (assignment_time) and 2.87 (post_contact), at PR-AUC 0.236 and 0.252 against a 0.058 base rate.
6. **Adding Sales_Agent raises scores** (post_contact ROC-AUC 0.809, creation_time 0.616) but agents were partly routed by mode and location and the label is agent-entered, so these stay sensitivity results.
7. **Agents differ after adjusting for segment.** Within comparable Mode x Location x month strata, Agent 11 (+6.5 pp) is above, and Agents 5 (-9.4), 3 (-9.1) and 12 (-16.7) are below, their peers, and the result survives dropping Agent-9 (+19.2 pp, the largest outlier). Agents 7, 4, 2 and 6 flip in or out of significance depending on the peer pool, so I do not rank them.

## Limits to carry into the recommendations
- **Timing of Location, Delivery_Mode, Product and email_valid is unconfirmed.** If they are entered after a rep has worked the lead, the assignment_time and post_contact numbers partly restate the outcome and do not apply at lead creation.
- **Confidence intervals understate uncertainty.** They resample rows as if independent, but the contact identifiers are mostly shared placeholders (one value covers 11.5% of rows), so leads are clustered.
- **Label maturity.** Statuses change as leads age (CONVERTED falls from 16.0% of May leads to 3.7% of November leads while in-flight statuses rise; see `figures/fig22_status_maturity_by_month.png`). Test-month labels are less mature than training labels, so absolute rates, and the CONVERTED counts behind the top-20% lift, can move as those leads resolve.
- **One test window.** 1,383 leads, with November partial and 10.2% of its leads still Open. The label mix moved between train and test (36.7% to 45.6% High), so threshold-based numbers will not transfer unchanged.
- **Test scores are higher than CV scores** (post_contact ROC-AUC 0.795 vs 0.772). The final model trains on more months, and the test mix differs; this does not guarantee the same level in future months.
- **Status is agent-entered.** Predicting it predicts how agents record leads, not only customer behavior.
- **Hyperparameter grids are coarse by design.** Logistic regression's best C sat at the grid edge (10), but C=1 and C=10 tie to three decimals, so the choice does not matter.

## Reproduce
```
FICZON_HASH_SALT=<random> python scripts/build_sqlite.py
python scripts/run_tuning.py        # CV + tuning, train+validation only; writes the frozen config
python scripts/run_final_eval.py    # single test evaluation from the frozen config
python scripts/run_agent_analysis.py
python -m pytest
```

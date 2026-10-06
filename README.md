# Sales Effectiveness: Lead Quality Prediction

An end-to-end analysis for a case-study company, FicZon, that sells through several lead channels. Goals: (1) understand what drives sales effectiveness, and (2) build a model that pre-categorizes each lead as **High** or **Low Potential**, so effort goes to the right leads during the conversation rather than after it.

Stack: Python, pandas, NumPy, scikit-learn, Matplotlib/Seaborn, SQL (SQLite), Power BI (specification).

> **Data and confidentiality.** The case brief and dataset are proprietary training material and are **not included**. Place your own copy at `data/raw/sales_project.csv` to reproduce. Mobile and EMAIL are dropped at load and never exported. No credentials are stored; use `.env.example`.

## Headline results (held-out later months, 1,383 leads, 45.6% High)
| Model | Features | ROC-AUC | Top-20% lift |
|---|---|---|---|
| Source-rate rule | Source only | 0.545 | 1.18 |
| creation_time | Source + time of day/week | 0.559 | 1.21 |
| assignment_time | + Delivery_Mode, Location | 0.764 | 1.63 |
| post_contact | + Product, email validity | 0.795 | 1.74 |

- **Channel alone says little; segment (Location, Product, Mode) carries the signal.** Creation-time features barely beat a one-column source lookup.
- **Open question that decides how far the strong models apply:** are Location, Delivery_Mode, Product and email recorded *before* a rep works the lead? Unconfirmed, so those scores may partly restate the outcome.
- **Labels mature with lead age**: CONVERTED falls from 16.0% (May leads) to 3.7% (Nov) while in-flight statuses rise, so part of the rising High rate is unresolved leads.
- No revenue or cost data exists, so **no monetary impact is claimed**.

## Method
Time-ordered split (train to Aug, validate Sep, test Oct-Nov); expanding-window CV; preprocessing inside Pipelines; thresholds and hyperparameters frozen to JSON before the single test evaluation; leakage audit and feature-timing ladder; leave-agent-out segment adjustment for agent comparison; bootstrap intervals. Read `reports/phase6_evaluation.md` first.

## Where things are
| Path | Contents |
|---|---|
| `src/ficzon/` | reusable modules (data, cleaning, features, splits, models, evaluate, agents, scoring, kpis) |
| `scripts/` | one script per phase; `run_all.sh` runs everything |
| `sql/` | audit and business queries (SQLite), MySQL extract template (untested) |
| `reports/`, `reports/figures/` | generated results and 22 figures |
| `docs/` | feature decisions, recommendations and pilot design, Power BI spec, model card, data dictionary, resume/interview notes |
| `powerbi/` | DAX measures and expected KPI values (row-level exports are gitignored) |
| `tests/` | pytest suite on synthetic data |

## Run
```
pip install -r requirements.txt
export FICZON_HASH_SALT=$(python -c "import secrets;print(secrets.token_hex(16))")
bash scripts/run_all.sh
```

## Limitations
Timing of key fields unconfirmed; label maturity; one test window; confidence intervals ignore clustering of leads that share placeholder contacts (too narrow); Status is agent-entered; the Power BI DAX has not been executed; the MySQL template was never run; CI config has not run on GitHub. See `docs/model_card.md`.

## Recommendations
Confirm field capture times, run a shadow-mode then randomized prioritization pilot, investigate the low-yield segment (Product 15, "Other Locations"), and log status-change timestamps: `docs/phase7_recommendations_and_pilot.md`.

License: not yet chosen (add one before publishing).

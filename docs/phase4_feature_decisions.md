# Phase 4: cleaning and feature decisions

Source of truth in code: `src/ficzon/features.py` (`FEATURE_SETS`, `EXCLUDED_FEATURES`) and `configs/config.yaml`. Evidence: `reports/phase4_leakage_audit.md`.

## Cleaning (no row is dropped)
- Whitespace trimmed, blank strings become missing, Source labels consolidated via `source_map` in the config (25 -> 22 labels; these merges are assumptions pending owner confirmation).
- Missing categories become an explicit `Unknown` level **inside the pipeline**, never before the split.
- Rare levels (under 30 training rows) are pooled by the one-hot encoder.

## Split
Time-ordered: train up to 31 Aug 2018, validation = September, test = October onward (reserved for the final evaluation in Phase 6; not used in Phases 4-5 for any decision).

## Feature sets
| Set | Columns | Status |
|---|---|---|
| `creation_time` | Source (cleaned), creation-hour bucket, creation weekday | Believed available at lead creation |
| `assignment_time` | `creation_time` + Delivery_Mode, Location | Added in Phase 6: owner says agents were routed on these; whether the customer enters them is unconfirmed |
| `post_contact` | `assignment_time` + Product, email_valid | Large lift, but timing unconfirmed; always reported with that caveat |
| `*+agent` (sensitivity) | any set above + Sales_Agent | Never headline: routing was partly segmented and Status is agent-entered |

## Excluded, with reason
- `mobile_missing`: time artifact (100% missing through May, 0% from July).
- `prior_contact_leads`: time proxy; the contact identifiers are mostly shared placeholders.
- Four field-missing flags: no ranking signal and under 1% of rows each.
- `Sales_Agent`: leads look routed by product, location and mode; kept only as a labelled sensitivity run.
- Raw Mobile and EMAIL, and the contact hash: PII or non-reproducible, never model inputs.

## Open question that decides the headline result
Product, Location and Delivery_Mode carry most of the signal, overlap heavily, and concentrate in leads that never engage. If they are recorded at creation they are legitimate predictors; if they are filled in after a rep contacts the lead, the extended model partly restates the outcome. The project owner needs to confirm which.

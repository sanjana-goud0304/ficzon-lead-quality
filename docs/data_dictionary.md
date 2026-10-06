# Data dictionary
Source: a 9-column lead export (semicolon-delimited, cp1252) with 7,422 rows created 2018-04-28 to 2018-11-14. The raw file is proprietary and not in this repo.

| Column | Meaning | Handling |
|---|---|---|
| Created | lead creation time (dd-mm-YYYY HH:MM) | parsed; drives the time-ordered split; hour bucket and weekday features |
| Product_ID | product code | stored as category "P<id>" |
| Source | acquisition channel | trimmed; five label variants merged via `source_map` |
| Mobile, EMAIL | contact details (PII) | **dropped** at load; only a salted hash (grouping/audit checks) and `email_valid` flag are derived; never exported |
| Sales_Agent | assigned agent | not in main models (routing and label confound); handler-aware sensitivity only |
| Location | delivery location; "Other Locations" is a catch-all (about a third of rows) | missing kept as its own level |
| Delivery_Mode | delivery mode | category |
| Status | lead outcome, entered by the agent | the basis of the target |

**Target (`Lead Category`)**: High Potential = CONVERTED, In Progress Positive, Potential, Long Term (38.4% of leads); Low Potential = Junk Lead, Not Responding, Just Enquiry, LOST, In Progress Negative, Open. This mapping is a working assumption pending the project owner's sign-off; it lives in `configs/config.yaml`.

**Derived**: lead_id, Source_clean, Product, created_hour_bucket, created_dow, email_valid, split (train to 2018-08-31, validation Sep, test 2018-10-01 onward). Excluded features and reasons: `docs/phase4_feature_decisions.md`.

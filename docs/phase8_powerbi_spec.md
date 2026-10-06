# Phase 8: Power BI specification

**Status: specification and data only.** The authoring environment has no Power BI Desktop, so **the DAX in `powerbi/measures.dax` has not been executed and no `.pbix` is shipped.** Correctness is checked by comparing each dashboard value to `powerbi/expected_kpis.csv`, which is computed by pandas (`src/ficzon/kpis.py`) from the same data.

## 1. Data to load
Run `python scripts/export_powerbi.py` locally. It writes `powerbi/data/*.csv` (gitignored, row-level and derived from customer records, so share privately) and the committable aggregate file `powerbi/expected_kpis.csv`. Every export passes `assert_no_pii`: no Mobile, EMAIL or contact key; the script was run successfully and produced 7,422 lead rows, 6,047 scored rows, 10 statuses and 12 agent-gap rows.

| Table | Grain | Key columns |
|---|---|---|
| `fact_leads` | one row per lead | lead_id, created_at, created_date, month_start, source_raw, source_group, product, location, delivery_mode, sales_agent, status, split |
| `dim_status` | one row per status | status, lead_category, is_converted, is_no_engagement, is_unresolved, sort_order (generated from `configs/config.yaml`) |
| `fact_scores` | one row per **out-of-time** scored lead | lead_id, split, score_source (cv_oof for Jun-Sep, final_model for Oct-Nov), three score columns |
| `agent_gaps` | one row per agent (static) | gap_pp with 95% interval, comparable share; computed in Python |
| `dim_date` | calendar (create in DAX) | `CALENDAR(DATE(2018,4,1), DATE(2018,11,30))` |
| `ScoreModel`, `Top Share` | disconnected selectors | model names; `GENERATESERIES(0.05, 0.5, 0.05)` |

Apr-May leads have no score on purpose (insufficient history for out-of-time scoring); in-sample scores are never shown.

## 2. Relationships
`fact_leads[status]` → `dim_status[status]` (many-to-one, single). `dim_date[Date]` → `fact_leads[created_date]` (convert to Date type; many-to-one). `fact_scores[lead_id]` ↔ `fact_leads[lead_id]` (one-to-one, both directions, so model KPIs respond to Location/Product slicers). Mark `dim_date` as the date table. `agent_gaps` is unrelated (static table).

## 3. KPI definitions (the contract)
| KPI | Definition | Notes |
|---|---|---|
| High Potential Rate | leads with status CONVERTED, In Progress Positive, Potential or Long Term / total leads | Working label, not "converted". 38.4% overall |
| Conversion Rate | CONVERTED / total | 11.5% overall; falls with lead age (see maturity) |
| No Engagement Rate | Junk Lead + Not Responding / total | 35.9% |
| Unresolved Share | Open / total | 1.1% |
| Wilson 95% interval | on High Potential Rate | shown on every rate card; flag n < 30 |
| Precision/Recall/Lift at top share | rank by score desc, lead_id asc; k = round-half-up(share × scored leads) | out-of-time rows only |

Status mapping is edited in `configs/config.yaml`, never by hand in Power BI, so Python, SQL and the dashboard stay aligned.

## 4. Pages
1. **Executive overview.** Cards: Total Leads, High Potential Rate (with Wilson bounds), Conversion Rate, No Engagement Rate. Monthly line: High Potential Rate and Conversion Rate together, with a text note that recent months are less mature. Slicers: month, source_group, product, location, delivery_mode.
2. **Lead funnel and quality.** Stacked bar of status mix by month (shows the label-maturity shift). Bar charts of High Potential Rate by source_group, product, location, delivery_mode, each with the Small Sample Flag and a minimum-n filter of 30.
3. **Agent view (with a caveat banner).** Table from `agent_gaps`: adjusted gap with interval, comparable share. Banner text: "Adjusted for mode, location and month. Outcome labels are agent-entered and routing was not random. Not a performance ranking."
4. **Model performance.** ScoreModel slicer, Top Share slicer, score_source slicer. Cards: Scored Leads, Top K, Precision/Recall/Lift At Top, Scored Base Rate. Quintile chart of High and CONVERTED rate by score band.
5. **Data quality.** Missing share by field, "Other Locations" share, placeholder-contact count, unresolved share.

## 5. Validation (to do in Power BI Desktop; not done here)
1. Load the four CSVs and create relationships above.
2. Add `measures.dax` measures; place each in a card with no filters. Compare to `expected_kpis.csv` rows where scope = overall (7,422 leads; 2,849 High; 852 CONVERTED).
3. Filter score_source = final_model, Top Share = 0.20 and compare for each model with scope `model:test_months_final_model` (expected: 1,383 scored; top 277; High in top 206 / 152 / 219 for assignment_time / creation_time / post_contact).
4. Spot-check three monthly and three source slices against `scope = month` / `source_group`.
Record the outcome in a change note; any difference above rounding is a bug in the DAX or relationship.

## 6. Governance
- **PII:** none loaded. Do not add Mobile, EMAIL or free-text columns.
- **Row-level security:** if agents will view the report, restrict `agent_gaps` and the agent slicer to managers (RLS on the sales_agent role); do not publish agent comparisons broadly.
- **Refresh:** monthly after statuses mature; re-run the export script and retrain according to `docs/model_card.md`. Scores in the report are historical and out-of-time; live scoring is a separate engineering step not built here.
- **Honest labels:** title the High Potential Rate "working label" and keep the maturity note on every trend visual.

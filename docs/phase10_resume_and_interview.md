# Phase 10: resume bullets and interview preparation

All numbers come from `reports/`. Say "case-study project" rather than implying a production deployment.

## Resume bullets (pick 3-4)
- Built a leakage-aware lead-scoring pipeline (pandas, scikit-learn) on 7,422 sales leads; with a time-ordered train/validation/test split and frozen hyperparameters, a location/mode model reached ROC-AUC 0.764 and a fuller model 0.795 on later held-out leads, versus 0.545 for a source-lookup baseline.
- Audited target leakage and field timing, and showed that creation-time features alone (ROC-AUC 0.559) barely beat a one-column baseline, redirecting the recommendation from "deploy the model" to a shadow-mode pilot.
- Discovered label right-censoring (CONVERTED share fell from 16.0% to 3.7% across lead-creation months while in-flight statuses rose), correcting an earlier mix-shift explanation and adding a maturity caveat to every report.
- Compared sales agents with leave-one-out standardization within mode × location × month strata and bootstrap intervals, and kept the result as a label-audit prompt rather than a performance ranking.
- Wrote 31 SQL audit and business queries, a tested Python KPI module, a PII-free Power BI export with DAX measures, and a 48-test pytest suite; designed a randomized pilot with sample-size planning (a +30% relative CONVERTED lift needs about 3,236 leads per arm).

## 60-second pitch
"FicZon wanted leads pre-categorized so reps focus during the conversation. I defined High Potential from the status field, split data by time, and tested three feature sets by when the information is known. Source and time alone barely beat a lookup table; location, mode and product carry the signal, giving ROC-AUC 0.76-0.80 on later leads. I could not confirm when those fields are captured, and statuses mature over time, so I recommended a shadow pilot before any rollout rather than claiming revenue impact."

## Likely questions
1. **Why a time-ordered split?** Leads are sequential and routing/labels drift; a random split would leak future patterns.
2. **How did you handle leakage?** Status excluded; Sales_Agent, mobile_missing and prior-contact count excluded with documented reasons; preprocessing inside pipelines; test touched once after freezing config.
3. **Why is the F1 threshold bad for creation_time?** It flags 99.5% of leads (call everyone); I reported top-20% precision/lift instead.
4. **What is your biggest caveat?** Field capture timing and label maturity, both of which could inflate or distort results.
5. **Why not claim business impact?** No revenue, cost or rep-time data exists.
6. **Agent differences?** Adjusted for segment, but labels are agent-entered and routing was not random; treat as a prompt for audit.
7. **What would you do next?** Get capture timestamps and status-change history, rerun at fixed label age, pilot as designed.
8. **What did you get wrong at first?** I attributed the rising High rate to mix shift; checking status mix by month showed label maturity.

## Not yet done (be upfront)
DAX not executed in Power BI; MySQL template untested; CI not run on GitHub; no live deployment.

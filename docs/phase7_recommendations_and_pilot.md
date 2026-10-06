# Phase 7: recommendations and proposed pilot

**Scope of the claims.** Every figure below comes from a report in `reports/` (named in each row). The dataset has **no revenue, deal value, cost or rep-hours**, so this document makes **no monetary impact claim**. Where a benefit is measured it is a retrospective ranking result on historical leads, not a forecast.

## 1. What the project found, in one page
| Question from the brief | Answer supported by the data | Source |
|---|---|---|
| What drives sales effectiveness? | Lead *segment* separates outcomes far more than lead channel: Product, Location and Delivery_Mode are the strongest signals; the three biggest sources sit within a few points of each other (35.6-39.0% High). | `phase3_eda.md`, `phase6b_final_evaluation.md` |
| Can ML pre-categorize High vs Low Potential? | Yes, but only from fields beyond source and time. Creation-time features alone (ROC-AUC 0.559) do not clearly beat a one-column source lookup (0.545). With Location and Delivery_Mode (`assignment_time`) ROC-AUC is 0.764; adding Product and email validity (`post_contact`) 0.795, on 1,383 held-out later leads. | `phase6b_final_evaluation.md` |
| Is the model usable "in the conversation" rather than only for post analysis? | Depends on one unconfirmed fact: **when Location, Delivery_Mode, Product and email are recorded.** | `phase4_leakage_audit.md` |

## 2. Recommendations
Confidence reflects the strength of the evidence, not the size of the benefit.

| # | Recommendation | Evidence | Confidence | What would change it |
|---|---|---|---|---|
| 1 | **Confirm and document when each field is captured** (website order form vs rep entry). Make Location, Delivery_Mode and Product capture mandatory at the order form if they are not already. | Those fields carry almost all model signal (shuffling Location costs 0.177 PR-AUC in `assignment_time`); creation-time-only features add little. | High that it matters; unknown which fields are early | The owner's answer. If they are entered after contact, only the creation-time model applies at first touch and it is weak. |
| 2 | **Run a score-assisted prioritization pilot** (section 4) instead of rolling the model out. | On held-out Oct-Nov leads, working the top 40% by `post_contact` score reaches 62.5% of High and 86.2% of CONVERTED leads; the bottom 40% held 16.5% of High and 1.2% of CONVERTED (1 of 80). Top-20% lift 1.63 (`assignment_time`) and 1.74 (`post_contact`). | Medium | Only 80 CONVERTED leads in the test window, immature labels, one window, clustered leads. |
| 3 | **Investigate the low-yield segment before automating anything.** Product 15, "Other Locations" and Mode-5 hold 20.5%, 33.7% and 40.1% of leads and 42.4%, 54.9% and 55.3% of all no-engagement leads (Junk Lead or Not Responding). High rates: 5.3%, 17.2%, 24.6%. | `phase3b_sql_business_analysis.md` | High that the pattern exists; unknown why | What these segments are (non-target geography, support requests, spam, a form default). A rule "Product 15 or Other Locations" flagged 37.5% of test leads at a 22.2% High rate versus 59.6% for the rest, and held 18.3% of High and 6.2% of CONVERTED leads. |
| 4 | **Replace the catch-all "Other Locations" with real values** at capture. | It is a third of all leads and hides what may be the largest quality difference. | High (data quality) | n/a |
| 5 | **Protect referral and existing-client channels; audit US Website.** Customer Referral 89.4% High (n=180), Existing Client 94.1% (n=51), By Recommendation 87.5% (n=32). US Website 1.5% High (2 of 137). | `phase3_eda.md` (section 3) | Medium for referrals (small n, and referred/existing customers are high-intent by nature, so this is not evidence a referral program would create more); Medium-high for US Website | Traffic quality review for US Website; a test, not an assumption, for referral programs. |
| 6 | **Review lead routing and, separately, audit the Status labels by agent.** Within comparable mode, location and month segments, Agent 11 is +6.5 points above peers; Agents 5, 3 and 12 are 9 to 17 points below, and these hold with the top outlier (Agent 9, +19.2) removed. Routing changed over time (segmented Jun-Aug, closer to random in Sep). | `phase6c_agent_analysis.md` | Medium: the outcome label is agent-entered | A label audit (sampled outcomes verified independently) before any coaching or staffing decision. Not a performance ranking. |
| 7 | **Fix label and identifier quality.** (a) Record **status-change timestamps** so outcomes can be measured at a fixed age. (b) Resolve ambiguous statuses (Open, Long Term, In Progress Negative). (c) Replace placeholder contact identifiers: one value covers 11.5% of rows and 34.8% of emails are not valid. | Section 3 below; `phase2_audit_results.md` | High | n/a |

## 3. The label-maturity problem (why recommendation 7(a) comes first)
Statuses mature with lead age: CONVERTED falls from 16.0% of May leads to 3.7% of November leads, while Potential + In Progress Positive rises from 6.5% to 42.3% (`phase3_eda.md`, addendum). The working High Potential label counts those in-flight states as High, so **the rising High rate is partly unresolved leads**, not better leads. Evaluation rankings remained strong on later months, but any claim about conversions needs outcomes measured at a fixed age. This is the main reason the pilot starts in shadow mode.

## 4. Proposed pilot: score-assisted lead prioritization

### 4.1 Preconditions (before the pilot starts)
1. The owner confirms the capture time of Location, Delivery_Mode, Product and email. This chooses the model that can run at first touch (`creation_time`, `assignment_time` or `post_contact`).
2. Status-change timestamps are logged, and a maturity window is chosen from them (unknown today; do not assume one).
3. Deal value and rep time are available, so outcomes can be valued. Without them, the pilot reports counts only.

### 4.2 Phase A: shadow mode (about 6 to 8 weeks)
Score every new lead when its fields are captured and store score and timestamp. Reps see nothing and work as today. After leads mature, compare scores with realized outcomes prospectively.
- **Checks:** ROC-AUC and PR-AUC with intervals; High and CONVERTED rates by score quintile; share of CONVERTED leads found in the bottom 40%; score drift versus the training period.
- **Proposed go/no-go (to be agreed with the business before starting):** top-quintile CONVERTED rate with an interval clearly above the overall rate, and the bottom 40% holding a small minority of CONVERTED leads. The retrospective reference values are top quintile 17.7% versus 5.8% overall, bottom 40% holding 1 of 80 CONVERTED (`post_contact`, test months, immature labels).

### 4.3 Phase B: randomized process pilot (about 8 to 12 weeks, only if Phase A passes)
- **Unit and assignment:** lead-level randomization, stratified by agent and week, 50/50. Randomizing within agent removes the routing confound.
- **Treatment:** the call queue is ordered by score; leads in the bottom 40% get a lower-touch path (for example automated nurture with a manual override). **Control:** current practice.
- **Primary metric (recommended): share of top-quintile leads receiving a first contact within a set time**, a leading indicator that the process actually changed.
- **Secondary:** CONVERTED rate and High Potential rate at the fixed maturity age; rep contacts per CONVERTED lead (if rep effort is logged).
- **Guardrails:** CONVERTED count among bottom-40% leads must not fall versus control (non-inferiority margin agreed in advance); rep override rate; periodic independent audit of a sample of recorded statuses, because agents enter them and treatment could change how they label.

### 4.4 Why conversion is not the primary metric
Sample sizes below come from `reports/phase7_pilot_planning.md` (alpha 0.05, power 80%, normal approximation, test-period baselines, 215 leads/week, leads assumed independent, so they are **lower bounds**):

| Outcome | Baseline | Detectable change | Leads per arm | Weeks at 50/50 |
|---|---|---|---|---|
| CONVERTED rate | 5.8% | +50% relative (to 8.7%) | 1,258 | 11.7 |
| CONVERTED rate | 5.8% | +30% relative (to 7.5%) | 3,236 | 30.1 |
| CONVERTED rate | 5.8% | +20% relative (to 6.9%) | 6,987 | 65.0 |
| High Potential rate | 45.6% | +8 points | 612 | 5.7 |
| High Potential rate | 45.6% | +5 points | 1,567 | 14.6 |

At FicZon's lead volume a conversion-powered test of a realistic effect would take 30 weeks or more, plus time for outcomes to mature. Hence the process metric as primary, conversion as secondary with a non-inferiority guardrail, and an option to extend the pilot if the business wants a conversion read.

### 4.5 Risks and mitigations
| Risk | Mitigation |
|---|---|
| Fields are filled after contact, so the model restates the outcome | Precondition 1; use the `creation_time` or `assignment_time` model if the timing demands it |
| Low-score leads are neglected and some would have converted | Bottom-40% path is lower-touch, not dropped; non-inferiority guardrail; manual override |
| Agents change how they record Status under treatment | Randomize within agent; audit sampled statuses against an independent record |
| Model drifts as the product mix or routing changes | Shadow-phase drift checks; quarterly retraining on matured leads; keep the simple segment rule as a fallback |
| Immature labels distort evaluation | Fixed maturity window and status-change timestamps (recommendation 7a) |
| Bias against segments (for example a location) | Review score distributions by location and product with the business before launch; treat scores as prioritization, not a reason to refuse service |

### 4.6 Decision rule at the end of Phase B
Scale up only if the primary metric shows the process changed, the guardrails hold, and the secondary conversion read is not worse than control. If the data is inconclusive, extend Phase B or fall back to the segment rule from recommendation 3, which captures part of the benefit with no model.

## 5. What this project cannot say
- How much revenue the model would add. No value or cost data exists.
- Whether any difference between agents is skill. Routing and agent-entered labels confound it.
- Whether the `assignment_time` or `post_contact` result applies at first touch. That hinges on one answer from the business.

-- Phase 2/3 business analysis. SQLite dialect (needs window functions, SQLite >= 3.25).
-- Runs against the PII-free `leads` table built by scripts/build_sqlite.py. Aggregates only.
-- High Potential follows configs/config.yaml (target_high). CONVERTED / Junk Lead / Not Responding / Open
-- follow kpi_definitions in the same file. Each statement is preceded by "-- name: <id>".

-- name: kpi_overview
SELECT COUNT(*) AS total_leads,
       SUM(target_high) AS high_potential_leads,
       ROUND(100.0 * AVG(target_high), 2) AS high_potential_rate_pct,
       SUM(Status = 'CONVERTED') AS converted_leads,
       ROUND(100.0 * AVG(Status = 'CONVERTED'), 2) AS conversion_rate_pct,
       ROUND(100.0 * AVG(Status IN ('Junk Lead', 'Not Responding')), 2) AS no_engagement_rate_pct,
       ROUND(100.0 * AVG(Status = 'Open'), 2) AS unresolved_share_pct
FROM leads;

-- name: source_performance
SELECT COALESCE(Source, '(missing)') AS source, COUNT(*) AS leads,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct,
       ROUND(100.0 * AVG(Status = 'CONVERTED'), 1) AS conversion_rate_pct,
       ROUND(100.0 * AVG(Status IN ('Junk Lead', 'Not Responding')), 1) AS no_engagement_pct
FROM leads GROUP BY Source HAVING COUNT(*) >= 30 ORDER BY leads DESC;

-- name: product_performance
SELECT COALESCE(Product_ID, -1) AS product_id, COUNT(*) AS leads,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM leads), 1) AS share_of_leads_pct,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct,
       ROUND(100.0 * AVG(Status = 'CONVERTED'), 1) AS conversion_rate_pct,
       ROUND(100.0 * AVG(Status IN ('Junk Lead', 'Not Responding')), 1) AS no_engagement_pct
FROM leads GROUP BY Product_ID HAVING COUNT(*) >= 30 ORDER BY leads DESC;

-- name: location_by_mode_matrix
SELECT COALESCE(Location, '(missing)') AS location, Delivery_Mode, COUNT(*) AS leads,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct
FROM leads GROUP BY Location, Delivery_Mode HAVING COUNT(*) >= 30 ORDER BY leads DESC;

-- name: low_yield_segment_share
-- Share of all leads, and of all no-engagement leads, sitting in the low-yield segments found in the EDA.
SELECT segment, COUNT(*) AS leads,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM leads), 1) AS share_of_all_leads_pct,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct,
       ROUND(100.0 * SUM(Status IN ('Junk Lead', 'Not Responding')) /
             (SELECT SUM(Status IN ('Junk Lead', 'Not Responding')) FROM leads), 1) AS share_of_all_no_engagement_pct
FROM (SELECT 'Product 15' AS segment, * FROM leads WHERE Product_ID = 15
      UNION ALL SELECT 'Location = Other Locations', * FROM leads WHERE Location = 'Other Locations'
      UNION ALL SELECT 'Delivery_Mode = Mode-5', * FROM leads WHERE Delivery_Mode = 'Mode-5')
GROUP BY segment;

-- name: monthly_kpis_with_rolling_3m
WITH m AS (
  SELECT substr(Created, 1, 7) AS month, COUNT(*) AS leads, SUM(target_high) AS high,
         SUM(Status = 'CONVERTED') AS converted, SUM(Status = 'Open') AS open_leads
  FROM leads GROUP BY month)
SELECT month, leads,
       ROUND(100.0 * high / leads, 1) AS high_rate_pct,
       ROUND(100.0 * converted / leads, 1) AS conversion_rate_pct,
       ROUND(100.0 * open_leads / leads, 1) AS open_share_pct,
       ROUND(100.0 * SUM(high) OVER w / SUM(leads) OVER w, 1) AS high_rate_rolling_3m_pct
FROM m WINDOW w AS (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) ORDER BY month;

-- name: status_mix_by_creation_month
-- Label-maturity check: statuses change as leads age, so recent months hold more in-flight statuses
-- (Potential, In Progress Positive) and fewer resolved ones (CONVERTED, LOST, Long Term).
SELECT substr(Created, 1, 7) AS month, COUNT(*) AS leads,
       ROUND(100.0 * AVG(Status = 'CONVERTED'), 1) AS converted_pct,
       ROUND(100.0 * AVG(Status IN ('Potential', 'In Progress Positive')), 1) AS potential_or_in_progress_positive_pct,
       ROUND(100.0 * AVG(Status IN ('Long Term', 'LOST')), 1) AS long_term_or_lost_pct,
       ROUND(100.0 * AVG(Status IN ('Junk Lead', 'Not Responding', 'Just Enquiry')), 1) AS junk_notresp_enquiry_pct,
       ROUND(100.0 * AVG(Status IN ('In Progress Negative', 'Open')), 1) AS in_progress_negative_or_open_pct
FROM leads GROUP BY month ORDER BY month;

-- name: agent_workload_by_month
SELECT COALESCE(Sales_Agent, '(missing)') AS agent, substr(Created, 1, 7) AS month, COUNT(*) AS leads
FROM leads GROUP BY Sales_Agent, month ORDER BY agent, month;

-- name: agent_unadjusted_outcomes
-- Unadjusted: agents were routed by mode and location, so do not read these as skill (see reports/phase6c_agent_analysis.md).
SELECT COALESCE(Sales_Agent, '(missing)') AS agent, COUNT(*) AS leads,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct,
       ROUND(100.0 * AVG(Status = 'CONVERTED'), 1) AS conversion_rate_pct,
       ROUND(100.0 * AVG(Delivery_Mode = 'Mode-5'), 1) AS pct_mode5_leads,
       ROUND(100.0 * AVG(Location = 'Other Locations'), 1) AS pct_other_locations_leads
FROM leads GROUP BY Sales_Agent HAVING COUNT(*) >= 30 ORDER BY leads DESC;

-- name: status_breakdown
SELECT Status, target_high AS in_high_potential, COUNT(*) AS leads,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM leads), 1) AS pct_of_leads
FROM leads GROUP BY Status, target_high ORDER BY leads DESC;

-- name: lead_source_hour_of_day
SELECT CAST(substr(Created, 12, 2) AS INTEGER) AS hour_of_day, COUNT(*) AS leads,
       ROUND(100.0 * AVG(target_high), 1) AS high_rate_pct
FROM leads GROUP BY hour_of_day ORDER BY hour_of_day;

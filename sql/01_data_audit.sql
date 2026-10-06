-- Phase 2 data audit. SQLite dialect. Aggregates only: no row-level PII is selected.
-- Each statement is preceded by a "-- name: <id>" marker used by scripts/run_audit.py.

-- name: row_count
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT lead_id) AS n_ids FROM leads;

-- name: date_range
SELECT MIN(Created) AS first_created, MAX(Created) AS last_created FROM leads;

-- name: missingness
SELECT
  SUM(Product_ID    IS NULL) AS product_id_null,
  SUM(Source        IS NULL) AS source_null,
  SUM(Sales_Agent   IS NULL) AS sales_agent_null,
  SUM(Location      IS NULL) AS location_null,
  SUM(mobile_missing = 1)    AS mobile_missing,
  SUM(email_valid = 0)       AS email_not_valid,
  SUM(contact_key IS NULL)   AS contact_key_null
FROM leads;

-- name: status_raw_vs_clean
SELECT r.Status AS raw_status, c.Status AS clean_status, COUNT(*) AS n
FROM leads_raw r JOIN leads c USING (lead_id)
GROUP BY r.Status, c.Status ORDER BY n DESC;

-- name: target_balance
SELECT target_high, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM leads), 2) AS pct
FROM leads GROUP BY target_high;

-- name: exact_duplicate_rows
SELECT COALESCE(SUM(cnt - 1), 0) AS extra_rows, COUNT(*) AS duplicated_groups FROM (
  SELECT COUNT(*) AS cnt FROM leads_raw
  GROUP BY Created, Product_ID, Source, Sales_Agent, Location, Delivery_Mode, Status, contact_key
  HAVING COUNT(*) > 1);

-- name: near_duplicates_ignoring_created
SELECT COALESCE(SUM(cnt - 1), 0) AS extra_rows, COUNT(*) AS duplicated_groups FROM (
  SELECT COUNT(*) AS cnt FROM leads_raw
  GROUP BY Product_ID, Source, Sales_Agent, Location, Delivery_Mode, Status, contact_key
  HAVING COUNT(*) > 1);

-- name: contact_repeat_profile
SELECT leads_per_contact, COUNT(*) AS n_contacts FROM (
  SELECT contact_key, COUNT(*) AS leads_per_contact FROM leads
  WHERE contact_key IS NOT NULL GROUP BY contact_key)
GROUP BY leads_per_contact ORDER BY leads_per_contact LIMIT 15;

-- name: contact_concentration
SELECT COUNT(DISTINCT contact_key) AS distinct_contacts,
       MAX(c) AS max_leads_one_contact,
       ROUND(100.0 * SUM(CASE WHEN c >= 10 THEN c ELSE 0 END) / SUM(c), 2) AS pct_rows_in_contacts_with_10plus_leads
FROM (SELECT contact_key, COUNT(*) AS c FROM leads WHERE contact_key IS NOT NULL GROUP BY contact_key);

-- name: monthly_volume_and_high_rate
SELECT substr(Created, 1, 7) AS month, COUNT(*) AS n_leads,
       ROUND(100.0 * AVG(target_high), 2) AS pct_high,
       ROUND(100.0 * AVG(Status = 'Open'), 2) AS pct_open
FROM leads GROUP BY month ORDER BY month;

-- name: cardinality
SELECT COUNT(DISTINCT Product_ID) AS product_id, COUNT(DISTINCT Source) AS source,
       COUNT(DISTINCT Sales_Agent) AS sales_agent, COUNT(DISTINCT Location) AS location,
       COUNT(DISTINCT Delivery_Mode) AS delivery_mode FROM leads;

-- name: rare_levels_under_30_rows
SELECT 'Source' AS col, COUNT(*) AS n_levels, SUM(n) AS rows_affected FROM (SELECT Source, COUNT(*) n FROM leads GROUP BY Source HAVING n < 30)
UNION ALL SELECT 'Product_ID', COUNT(*), SUM(n) FROM (SELECT Product_ID, COUNT(*) n FROM leads GROUP BY Product_ID HAVING n < 30)
UNION ALL SELECT 'Sales_Agent', COUNT(*), SUM(n) FROM (SELECT Sales_Agent, COUNT(*) n FROM leads GROUP BY Sales_Agent HAVING n < 30)
UNION ALL SELECT 'Location', COUNT(*), SUM(n) FROM (SELECT Location, COUNT(*) n FROM leads GROUP BY Location HAVING n < 30)
UNION ALL SELECT 'Delivery_Mode', COUNT(*), SUM(n) FROM (SELECT Delivery_Mode, COUNT(*) n FROM leads GROUP BY Delivery_Mode HAVING n < 30);

-- name: high_rate_by_source
SELECT COALESCE(Source,'(missing)') AS source, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high
FROM leads GROUP BY Source ORDER BY n DESC;

-- name: high_rate_by_delivery_mode
SELECT Delivery_Mode, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high
FROM leads GROUP BY Delivery_Mode ORDER BY n DESC;

-- name: high_rate_by_agent
SELECT COALESCE(Sales_Agent,'(missing)') AS agent, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high
FROM leads GROUP BY Sales_Agent ORDER BY n DESC;

-- name: high_rate_by_location
SELECT COALESCE(Location,'(missing)') AS location, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high
FROM leads GROUP BY Location ORDER BY n DESC;

-- name: high_rate_by_mobile_missing
SELECT mobile_missing, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high FROM leads GROUP BY mobile_missing;

-- name: high_rate_by_email_valid
SELECT email_valid, COUNT(*) AS n, ROUND(100.0*AVG(target_high),1) AS pct_high FROM leads GROUP BY email_valid;

-- name: missing_value_vs_target
SELECT 'Product_ID' AS col, COUNT(*) AS n_missing, ROUND(100.0*AVG(target_high),1) AS pct_high FROM leads WHERE Product_ID IS NULL
UNION ALL SELECT 'Source', COUNT(*), ROUND(100.0*AVG(target_high),1) FROM leads WHERE Source IS NULL
UNION ALL SELECT 'Sales_Agent', COUNT(*), ROUND(100.0*AVG(target_high),1) FROM leads WHERE Sales_Agent IS NULL
UNION ALL SELECT 'Location', COUNT(*), ROUND(100.0*AVG(target_high),1) FROM leads WHERE Location IS NULL;

-- name: agent_by_delivery_mode_overlap
SELECT Delivery_Mode, COUNT(DISTINCT Sales_Agent) AS n_agents, COUNT(*) AS n FROM leads GROUP BY Delivery_Mode;

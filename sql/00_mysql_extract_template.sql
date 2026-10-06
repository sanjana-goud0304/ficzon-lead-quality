-- READ-ONLY extraction template for the source MySQL table. NOT executed by this project's tooling:
-- I could not reach the database from the analysis environment, and the CSV export was used instead.
-- Run it yourself with a read-only account whose credentials come from your environment, never from this repo.
--
--   mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" --batch < sql/00_mysql_extract_template.sql > data/raw/leads_extract.tsv
--
-- Design: raw Mobile and EMAIL are never selected. A salted, non-reversible contact key is computed in the
-- database instead (used only for grouped checks; the models do not use it). Set @salt from your environment.
-- Column types in the source are unconfirmed (the CSV export shows Created as 'dd-mm-YYYY HH:MM' text);
-- adjust the Created expression if it is a native DATETIME. Untested SQL: review before use.

SET @salt = 'REPLACE_WITH_VALUE_FROM_ENVIRONMENT';

SELECT Created,
       Product_ID,
       Source,
       Sales_Agent,
       Location,
       Delivery_Mode,
       Status,
       SHA2(CONCAT(@salt, IF(EMAIL LIKE '%@%', LOWER(TRIM(EMAIL)), TRIM(Mobile))), 256) AS contact_key,
       (Mobile IS NULL)            AS mobile_missing,
       (EMAIL LIKE '%@%')          AS email_valid
FROM `data`;

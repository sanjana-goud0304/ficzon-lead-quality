# SQL
- `01_data_audit.sql`, `02_business_analysis.sql`: run against the local SQLite database built by `scripts/build_sqlite.py` (executed and tested via `tests/test_sql_queries.py`).
- `00_mysql_extract_template.sql`: a read-only extraction template for the source MySQL table. **Never executed** (the authoring environment could not reach the server) and it contains no credentials. Selects no raw Mobile/EMAIL; the salt is a placeholder.
- Dialect: the analysis queries use SQLite date functions (`strftime`, etc.). On MySQL use `DATE_FORMAT`, and check window-function support (MySQL 8+). The queries were not validated on MySQL.

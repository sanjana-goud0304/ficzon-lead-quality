#!/usr/bin/env bash
# Rebuild everything from the raw CSV (not included in the repo). Needs data/raw/sales_project.csv and a salt.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FICZON_HASH_SALT:?Set FICZON_HASH_SALT to any random string (see .env.example)}"
[ -f data/raw/sales_project.csv ] || { echo "Place the raw CSV at data/raw/sales_project.csv"; exit 1; }
python scripts/build_sqlite.py
python scripts/run_audit.py sql/01_data_audit.sql reports/phase2_audit_results.md
python scripts/run_audit.py sql/02_business_analysis.sql reports/phase3b_sql_business_analysis.md
python scripts/run_eda.py
python scripts/run_leakage_audit.py
python scripts/run_baselines.py
python scripts/run_tuning.py
python scripts/run_final_eval.py
python scripts/run_agent_analysis.py
python scripts/run_pilot_planning.py
python scripts/export_powerbi.py
python -m pytest -q

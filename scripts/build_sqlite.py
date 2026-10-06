"""Build a local, PII-free SQLite database from the raw CSV.

Creates two tables:
  leads_raw   - values as exported (Status/Created untouched), minus raw Mobile/EMAIL
  leads       - cleaned status, parsed timestamp, hashed contact_key, flags, target_high

Usage:
    FICZON_HASH_SALT=<random> python scripts/build_sqlite.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ficzon.data import REPO_ROOT, contact_key, load_config, load_raw, to_safe_frame  # noqa: E402


def main() -> None:
    cfg = load_config()
    raw = load_raw(cfg=cfg)
    db_path = REPO_ROOT / cfg["paths"]["sqlite_db"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    raw_safe = raw.copy()
    raw_safe["contact_key"] = contact_key(raw)
    raw_safe["mobile_missing"] = raw["Mobile"].isna().astype(int)
    raw_safe["email_valid"] = raw["EMAIL"].astype("string").str.contains("@", na=False).astype(int)
    raw_safe = raw_safe.drop(columns=["Mobile", "EMAIL"])

    clean = to_safe_frame(raw, cfg)
    clean["Created"] = clean["Created"].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean.insert(0, "lead_id", range(1, len(clean) + 1))
    raw_safe.insert(0, "lead_id", range(1, len(raw_safe) + 1))

    with sqlite3.connect(db_path) as con:
        raw_safe.to_sql("leads_raw", con, index=False)
        clean.to_sql("leads", con, index=False)
        n_raw = con.execute("SELECT COUNT(*) FROM leads_raw").fetchone()[0]
        n_clean = con.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    print(f"Built {db_path.name}: leads_raw={n_raw}, leads={n_clean} (raw Mobile/EMAIL not stored)")


if __name__ == "__main__":
    main()

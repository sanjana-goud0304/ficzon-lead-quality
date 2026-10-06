"""Data loading and PII-safe preparation.

The raw export is semicolon-delimited, CRLF, and not UTF-8 (cp1252), so the
reader settings come from configs/config.yaml rather than pandas defaults.
Raw Mobile / EMAIL never leave this module: they are reduced to a salted hash
(for grouped CV) and boolean flags.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = [
    "Created", "Product_ID", "Source", "Mobile", "EMAIL",
    "Sales_Agent", "Location", "Delivery_Mode", "Status",
]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_raw(csv_path: str | Path | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Read the raw CSV and validate its schema. Returns data exactly as stored.

    Tries the configured delimiter/encoding first, then auto-detects (the file may have been
    re-saved from Excel as comma-separated or UTF-8). Column names are trimmed of spaces and BOM.
    """
    cfg = cfg or load_config()
    csv_path = Path(csv_path) if csv_path else REPO_ROOT / cfg["paths"]["raw_csv"]
    tried = []
    seps = [cfg["csv"]["sep"]] + [s for s in (";", ",", "\t", "|") if s != cfg["csv"]["sep"]]
    encs = [cfg["csv"]["encoding"]] + [e for e in ("utf-8-sig", "cp1252", "latin-1") if e != cfg["csv"]["encoding"]]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding=enc)
            except Exception as exc:  # wrong encoding or ragged rows: try the next combination
                tried.append(f"sep={sep!r} enc={enc}: {type(exc).__name__}")
                continue
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            if all(c in df.columns for c in EXPECTED_COLUMNS):
                return df[EXPECTED_COLUMNS]
            tried.append(f"sep={sep!r} enc={enc}: columns {list(df.columns)[:9]}")
    raise ValueError(f"Could not find the expected columns {EXPECTED_COLUMNS} in {csv_path}. Tried: " + " | ".join(tried[:6]))


def parse_created(s: pd.Series) -> pd.Series:
    """Parse Created. Expected dd-mm-YYYY HH:MM; falls back to day-first parsing for re-saved files."""
    try:
        return pd.to_datetime(s, format="%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        out = pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")
        if out.isna().any():
            raise ValueError(f"{int(out.isna().sum())} Created values could not be parsed as dates")
        return out


def normalize_status(s: pd.Series) -> pd.Series:
    """Trim whitespace and fix case-only variants (e.g. 'converted' -> 'CONVERTED')."""
    s = s.astype("string").str.strip()
    canon = {v.lower(): v for v in s.dropna().unique() if v == v.upper() or v[0].isupper()}
    return s.map(lambda v: canon.get(v.lower(), v) if pd.notna(v) else v)


def _hash(value: str, salt: str) -> str:
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:16]


def contact_key(df: pd.DataFrame, salt: str | None = None) -> pd.Series:
    """Non-reversible grouping key per contact.

    Uses EMAIL when it looks like an email, otherwise Mobile, otherwise null.
    Intended only for GroupKFold; never export or join it to raw PII.
    """
    salt = salt or os.environ.get("FICZON_HASH_SALT")
    if not salt:
        raise EnvironmentError("Set FICZON_HASH_SALT (see .env.example).")
    email = df["EMAIL"].astype("string").str.strip().str.lower()
    mobile = df["Mobile"].astype("string").str.strip()
    valid_email = email.str.contains("@", na=False)
    raw = email.where(valid_email, mobile)
    return raw.map(lambda v: _hash(v, salt) if pd.notna(v) and v != "" else pd.NA)


def to_safe_frame(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return a PII-free frame: raw Mobile/EMAIL replaced by hash key + flags."""
    cfg = cfg or load_config()
    out = df.copy()
    out["Status"] = normalize_status(out["Status"])
    out["Created"] = parse_created(out["Created"])
    out["contact_key"] = contact_key(out)
    out["mobile_missing"] = out["Mobile"].isna().astype(int)
    out["email_valid"] = out["EMAIL"].astype("string").str.contains("@", na=False).astype(int)
    high = set(cfg["target"]["high_potential"])
    out["target_high"] = out["Status"].isin(high).astype(int)
    return out.drop(columns=["Mobile", "EMAIL"])


def assert_no_pii(df: pd.DataFrame, extra_forbidden: tuple[str, ...] = ()) -> None:
    """Raise if a frame about to be exported contains PII-bearing columns or email-looking / long-digit values.

    Guards the Power BI exports and any other published table. Column names are checked
    case-insensitively; string cells are scanned for '@' and for phone-like digit runs.
    """
    banned = {"mobile", "email", "contact_key", "phone", *[c.lower() for c in extra_forbidden]}
    hit = [c for c in df.columns if c.lower() in banned]
    if hit:
        raise ValueError(f"PII columns present: {hit}")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str)
        if s.str.contains("@", regex=False).any():
            raise ValueError(f"Email-like values in column {col!r}")
        if s.str.contains(r"\d{9,}", regex=True).any():
            raise ValueError(f"Phone-like digit runs in column {col!r}")

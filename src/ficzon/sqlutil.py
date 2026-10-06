"""Helpers for the named-query SQL files in sql/."""
from __future__ import annotations

import re


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split a SQL file into (name, query) pairs using '-- name: <id>' markers."""
    parts = re.split(r"^-- name:\s*(\S+)\s*$", sql_text, flags=re.M)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]

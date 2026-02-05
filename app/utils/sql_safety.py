import re
from typing import List

_DISALLOWED_KEYWORDS: List[str] = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "CALL",
    "MERGE",
]

_COMMENT_RE = re.compile(r"(--[^\n]*\n)|(/\*.*?\*/)", re.DOTALL)


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def is_safe_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    cleaned = _strip_comments(sql).strip()
    if not cleaned:
        return False

    # Disallow multiple statements separated by semicolons (except a trailing one).
    parts = [p.strip() for p in cleaned.split(";") if p.strip()]
    if len(parts) != 1:
        return False

    upper = parts[0].upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False

    for kw in _DISALLOWED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False

    return True

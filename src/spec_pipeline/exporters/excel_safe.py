from __future__ import annotations

FORMULA_PREFIXES = ("=", "+", "-", "@")


def excel_safe_text(value: str | None) -> str | None:
    """Keep untrusted document text from being interpreted as an Excel formula."""

    if value is None:
        return None
    candidate = value.lstrip(" \t\r\n")
    if candidate.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def excel_safe_value(value: object | None) -> object | None:
    return excel_safe_text(value) if isinstance(value, str) else value

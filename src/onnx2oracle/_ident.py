"""Shared Oracle identifier validation."""
from __future__ import annotations

import re

_SAFE_ORACLE_IDENT = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


def validate_oracle_name(name: str) -> str:
    if not _SAFE_ORACLE_IDENT.match(name):
        raise ValueError(
            f"Invalid Oracle model name {name!r}: must match /^[A-Z_][A-Z0-9_]*$/ "
            "(uppercase letters, digits, and underscores only, max 128 chars)."
        )
    return name

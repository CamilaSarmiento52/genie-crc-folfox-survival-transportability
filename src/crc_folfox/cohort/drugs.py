"""Exact drug normalization used by the frozen EDA."""

from collections.abc import Iterable

import pandas as pd

DRUG_COLUMNS = tuple(f"drugs_drug_{number}" for number in range(1, 6))
DRUG_START_COLUMNS = tuple(f"drugs_startdt_int_{number}" for number in range(1, 6))

FOLFOX_BACKBONE = frozenset({"fluorouracil", "leucovorin calcium", "oxaliplatin"})
ALLOWED_BIOLOGICS = frozenset(
    {
        "aflibercept",
        "atezolizumab",
        "bevacizumab",
        "cetuximab",
        "nivolumab",
        "panitumumab",
        "pembrolizumab",
        "ramucirumab",
        "ziv-aflibercept",
    }
)


def generic_drug_name(value: object) -> str:
    """Return the case-folded generic component before a synonym parenthesis."""
    if pd.isna(value):
        return ""
    return str(value).split("(", maxsplit=1)[0].strip().casefold()


def regimen_drug_set(values: Iterable[object]) -> frozenset[str]:
    return frozenset(name for value in values if (name := generic_drug_name(value)))


def contains_drug(drug_set: frozenset[str], fragment: str) -> bool:
    """Match the frozen EDA's generic-name fragment behavior."""
    return any(fragment in name for name in drug_set)

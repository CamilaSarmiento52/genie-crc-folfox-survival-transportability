"""Aggregate raw missingness and category-support profiles."""

from __future__ import annotations

import pandas as pd


def missingness_profile(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    groups = [("All", frame), *list(frame.groupby("institution", sort=True))]
    for institution, group in groups:
        for column in columns:
            missing = int(group[column].isna().sum())
            rows.append(
                {
                    "institution": institution,
                    "variable": column,
                    "patients": int(len(group)),
                    "missing_n": missing,
                    "missing_percent": 100 * missing / len(group) if len(group) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def category_frequency_profile(
    frame: pd.DataFrame, columns: tuple[str, ...], rare_threshold: int = 10
) -> pd.DataFrame:
    rows = []
    groups = [("All", frame), *list(frame.groupby("institution", sort=True))]
    institutions = set(frame["institution"].dropna().unique())
    for column in columns:
        normalized = frame[column].astype("string").fillna("<MISSING>")
        levels = sorted(normalized.unique())
        level_institutions = (
            frame.assign(_level=normalized).groupby("_level")["institution"].nunique().to_dict()
        )
        for institution, group in groups:
            values = group[column].astype("string").fillna("<MISSING>")
            counts = values.value_counts(dropna=False).reindex(levels, fill_value=0)
            for level, count in counts.items():
                rows.append(
                    {
                        "institution": institution,
                        "variable": column,
                        "level": str(level),
                        "patients": int(count),
                        "percent": 100 * int(count) / len(group),
                        "rare_within_institution": institution != "All"
                        and int(count) < rare_threshold,
                        "absent_from_institution": institution != "All" and int(count) == 0,
                        "institution_exclusive": institution != "All"
                        and level != "<MISSING>"
                        and level_institutions.get(level, 0) == 1
                        and len(institutions) > 1,
                    }
                )
    return pd.DataFrame(rows)

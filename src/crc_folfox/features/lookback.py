"""Deterministic selection of the closest eligible pre-FOLFOX source record."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.features.manifest import FeatureContractError


@dataclass(frozen=True)
class LookbackSpec:
    name: str
    source_day: str
    key_columns: tuple[str, ...]
    value_columns: tuple[str, ...]


@dataclass(frozen=True)
class LookbackResult:
    data: pd.DataFrame
    audit: pd.DataFrame
    spec: LookbackSpec
    window_days: int | None


def closest_prior_record(
    cohort: pd.DataFrame,
    source: pd.DataFrame,
    spec: LookbackSpec,
    window_days: int | None = None,
) -> LookbackResult:
    """Select one prior row per patient using an inclusive deterministic lookback."""
    if window_days is not None and (not isinstance(window_days, int) or window_days < 0):
        raise FeatureContractError("Lookback window must be a nonnegative integer or None")
    cohort_required = {"record_id", "institution", "regimen_origin_day"}
    source_required = {
        "record_id",
        "institution",
        spec.source_day,
        *spec.key_columns,
        *spec.value_columns,
    }
    if cohort_required - set(cohort) or source_required - set(source):
        raise FeatureContractError("Lookback inputs are missing required columns")
    if not cohort["record_id"].is_unique:
        raise FeatureContractError("Lookback cohort must be unique by patient")
    joined = cohort.loc[:, list(cohort_required)].merge(
        source.loc[:, list(source_required)],
        on=["record_id", "institution"],
        how="left",
        validate="one_to_many",
    )
    joined["source_day"] = pd.to_numeric(joined[spec.source_day], errors="coerce")
    joined["lookback_days"] = joined["regimen_origin_day"] - joined["source_day"]
    eligible = joined["lookback_days"].ge(0)
    if window_days is not None:
        eligible &= joined["lookback_days"].le(window_days)
    eligible &= joined[list(spec.key_columns)].notna().all(axis=1)
    candidates = joined.loc[eligible].copy()

    if candidates.empty:
        selected = candidates
        conflicts = 0
    else:
        value_signature = candidates[list(spec.value_columns)].astype("string").fillna("<MISSING>")
        candidates["_value_signature"] = value_signature.agg("|".join, axis=1)
        same_day = candidates.groupby(["record_id", "source_day"])["_value_signature"].nunique()
        conflicts = int(same_day.gt(1).sum())
        sort_columns = ["record_id", "lookback_days", *spec.key_columns]
        selected = (
            candidates.sort_values(
                sort_columns,
                ascending=[True, True, *([False] * len(spec.key_columns))],
                kind="mergesort",
            )
            .drop_duplicates("record_id")
            .drop(columns="_value_signature")
        )
        selected["selection_reason"] = "closest_prior"

    audit = pd.DataFrame(
        [
            {"check": "cohort_patients", "count": int(len(cohort))},
            {"check": "patients_selected", "count": int(selected["record_id"].nunique())},
            {"check": "eligible_source_rows", "count": int(len(candidates))},
            {"check": "conflicting_same_day_values", "count": conflicts},
            {
                "check": "invalid_or_missing_source_days",
                "count": int(joined["source_day"].isna().sum()),
            },
        ]
    )
    result = LookbackResult(selected.reset_index(drop=True), audit, spec, window_days)
    validate_lookback_result(result)
    return result


def validate_lookback_result(result: LookbackResult) -> None:
    frame = result.data
    if not frame["record_id"].is_unique:
        raise FeatureContractError("Lookback result is not unique by patient")
    invalid = frame["lookback_days"].lt(0)
    if result.window_days is not None:
        invalid |= frame["lookback_days"].gt(result.window_days)
    if invalid.any():
        raise FeatureContractError(
            f"Lookback provenance contains {int(invalid.sum())} ineligible records"
        )

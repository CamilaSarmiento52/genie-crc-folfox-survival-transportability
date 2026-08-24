"""Cancer-panel sequencing entry and delayed-entry feasibility summaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.cohort import CohortResult
from crc_folfox.features.manifest import FeatureContractError


@dataclass(frozen=True)
class SequencingEntryResult:
    data: pd.DataFrame
    lag_summary: pd.DataFrame


def build_sequencing_entry(
    cohort: CohortResult | pd.DataFrame, cancer_panel_tests: pd.DataFrame
) -> SequencingEntryResult:
    """Derive first same-cancer report and nonnegative FOLFOX-relative entry."""
    source = cohort.cohort if isinstance(cohort, CohortResult) else cohort
    required = {"record_id", "institution", "ca_seq", "regimen_origin_day"}
    if required - set(source):
        raise FeatureContractError("Sequencing entry cohort is missing required columns")
    if not source["record_id"].is_unique:
        raise FeatureContractError("Sequencing entry cohort must be unique by patient")
    report_required = {
        "record_id",
        "institution",
        "ca_seq",
        "cpt_number",
        "dob_cpt_report_days",
    }
    if report_required - set(cancer_panel_tests):
        raise FeatureContractError("Cancer-panel table is missing entry fields")
    joined = source.loc[:, list(required)].merge(
        cancer_panel_tests.loc[:, list(report_required)],
        on=["record_id", "institution", "ca_seq"],
        how="left",
        validate="one_to_many",
    )
    joined["dob_cpt_report_days"] = pd.to_numeric(joined["dob_cpt_report_days"], errors="coerce")
    first = (
        joined.sort_values(
            ["record_id", "dob_cpt_report_days", "cpt_number"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .drop_duplicates("record_id")
        .rename(columns={"dob_cpt_report_days": "first_report_day"})
    )
    missing = int(first["first_report_day"].isna().sum())
    if missing:
        raise FeatureContractError(f"Sequencing entry has {missing} patients without linked report")
    first["sequencing_lag_days"] = first["first_report_day"] - first["regimen_origin_day"]
    first["entry"] = first["sequencing_lag_days"].clip(lower=0)
    invalid = int((~pd.to_numeric(first["entry"], errors="coerce").ge(0)).sum())
    if invalid:
        raise FeatureContractError(f"Sequencing entry contains {invalid} invalid values")

    rows = []
    for institution, group in [("All", first), *list(first.groupby("institution", sort=True))]:
        lag = group["sequencing_lag_days"]
        rows.append(
            {
                "institution": institution,
                "patients": int(len(group)),
                "missing_reports": int(group["first_report_day"].isna().sum()),
                "by_folfox": int(lag.le(0).sum()),
                "median_lag_days": float(lag.median()),
                "lag_q1_days": float(lag.quantile(0.25)),
                "lag_q3_days": float(lag.quantile(0.75)),
                "lag_min_days": float(lag.min()),
                "lag_max_days": float(lag.max()),
            }
        )
    columns = [
        "record_id",
        "institution",
        "ca_seq",
        "regimen_origin_day",
        "first_report_day",
        "sequencing_lag_days",
        "entry",
    ]
    return SequencingEntryResult(first.loc[:, columns].reset_index(drop=True), pd.DataFrame(rows))


def horizon_entry_support(os_data: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Count structurally eligible patients entered and deaths observed by each horizon."""
    rows = []
    groups = [("All", os_data), *list(os_data.groupby("institution", sort=True))]
    for horizon in horizons:
        for institution, group in groups:
            entered = group["entry"].le(horizon) & group["entry"].lt(group["exit"])
            deaths = entered & group["event"].eq(1) & group["exit"].le(horizon)
            rows.append(
                {
                    "horizon_days": horizon,
                    "institution": institution,
                    "entered_patients": int(entered.sum()),
                    "events_by_horizon": int(deaths.sum()),
                }
            )
    return pd.DataFrame(rows)

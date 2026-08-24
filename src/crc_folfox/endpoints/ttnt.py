"""Death-inclusive same-cancer time-to-next-treatment reconstruction."""

import numpy as np
import pandas as pd

from crc_folfox.cohort.build import CohortResult
from crc_folfox.cohort.regimens import prepare_regimens
from crc_folfox.endpoints.validate import EndpointContractError, EndpointResult


def build_ttnt(
    cohort: CohortResult | pd.DataFrame, regimens: pd.DataFrame | None = None
) -> EndpointResult:
    if isinstance(cohort, CohortResult):
        source = cohort.cohort
        prepared = cohort.prepared_regimens if regimens is None else prepare_regimens(regimens)
    else:
        source = cohort
        if regimens is None:
            raise ValueError("regimens are required when cohort is a DataFrame")
        prepared = prepare_regimens(regimens)
    result = source.copy(deep=True)
    future = result[["record_id", "ca_seq", "regimen_number", "regimen_origin_day"]].merge(
        prepared[["record_id", "ca_seq", "regimen_number", "regimen_origin_day"]],
        on=["record_id", "ca_seq"],
        how="left",
        suffixes=("_index", "_future"),
    )
    future = future.loc[
        future["regimen_number_future"].ne(future["regimen_number_index"])
        & future["regimen_origin_day_future"].ge(future["regimen_origin_day_index"])
    ]
    next_start = future.groupby("record_id")["regimen_origin_day_future"].min()
    result["next_same_cancer_regimen_day"] = result["record_id"].map(next_start)
    result["days_to_next_same_cancer_regimen"] = (
        result["next_same_cancer_regimen_day"] - result["regimen_origin_day"]
    )
    death_day = pd.to_numeric(result["hybrid_death_int"], errors="coerce")
    result["days_to_death"] = death_day - result["regimen_origin_day"]
    result["expected_ttnt_days"] = np.where(
        result["os_g_status"].eq(1),
        np.fmin(result["days_to_next_same_cancer_regimen"], result["days_to_death"]),
        result["days_to_next_same_cancer_regimen"],
    )
    result["expected_ttnt_days"] = np.where(
        pd.isna(result["expected_ttnt_days"]),
        np.where(
            result["os_g_status"].eq(1),
            result["days_to_death"],
            result["dob_lastalive_int"] - result["regimen_origin_day"],
        ),
        result["expected_ttnt_days"],
    )
    result["ttnt_difference_days"] = result["ttnt_ca_seq_days"] - result["expected_ttnt_days"]
    result["ttnt_event_source"] = np.select(
        [
            result["ttnt_ca_seq_status"].eq(0),
            result["days_to_next_same_cancer_regimen"].notna()
            & (
                result["days_to_death"].isna()
                | result["days_to_next_same_cancer_regimen"].le(result["days_to_death"])
            ),
        ],
        ["Censored", "Subsequent same-cancer regimen"],
        default="Death",
    )
    validation = pd.DataFrame(
        [
            {
                "check": "Missing same-cancer TTNT status",
                "patients": int(result["ttnt_ca_seq_status"].isna().sum()),
            },
            {
                "check": "Missing same-cancer TTNT time",
                "patients": int(result["ttnt_ca_seq_days"].isna().sum()),
            },
            {
                "check": "Negative same-cancer TTNT time",
                "patients": int(result["ttnt_ca_seq_days"].lt(0).sum()),
            },
            {
                "check": "Zero-day same-cancer TTNT event",
                "patients": int(
                    (result["ttnt_ca_seq_status"].eq(1) & result["ttnt_ca_seq_days"].eq(0)).sum()
                ),
            },
            {
                "check": "Supplied TTNT differs from reconstruction",
                "patients": int(result["ttnt_difference_days"].ne(0).sum()),
            },
        ]
    )
    invalid_status = int((~result["ttnt_ca_seq_status"].isin([0, 1])).sum())
    missing_or_negative = int(
        result[["ttnt_ca_seq_status", "ttnt_ca_seq_days", "expected_ttnt_days"]]
        .isna()
        .any(axis=1)
        .sum()
        + result["ttnt_ca_seq_days"].lt(0).sum()
        + result["expected_ttnt_days"].lt(0).sum()
    )
    if invalid_status or missing_or_negative:
        raise EndpointContractError(
            "TTNT-D contract failed: "
            f"invalid_status={invalid_status}, invalid_time={missing_or_negative}"
        )
    by_institution = (
        result.groupby("institution")
        .agg(
            patients=("record_id", "size"),
            events=("ttnt_ca_seq_status", "sum"),
            median_observed_days=("ttnt_ca_seq_days", "median"),
        )
        .reset_index()
    )
    event_sources = (
        result["ttnt_event_source"]
        .value_counts()
        .rename_axis("event_source")
        .reset_index(name="patients")
    )
    return EndpointResult(result, validation, by_institution, event_sources)

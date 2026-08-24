"""Overall-survival reconstruction from FOLFOX initiation."""

import numpy as np
import pandas as pd

from crc_folfox.cohort.build import CohortResult
from crc_folfox.endpoints.validate import EndpointContractError, EndpointResult


def build_os(cohort: CohortResult | pd.DataFrame) -> EndpointResult:
    source = cohort.cohort if isinstance(cohort, CohortResult) else cohort
    result = source.copy(deep=True)
    result["expected_os_days"] = np.where(
        result["os_g_status"].eq(1),
        result["hybrid_death_int"] - result["regimen_origin_day"],
        result["dob_lastalive_int"] - result["regimen_origin_day"],
    )
    result["os_difference_days"] = result["tt_os_g_days"] - result["expected_os_days"]
    validation = pd.DataFrame(
        [
            {"check": "Missing OS status", "patients": int(result["os_g_status"].isna().sum())},
            {"check": "Missing OS time", "patients": int(result["tt_os_g_days"].isna().sum())},
            {"check": "Negative OS time", "patients": int(result["tt_os_g_days"].lt(0).sum())},
            {"check": "Zero-day OS", "patients": int(result["tt_os_g_days"].eq(0).sum())},
            {
                "check": "Supplied OS differs from recomputed OS",
                "patients": int(result["os_difference_days"].ne(0).sum()),
            },
        ]
    )
    invalid_status = int((~result["os_g_status"].isin([0, 1])).sum())
    missing_or_negative = int(
        result[["os_g_status", "tt_os_g_days", "expected_os_days"]].isna().any(axis=1).sum()
        + result["tt_os_g_days"].lt(0).sum()
        + result["expected_os_days"].lt(0).sum()
    )
    if invalid_status or missing_or_negative:
        raise EndpointContractError(
            "OS contract failed: "
            f"invalid_status={invalid_status}, invalid_time={missing_or_negative}"
        )
    by_institution = (
        result.groupby("institution")
        .agg(
            patients=("record_id", "size"),
            events=("os_g_status", "sum"),
            median_observed_days=("tt_os_g_days", "median"),
        )
        .reset_index()
    )
    return EndpointResult(result, validation, by_institution)

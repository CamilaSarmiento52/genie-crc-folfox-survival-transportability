"""End-to-end construction of the frozen cohort-reconstruction workflow source cohort."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.cohort.metastatic import derive_metastatic_presentation
from crc_folfox.cohort.phenotype import (
    EXACT_FOLFOX,
    FOLFOX_PLUS_BIOLOGIC,
    classify_first_regimen,
    is_broad_folfox,
)
from crc_folfox.cohort.regimens import (
    add_prior_oxaliplatin,
    first_post_metastatic_regimen,
    prepare_regimens,
)
from crc_folfox.data.contracts import ReleaseContractError, ReleaseTables


@dataclass(frozen=True)
class CohortResult:
    cohort: pd.DataFrame
    cohort_flow: pd.DataFrame
    phenotype_summary: pd.DataFrame
    characteristics: pd.DataFrame
    advanced_cancers: pd.DataFrame
    post_metastatic_regimens: pd.DataFrame
    first_post_metastatic: pd.DataFrame
    prepared_regimens: pd.DataFrame


@dataclass(frozen=True)
class GenomicTimingResult:
    summary: pd.DataFrame
    baseline_patients: int
    reports_after_folfox: int


def build_cohort(tables: ReleaseTables) -> CohortResult:
    index_cancers = derive_metastatic_presentation(tables["cancer_index"])
    advanced = index_cancers.loc[index_cancers["advanced_origin_day"].notna()].copy()
    prepared = prepare_regimens(tables["regimens"])
    post, first = first_post_metastatic_regimen(prepared, advanced)
    first["regimen_class"] = first["drug_set"].map(classify_first_regimen)
    first["broad_folfox"] = first["drug_set"].map(is_broad_folfox)
    first["primary_folfox"] = first["regimen_class"].isin([EXACT_FOLFOX, FOLFOX_PLUS_BIOLOGIC])
    first["adult"] = first["age_at_regimen"].ge(18)

    cohort = first.loc[first["primary_folfox"] & first["adult"]].copy()
    cohort = add_prior_oxaliplatin(cohort, prepared)
    cohort["biologic_with_folfox"] = cohort["regimen_class"].eq(FOLFOX_PLUS_BIOLOGIC)
    patient_columns = [
        "record_id",
        "naaccr_sex_code",
        "race_ethnicity",
        "hybrid_death_int",
        "dob_lastalive_int",
    ]
    available_patient_columns = [
        column for column in patient_columns if column in tables["patient"].columns
    ]
    cohort = cohort.merge(
        tables["patient"][available_patient_columns],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if not cohort["record_id"].is_unique:
        raise ReleaseContractError("Primary cohort is not unique by record_id")

    cohort_flow = pd.DataFrame(
        [
            {"step": "All release patients", "patients": tables["patient"]["record_id"].nunique()},
            {
                "step": "Documented advanced index CRC",
                "patients": advanced["record_id"].nunique(),
            },
            {
                "step": "At least one linked post-metastatic regimen",
                "patients": post["record_id"].nunique(),
            },
            {
                "step": "First regimen contains FOLFOX backbone",
                "patients": int(first["broad_folfox"].sum()),
            },
            {"step": "Primary FOLFOX definition and adult", "patients": len(cohort)},
            {
                "step": "Complete regimen-level OS",
                "patients": int(cohort[["os_g_status", "tt_os_g_days"]].notna().all(axis=1).sum()),
            },
        ]
    )
    phenotype = (
        first.groupby("regimen_class", as_index=False)
        .agg(patients=("record_id", "nunique"))
        .sort_values("patients", ascending=False, ignore_index=True)
    )
    characteristics = pd.DataFrame(
        [
            {"characteristic": "Patients", "value": len(cohort)},
            {
                "characteristic": "De novo Stage IV",
                "value": int(cohort["metastatic_presentation"].eq("De novo Stage IV").sum()),
            },
            {
                "characteristic": "Recurrent/metachronous",
                "value": int(cohort["metastatic_presentation"].eq("Recurrent/metachronous").sum()),
            },
            {
                "characteristic": "FOLFOX plus biologic",
                "value": int(cohort["biologic_with_folfox"].sum()),
            },
            {
                "characteristic": "Previous oxaliplatin",
                "value": int(cohort["prior_oxaliplatin"].sum()),
            },
            {
                "characteristic": "Physical regimens with multiple advanced associations",
                "value": int((cohort["advanced_associations_per_physical_regimen"] > 1).sum()),
            },
        ]
    )
    return CohortResult(
        cohort, cohort_flow, phenotype, characteristics, advanced, post, first, prepared
    )


def build_baseline_genomic_audit(
    cohort: CohortResult | pd.DataFrame, cancer_panel_tests: pd.DataFrame
) -> GenomicTimingResult:
    frame = cohort.cohort if isinstance(cohort, CohortResult) else cohort
    relative = frame[["record_id", "institution", "ca_seq", "regimen_origin_day"]].merge(
        cancer_panel_tests,
        on=["record_id", "institution", "ca_seq"],
        how="left",
        suffixes=("", "_cpt"),
    )
    relative["cpt_lag_from_folfox_days"] = (
        relative["dob_cpt_report_days"] - relative["regimen_origin_day"]
    )
    baseline_all = relative.loc[relative["cpt_lag_from_folfox_days"].le(0)].copy()
    baseline = (
        baseline_all.sort_values(
            ["record_id", "dob_cpt_report_days", "cpt_number"],
            ascending=[True, False, False],
        )
        .drop_duplicates("record_id")
        .copy()
    )
    summary = (
        frame.groupby("institution")
        .agg(cohort_patients=("record_id", "size"))
        .join(
            baseline.groupby("institution")["record_id"].nunique().rename("baseline_patients"),
            how="left",
        )
        .fillna(0)
        .reset_index()
    )
    summary["baseline_patients"] = summary["baseline_patients"].astype(int)
    return GenomicTimingResult(
        summary=summary,
        baseline_patients=int(baseline["record_id"].nunique()),
        reports_after_folfox=int((baseline_all["cpt_lag_from_folfox_days"] > 0).sum()),
    )

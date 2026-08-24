"""Timed extended-candidate extraction for aggregate feasibility auditing."""

from __future__ import annotations

import pandas as pd

from crc_folfox.data import ReleaseTables
from crc_folfox.features.lookback import LookbackResult, LookbackSpec, closest_prior_record

SPECS = {
    "pathology": LookbackSpec(
        "pathology",
        "path_proc_int",
        ("path_proc_number", "path_rep_number"),
        (
            "msi_testing",
            "msi_result",
            "msi_result_2",
            "msi_result_3",
            "msi_high_any",
            "mmr_testing",
            "mmr_result",
            "mmr_result_2",
            "mmr_result_3",
        ),
    ),
    "imaging": LookbackSpec(
        "imaging", "image_scan_int", ("scan_number",), ("image_ca", "image_overall")
    ),
    "medical_oncology": LookbackSpec(
        "medical_oncology", "md_onc_visit_int", ("md_visit_number",), ("md_ca_status",)
    ),
    "cea": LookbackSpec(
        "cea", "tm_spec_collect_int", ("tm_number",), ("tm_num_result", "tm_result_units")
    ),
}


def extract_extended_candidates(
    cohort: pd.DataFrame, tables: ReleaseTables, window_days: int | None = 90
) -> dict[str, LookbackResult]:
    """Return timed source extracts; pathology dates remain procedure-date proxies."""
    marker_type = tables["tumor_markers"]["tm_type"].astype("string").str.strip().str.casefold()
    cea = tables["tumor_markers"].loc[marker_type.eq("cea")].copy()
    sources = {
        "pathology": tables["pathology"],
        "imaging": tables["imaging"],
        "medical_oncology": tables["medical_oncology"],
        "cea": cea,
    }
    return {
        name: closest_prior_record(cohort, sources[name], spec, window_days)
        for name, spec in SPECS.items()
    }


def baseline_availability_profile(
    cohort: pd.DataFrame,
    tables: ReleaseTables,
    windows: tuple[int | None, ...] = (30, 90, 180, None),
) -> pd.DataFrame:
    """Summarize patients with an eligible prior source record by lookback window."""
    rows = []
    for window in windows:
        extracted = extract_extended_candidates(cohort, tables, window)
        for name, result in extracted.items():
            rows.append(
                {
                    "source": name,
                    "window_days": window if window is not None else "any_prior",
                    "patients": int(len(result.data)),
                    "percent": 100 * len(result.data) / len(cohort),
                }
            )

        same_cancer = cohort[["record_id", "institution", "ca_seq"]].merge(
            tables["cancer_panel_tests"],
            on=["record_id", "institution", "ca_seq"],
            how="inner",
        )
        genomic_spec = LookbackSpec(
            "genomic_report",
            "dob_cpt_report_days",
            ("cpt_number",),
            ("cpt_seq_assay_id",),
        )
        genomic = closest_prior_record(cohort, same_cancer, genomic_spec, window)
        rows.append(
            {
                "source": "genomic_report",
                "window_days": window if window is not None else "any_prior",
                "patients": int(len(genomic.data)),
                "percent": 100 * len(genomic.data) / len(cohort),
            }
        )
    return pd.DataFrame(rows)

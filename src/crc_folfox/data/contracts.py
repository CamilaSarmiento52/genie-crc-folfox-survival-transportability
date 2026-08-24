"""Central contracts for the nine GENIE BPC CRC analytic tables."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pandas as pd


class ReleaseContractError(ValueError):
    """A privacy-safe failure of a release boundary contract."""


@dataclass(frozen=True)
class TableSpec:
    """Filename, candidate grain, and cohort-reconstruction workflow column requirements for a
    table."""

    filename: str
    key: tuple[str, ...]
    required: tuple[str, ...]
    numeric: tuple[str, ...] = ()

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*COMMON_COLUMNS, *self.key, *self.required)))


COMMON_COLUMNS = ("cohort", "record_id", "institution")

TABLE_SPECS: dict[str, TableSpec] = {
    "patient": TableSpec(
        "patient_level_dataset.csv",
        ("record_id",),
        ("hybrid_death_int", "dob_lastalive_int", "naaccr_sex_code", "race_ethnicity"),
        ("hybrid_death_int", "dob_lastalive_int"),
    ),
    "cancer_index": TableSpec(
        "cancer_level_dataset_index.csv",
        ("record_id", "ca_seq"),
        (
            "stage_dx_iv",
            "dmets_stage_i_iii",
            "dob_ca_dx_days",
            "dx_to_dmets_days",
            "age_dx",
            "crc_type",
            "ca_d_site",
            "ca_grade",
            "dmets_abdomen",
            "dmets_bone",
            "dmets_brain",
            "dmets_liver",
            "dmets_pelvis",
            "dmets_thorax",
        ),
        ("ca_seq", "dmets_stage_i_iii", "dob_ca_dx_days", "dx_to_dmets_days", "age_dx"),
    ),
    "cancer_non_index": TableSpec(
        "cancer_level_dataset_non_index.csv",
        ("record_id", "ca_seq"),
        ("dob_ca_dx_days",),
        ("ca_seq", "dob_ca_dx_days"),
    ),
    "regimens": TableSpec(
        "regimen_cancer_level_dataset.csv",
        ("record_id", "regimen_number", "ca_seq"),
        (
            "drugs_drug_1",
            "drugs_drug_2",
            "drugs_drug_3",
            "drugs_drug_4",
            "drugs_drug_5",
            "drugs_startdt_int_1",
            "drugs_startdt_int_2",
            "drugs_startdt_int_3",
            "drugs_startdt_int_4",
            "drugs_startdt_int_5",
            "dx_reg_start_int",
            "os_g_status",
            "tt_os_g_days",
            "ttnt_any_ca_days",
            "ttnt_ca_seq_status",
            "ttnt_ca_seq_days",
        ),
        (
            "ca_seq",
            "regimen_number",
            "drugs_startdt_int_1",
            "drugs_startdt_int_2",
            "drugs_startdt_int_3",
            "drugs_startdt_int_4",
            "drugs_startdt_int_5",
            "dx_reg_start_int",
            "os_g_status",
            "tt_os_g_days",
            "ttnt_any_ca_days",
            "ttnt_ca_seq_status",
            "ttnt_ca_seq_days",
        ),
    ),
    "pathology": TableSpec(
        "pathology_report_level_dataset.csv",
        ("record_id", "path_proc_number", "path_rep_number"),
        (
            "path_proc_int",
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
        ("path_proc_number", "path_rep_number", "path_proc_int"),
    ),
    "imaging": TableSpec(
        "imaging_level_dataset.csv",
        ("record_id", "scan_number"),
        ("image_scan_int", "image_ca", "image_overall"),
        ("scan_number", "image_scan_int"),
    ),
    "medical_oncology": TableSpec(
        "med_onc_note_level_dataset.csv",
        ("record_id", "md_visit_number"),
        ("md_onc_visit_int", "md_ca_status"),
        ("md_visit_number", "md_onc_visit_int"),
    ),
    "tumor_markers": TableSpec(
        "tm_level_dataset.csv",
        ("record_id", "tm_number", "tm_type"),
        ("tm_spec_collect_int", "tm_num_result", "tm_result_units"),
        ("tm_number", "tm_spec_collect_int", "tm_num_result"),
    ),
    "cancer_panel_tests": TableSpec(
        "cancer_panel_test_level_dataset.csv",
        ("record_id", "cpt_number", "ca_seq"),
        ("dob_cpt_report_days",),
        ("cpt_number", "ca_seq", "dob_cpt_report_days"),
    ),
}


@dataclass(frozen=True)
class ReleaseTables(Mapping[str, pd.DataFrame]):
    """Read-only mapping facade over caller-independent data frames."""

    frames: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        missing = set(TABLE_SPECS) - set(self.frames)
        extra = set(self.frames) - set(TABLE_SPECS)
        if missing or extra:
            raise ReleaseContractError(
                f"Release table registry mismatch: missing={len(missing)}, extra={len(extra)}"
            )

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.frames[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def copy(self) -> ReleaseTables:
        return ReleaseTables({name: frame.copy(deep=True) for name, frame in self.frames.items()})

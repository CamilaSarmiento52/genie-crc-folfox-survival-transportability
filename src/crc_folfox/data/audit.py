"""Aggregate-only schema, grain, and linkage audits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.data.contracts import (
    TABLE_SPECS,
    ReleaseContractError,
    ReleaseTables,
)

INSTITUTIONS = frozenset({"DFCI", "MSK", "VICC"})


@dataclass(frozen=True)
class ReleaseAudit:
    schema: pd.DataFrame
    linkage: pd.DataFrame
    cross_table: pd.DataFrame


def audit_release(tables: ReleaseTables) -> ReleaseAudit:
    """Audit release structure and fail on the frozen linkage invariants."""
    patient = tables["patient"]
    if patient["record_id"].isna().any() or not patient["record_id"].is_unique:
        raise ReleaseContractError("Patient record_id must be complete and unique")

    unknown_institutions = sum(
        int((~frame["institution"].dropna().isin(INSTITUTIONS)).sum()) for frame in tables.values()
    )
    if unknown_institutions:
        raise ReleaseContractError(
            f"Release contains {unknown_institutions} rows with unknown institution labels"
        )

    schema_rows = []
    linkage_rows = []
    patient_ids = set(patient["record_id"])
    institution_lookup = patient.set_index("record_id")["institution"]
    for name, frame in tables.items():
        key = list(TABLE_SPECS[name].key)
        complete = frame[key].notna().all(axis=1)
        duplicate = frame.loc[complete, key].duplicated(keep=False)
        schema_rows.append(
            {
                "table": name,
                "expected_key": " + ".join(key),
                "rows": len(frame),
                "columns": frame.shape[1],
                "rows_with_any_missing": int(frame.isna().any(axis=1).sum()),
                "columns_with_missing": int(frame.isna().any().sum()),
                "missing_cells": int(frame.isna().sum().sum()),
                "full_row_duplicates": int(frame.duplicated().sum()),
                "duplicate_column_names": int(frame.columns.duplicated().sum()),
                "all_missing_columns": int(frame.isna().all().sum()),
                "rows_missing_key_component": int((~complete).sum()),
                "rows_in_duplicate_complete_keys": int(duplicate.sum()),
            }
        )
        orphan = int((~frame["record_id"].isin(patient_ids)).sum())
        expected_institution = frame["record_id"].map(institution_lookup)
        mismatch = int(
            (
                frame["institution"].notna()
                & expected_institution.notna()
                & frame["institution"].ne(expected_institution)
            ).sum()
        )
        linkage_rows.append(
            {
                "table": name,
                "rows": len(frame),
                "patient_ids_not_in_patient_table": orphan,
                "institution_disagreements": mismatch,
            }
        )

    linkage = pd.DataFrame(linkage_rows)
    orphan_total = int(linkage["patient_ids_not_in_patient_table"].sum())
    mismatch_total = int(linkage["institution_disagreements"].sum())
    if orphan_total or mismatch_total:
        raise ReleaseContractError(
            "Release linkage failed: "
            f"orphan_rows={orphan_total}, institution_disagreements={mismatch_total}"
        )

    all_cancers = pd.concat(
        [
            tables["cancer_index"][["record_id", "ca_seq"]],
            tables["cancer_non_index"][["record_id", "ca_seq"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    regimen_links = (
        tables["regimens"]
        .dropna(subset=["ca_seq"])[["record_id", "ca_seq"]]
        .merge(all_cancers.assign(found=True), on=["record_id", "ca_seq"], how="left")
    )
    cpt_links = (
        tables["cancer_panel_tests"]
        .dropna(subset=["ca_seq"])[["record_id", "ca_seq"]]
        .merge(
            tables["cancer_index"][["record_id", "ca_seq"]].assign(found=True),
            on=["record_id", "ca_seq"],
            how="left",
        )
    )
    physical = (
        tables["regimens"]
        .groupby(["record_id", "regimen_number"], dropna=False)
        .agg(associated_cancers=("ca_seq", "nunique"))
        .reset_index()
    )
    cross_table = pd.DataFrame(
        [
            {
                "check": "regimen cancer links",
                "eligible_rows": len(regimen_links),
                "matched_rows": int(regimen_links["found"].fillna(False).sum()),
            },
            {
                "check": "cancer panel index links",
                "eligible_rows": len(cpt_links),
                "matched_rows": int(cpt_links["found"].fillna(False).sum()),
            },
            {
                "check": "physical regimens with multiple cancer associations",
                "eligible_rows": len(physical),
                "matched_rows": int((physical["associated_cancers"] > 1).sum()),
            },
        ]
    )
    return ReleaseAudit(pd.DataFrame(schema_rows), linkage, cross_table)

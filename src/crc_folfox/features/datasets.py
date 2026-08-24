"""Endpoint-specific delayed-entry analysis dataset assembly."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from crc_folfox.cohort import CohortResult
from crc_folfox.data import ReleaseTables
from crc_folfox.endpoints import EndpointResult
from crc_folfox.features.audit import truncation_assumption_audit
from crc_folfox.features.clinical import materialize_clinical_features
from crc_folfox.features.entry import (
    SequencingEntryResult,
    build_sequencing_entry,
    horizon_entry_support,
)
from crc_folfox.features.manifest import FeatureContractError, FeatureManifest
from crc_folfox.features.profiles import category_frequency_profile, missingness_profile


@dataclass(frozen=True)
class AnalysisDatasetResult:
    data: pd.DataFrame
    excluded: pd.DataFrame
    summary: pd.DataFrame
    folds: pd.DataFrame
    complete_case_summary: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class AnalysisDatasets:
    source: pd.DataFrame
    features: pd.DataFrame
    entry: SequencingEntryResult
    os: AnalysisDatasetResult
    ttnt: AnalysisDatasetResult
    missingness: pd.DataFrame
    categories: pd.DataFrame
    horizon_support: pd.DataFrame
    truncation_audit: pd.DataFrame

    def aggregate_manifest(self) -> dict:
        payload = {
            "source": {"patients": int(len(self.source))},
            "os": _manifest_for_endpoint(self.os),
            "ttnt": _manifest_for_endpoint(self.ttnt),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


def _manifest_for_endpoint(result: AnalysisDatasetResult) -> dict:
    return {
        "summary": result.summary.to_dict("records"),
        "folds": result.folds.to_dict("records"),
        "complete_case": result.complete_case_summary.to_dict("records"),
        "exclusions": result.excluded["exclusion_reason"].value_counts().to_dict(),
    }


def _build_endpoint_dataset(
    endpoint_name: str,
    endpoint: EndpointResult | pd.DataFrame,
    features: pd.DataFrame,
    entry: pd.DataFrame,
    manifest: FeatureManifest,
) -> AnalysisDatasetResult:
    source = endpoint.data if isinstance(endpoint, EndpointResult) else endpoint
    if endpoint_name == "os":
        time_column, event_column = "tt_os_g_days", "os_g_status"
        extra_columns: list[str] = []
    elif endpoint_name == "ttnt":
        time_column, event_column = "ttnt_ca_seq_days", "ttnt_ca_seq_status"
        extra_columns = ["ttnt_event_source"]
    else:
        raise FeatureContractError("Unknown endpoint dataset")
    outcome_columns = ["record_id", "institution", time_column, event_column, *extra_columns]
    if set(outcome_columns) - set(source):
        raise FeatureContractError("Endpoint source is missing analysis fields")
    outcome = source.loc[:, outcome_columns].rename(
        columns={time_column: "exit", event_column: "event"}
    )
    combined = (
        outcome.merge(entry.loc[:, ["record_id", "sequencing_lag_days", "entry"]], on="record_id")
        .merge(features, on=["record_id", "institution"], validate="one_to_one")
        .copy()
    )
    combined["eligible"] = combined["entry"].lt(combined["exit"])
    combined["exclusion_reason"] = pd.Series(pd.NA, index=combined.index, dtype="string")
    combined.loc[~combined["eligible"], "exclusion_reason"] = "entry_at_or_after_exit"
    data = combined.loc[combined["eligible"]].drop(columns=["eligible", "exclusion_reason"])
    excluded = combined.loc[~combined["eligible"], ["exclusion_reason"]].copy()

    validation = validate_analysis_dataset(data, len(combined))
    summary_rows = []
    for institution, group in [("All", data), *list(data.groupby("institution", sort=True))]:
        source_n = (
            len(combined)
            if institution == "All"
            else int(combined["institution"].eq(institution).sum())
        )
        summary_rows.append(
            {
                "institution": institution,
                "source_patients": source_n,
                "excluded_patients": source_n - len(group),
                "analysis_patients": int(len(group)),
                "events": int(group["event"].sum()),
                "censored": int(group["event"].eq(0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    fold_rows = []
    for held_out in sorted(data["institution"].unique()):
        held = data.loc[data["institution"].eq(held_out)]
        train = data.loc[data["institution"].ne(held_out)]
        fold_rows.append(
            {
                "held_out_institution": held_out,
                "held_out_patients": int(len(held)),
                "held_out_events": int(held["event"].sum()),
                "training_patients": int(len(train)),
                "training_events": int(train["event"].sum()),
            }
        )
    folds = pd.DataFrame(fold_rows)

    complete = data[list(manifest.core)].notna().all(axis=1)
    complete_rows = [
        {
            "institution": "All",
            "patients": int(complete.sum()),
            "events": int(data.loc[complete, "event"].sum()),
        }
    ]
    for institution, group in data.assign(_complete=complete).groupby("institution", sort=True):
        mask = group["_complete"]
        complete_rows.append(
            {
                "institution": institution,
                "patients": int(mask.sum()),
                "events": int(group.loc[mask, "event"].sum()),
            }
        )
    return AnalysisDatasetResult(
        data.reset_index(drop=True),
        excluded.reset_index(drop=True),
        summary,
        folds,
        pd.DataFrame(complete_rows),
        validation,
    )


def validate_analysis_dataset(
    data: pd.DataFrame, source_patients: int | None = None
) -> pd.DataFrame:
    """Validate one-interval counting-process structure with aggregate-safe failures."""
    required = {"record_id", "institution", "entry", "exit", "event"}
    if required - set(data):
        raise FeatureContractError("Analysis dataset is missing structural columns")
    checks = {
        "duplicate_patients": int(data["record_id"].duplicated().sum()),
        "missing_outcomes": int(data[["entry", "exit", "event"]].isna().any(axis=1).sum()),
        "negative_entry": int(data["entry"].lt(0).sum()),
        "entry_not_before_exit": int(data["entry"].ge(data["exit"]).sum()),
        "invalid_event": int((~data["event"].isin([0, 1])).sum()),
        "source_rows_unreconciled": 0
        if source_patients is None or len(data) <= source_patients
        else len(data) - source_patients,
    }
    if any(checks.values()):
        raise FeatureContractError(
            "Analysis dataset structural contract failed: "
            + ", ".join(f"{name}={value}" for name, value in checks.items())
        )
    return pd.DataFrame([{"check": key, "patients": value} for key, value in checks.items()])


def build_analysis_datasets(
    cohort: CohortResult,
    os_endpoint: EndpointResult,
    ttnt_endpoint: EndpointResult,
    tables: ReleaseTables,
    manifest: FeatureManifest | None = None,
) -> AnalysisDatasets:
    """Build source, OS, and TTNT-D datasets without fitting preprocessing or models."""
    active = manifest or FeatureManifest.load()
    features = materialize_clinical_features(cohort, active)
    entry = build_sequencing_entry(cohort, tables["cancer_panel_tests"])
    source = cohort.cohort.loc[
        :, ["record_id", "institution", "ca_seq", "regimen_origin_day"]
    ].merge(entry.data, on=["record_id", "institution", "ca_seq", "regimen_origin_day"])
    source = source.merge(features, on=["record_id", "institution"], validate="one_to_one")
    os_result = _build_endpoint_dataset("os", os_endpoint, features, entry.data, active)
    ttnt_result = _build_endpoint_dataset("ttnt", ttnt_endpoint, features, entry.data, active)
    missingness = missingness_profile(features, active.core)
    categorical = tuple(
        feature.name for feature in active.features if feature.role in {"categorical", "binary"}
    )
    categories = category_frequency_profile(features, categorical)
    horizon_support = horizon_entry_support(os_result.data, (84, 365, 730))
    truncation_frame = os_endpoint.data.loc[:, ["record_id", "tt_os_g_days", "os_g_status"]].rename(
        columns={"tt_os_g_days": "exit", "os_g_status": "event"}
    )
    truncation_frame = truncation_frame.merge(entry.data, on="record_id").merge(
        features, on=["record_id", "institution"], validate="one_to_one"
    )
    truncation = truncation_assumption_audit(truncation_frame)
    return AnalysisDatasets(
        source,
        features,
        entry,
        os_result,
        ttnt_result,
        missingness,
        categories,
        horizon_support,
        truncation,
    )

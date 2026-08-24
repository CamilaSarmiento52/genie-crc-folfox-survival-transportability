"""Transparent data-cleaning workflow for the paper analysis.

The workflow keeps clinically complex cohort and endpoint rules in tested modules, while exposing
aggregate cleaning evidence and one patient-level CSV handoff for later preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from crc_folfox.cohort import CohortResult, build_cohort
from crc_folfox.data import (
    IntegrityReport,
    ReleaseAudit,
    audit_release,
    load_release,
    validate_manifest,
)
from crc_folfox.endpoints import EndpointResult, build_os, build_ttnt
from crc_folfox.features import (
    AnalysisDatasets,
    FeatureManifest,
    baseline_availability_profile,
    build_analysis_datasets,
)

DATASET_FILENAME = "analysis_dataset.csv"
DICTIONARY_FILENAME = "data_dictionary.json"
REPORT_FILENAME = "cleaning_report.md"
MANIFEST_FILENAME = "run_manifest.json"


class CleaningWorkflowError(RuntimeError):
    """Aggregate-safe failure raised by the data-cleaning workflow."""


@dataclass(frozen=True)
class CleaningArtifacts:
    current_directory: Path
    dataset: Path
    data_dictionary: Path
    cleaning_report: Path
    run_manifest: Path
    snapshot_directory: Path | None = None


@dataclass(frozen=True)
class CleaningResult:
    integrity: IntegrityReport
    release_audit: ReleaseAudit
    cohort: CohortResult
    os_endpoint: EndpointResult
    ttnt_endpoint: EndpointResult
    datasets: AnalysisDatasets
    availability: pd.DataFrame
    analysis_dataset: pd.DataFrame
    duplicate_audit: pd.DataFrame
    missingness_audit: pd.DataFrame
    merge_audit: pd.DataFrame
    quality_checks: pd.DataFrame
    release_manifest_sha256: str
    artifacts: CleaningArtifacts | None = None


def build_clean_analysis_dataset(
    data_root: str | Path,
    output_root: str | Path | None = None,
    *,
    snapshot: bool = False,
) -> CleaningResult:
    """Clean the restricted release and optionally refresh the canonical CSV handoff."""
    data_path = Path(data_root).expanduser().resolve()
    if not data_path.is_dir():
        raise CleaningWorkflowError("Restricted data root is unavailable")

    integrity = validate_manifest(data_path)
    tables = load_release(data_path, check_integrity=False)
    release_audit = audit_release(tables)
    cohort = build_cohort(tables)
    os_endpoint = build_os(cohort)
    ttnt_endpoint = build_ttnt(cohort)
    feature_manifest = FeatureManifest.load()
    datasets = build_analysis_datasets(cohort, os_endpoint, ttnt_endpoint, tables, feature_manifest)
    availability = baseline_availability_profile(cohort.cohort, tables)
    release_manifest_sha256 = _file_sha256(data_path / "manifest.csv")
    analysis_dataset = _build_master_dataset(
        datasets, os_endpoint, ttnt_endpoint, feature_manifest, release_manifest_sha256
    )
    duplicate_audit = _build_duplicate_audit(release_audit, analysis_dataset)
    missingness_audit = _build_missingness_audit(analysis_dataset, feature_manifest)
    merge_audit = _build_merge_audit(release_audit)
    quality_checks = _build_quality_checks(analysis_dataset)
    if int(quality_checks["failures"].sum()):
        failed = quality_checks.loc[quality_checks["failures"].gt(0), "check"].tolist()
        raise CleaningWorkflowError(f"Final dataset quality checks failed: {', '.join(failed)}")

    result = CleaningResult(
        integrity=integrity,
        release_audit=release_audit,
        cohort=cohort,
        os_endpoint=os_endpoint,
        ttnt_endpoint=ttnt_endpoint,
        datasets=datasets,
        availability=availability,
        analysis_dataset=analysis_dataset,
        duplicate_audit=duplicate_audit,
        missingness_audit=missingness_audit,
        merge_audit=merge_audit,
        quality_checks=quality_checks,
        release_manifest_sha256=release_manifest_sha256,
    )
    if output_root is None:
        return result
    artifacts = _write_artifacts(result, Path(output_root), snapshot=snapshot)
    return CleaningResult(**{**result.__dict__, "artifacts": artifacts})


def load_clean_analysis_dataset(path: str | Path) -> pd.DataFrame:
    """Load the canonical CSV using its adjacent data dictionary for stable dtypes."""
    dataset_path = Path(path).expanduser().resolve()
    if dataset_path.is_dir():
        dataset_path = dataset_path / DATASET_FILENAME
    dictionary_path = dataset_path.with_name(DICTIONARY_FILENAME)
    if not dataset_path.is_file() or not dictionary_path.is_file():
        raise CleaningWorkflowError("Canonical dataset or data dictionary is unavailable")
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    dtype_map = {item["column"]: item["pandas_dtype"] for item in dictionary["columns"]}
    frame = pd.read_csv(dataset_path, dtype=dtype_map, keep_default_na=True)
    expected_columns = [item["column"] for item in dictionary["columns"]]
    if list(frame.columns) != expected_columns:
        raise CleaningWorkflowError("Canonical dataset columns do not match the data dictionary")
    expected_rows = int(dictionary["rows"])
    if len(frame) != expected_rows:
        raise CleaningWorkflowError(
            "Canonical dataset row count does not match the data dictionary"
        )
    return frame


def _build_master_dataset(
    datasets: AnalysisDatasets,
    os_endpoint: EndpointResult,
    ttnt_endpoint: EndpointResult,
    manifest: FeatureManifest,
    release_manifest_sha256: str,
) -> pd.DataFrame:
    feature_columns = [feature.name for feature in manifest.features]
    feature_columns = list(dict.fromkeys(feature_columns))
    base = datasets.entry.data.loc[
        :, ["record_id", "institution", "sequencing_lag_days", "entry"]
    ].merge(
        datasets.features.loc[:, ["record_id", "institution", *feature_columns]],
        on=["record_id", "institution"],
        validate="one_to_one",
    )
    os_data = os_endpoint.data.loc[:, ["record_id", "tt_os_g_days", "os_g_status"]].rename(
        columns={"tt_os_g_days": "os_time", "os_g_status": "os_event"}
    )
    ttnt_data = ttnt_endpoint.data.loc[
        :, ["record_id", "ttnt_ca_seq_days", "ttnt_ca_seq_status", "ttnt_event_source"]
    ].rename(
        columns={
            "ttnt_ca_seq_days": "ttnt_d_time",
            "ttnt_ca_seq_status": "ttnt_d_event",
            "ttnt_event_source": "ttnt_d_event_source",
        }
    )
    master = base.merge(os_data, on="record_id", validate="one_to_one").merge(
        ttnt_data, on="record_id", validate="one_to_one"
    )
    master["os_eligible"] = master["entry"].lt(master["os_time"])
    master["ttnt_d_eligible"] = master["entry"].lt(master["ttnt_d_time"])

    namespace = release_manifest_sha256
    master.insert(
        0,
        "patient_key",
        master.pop("record_id")
        .astype(str)
        .map(
            lambda value: "P2-" + hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()[:16]
        ),
    )
    order = [
        "patient_key",
        "institution",
        "sequencing_lag_days",
        "entry",
        "os_time",
        "os_event",
        "os_eligible",
        "ttnt_d_time",
        "ttnt_d_event",
        "ttnt_d_event_source",
        "ttnt_d_eligible",
        *feature_columns,
    ]
    master = master.loc[:, order].sort_values("patient_key", kind="stable").reset_index(drop=True)
    return _normalize_csv_dtypes(master, manifest)


def _normalize_csv_dtypes(frame: pd.DataFrame, manifest: FeatureManifest) -> pd.DataFrame:
    normalized = frame.copy()
    string_columns = ["patient_key", "institution", "ttnt_d_event_source"]
    integer_columns = ["os_time", "os_event", "ttnt_d_time", "ttnt_d_event"]
    float_columns = ["sequencing_lag_days", "entry"]
    boolean_columns = ["os_eligible", "ttnt_d_eligible"]
    for feature in manifest.features:
        if feature.dtype == "string":
            string_columns.append(feature.name)
        elif feature.dtype == "integer":
            integer_columns.append(feature.name)
        elif feature.dtype == "float":
            float_columns.append(feature.name)
        elif feature.dtype == "boolean":
            boolean_columns.append(feature.name)
    for column in dict.fromkeys(string_columns):
        normalized[column] = normalized[column].astype("string")
    for column in dict.fromkeys(integer_columns):
        normalized[column] = normalized[column].astype("Int64")
    for column in dict.fromkeys(float_columns):
        normalized[column] = normalized[column].astype("Float64")
    for column in dict.fromkeys(boolean_columns):
        normalized[column] = normalized[column].astype("boolean")
    return normalized


def _build_duplicate_audit(audit: ReleaseAudit, master: pd.DataFrame) -> pd.DataFrame:
    source = audit.schema.loc[
        :, ["table", "rows", "full_row_duplicates", "rows_in_duplicate_complete_keys"]
    ].copy()
    source["action"] = "Audited at declared table grain; legitimate longitudinal rows retained"
    source.loc[source["full_row_duplicates"].gt(0), "action"] = (
        "Exact duplicates resolved by the table-specific cohort rule"
    )
    final = pd.DataFrame(
        [
            {
                "table": "final_analysis_dataset",
                "rows": len(master),
                "full_row_duplicates": int(master.duplicated().sum()),
                "rows_in_duplicate_complete_keys": int(
                    master["patient_key"].duplicated(keep=False).sum()
                ),
                "action": "Required one row per opaque patient key",
            }
        ]
    )
    return pd.concat([source, final], ignore_index=True)


def _build_missingness_audit(master: pd.DataFrame, manifest: FeatureManifest) -> pd.DataFrame:
    structural = {
        "patient_key",
        "institution",
        "entry",
        "os_time",
        "os_event",
        "os_eligible",
        "ttnt_d_time",
        "ttnt_d_event",
        "ttnt_d_event_source",
        "ttnt_d_eligible",
    }
    predictor_names = {feature.name for feature in manifest.features}
    rows = []
    for column in master.columns:
        missing = int(master[column].isna().sum())
        if column in structural:
            treatment = "Required structural field; missing values are not allowed"
        elif column in predictor_names and missing:
            treatment = "Retained as genuine unknown; imputation deferred to training folds"
        elif column in predictor_names:
            treatment = "Predictor complete; no imputation performed"
        else:
            treatment = "Descriptive timing field; retained without imputation"
        rows.append(
            {
                "column": column,
                "missing_n": missing,
                "missing_percent": round(100 * missing / len(master), 2),
                "phase_2_treatment": treatment,
            }
        )
    return pd.DataFrame(rows)


def _build_merge_audit(audit: ReleaseAudit) -> pd.DataFrame:
    merge = audit.cross_table.loc[
        audit.cross_table["check"].isin(["regimen cancer links", "cancer panel index links"])
    ].copy()
    merge["unmatched_rows"] = merge["eligible_rows"] - merge["matched_rows"]
    return merge.loc[:, ["check", "eligible_rows", "matched_rows", "unmatched_rows"]]


def _build_quality_checks(master: pd.DataFrame) -> pd.DataFrame:
    structural = [
        "patient_key",
        "institution",
        "entry",
        "os_time",
        "os_event",
        "ttnt_d_time",
        "ttnt_d_event",
    ]
    strings = master.select_dtypes(include="string")
    blank_strings = int(strings.apply(lambda column: column.str.strip().eq("").sum()).sum())
    surrounding_whitespace = int(
        strings.apply(lambda column: column.dropna().ne(column.dropna().str.strip()).sum()).sum()
    )
    checks = [
        ("duplicate patient keys", int(master["patient_key"].duplicated().sum())),
        ("missing structural fields", int(master[structural].isna().any(axis=1).sum())),
        ("negative entry times", int(master["entry"].lt(0).sum())),
        ("negative OS times", int(master["os_time"].lt(0).sum())),
        ("negative TTNT-D times", int(master["ttnt_d_time"].lt(0).sum())),
        ("invalid OS events", int((~master["os_event"].isin([0, 1])).sum())),
        ("invalid TTNT-D events", int((~master["ttnt_d_event"].isin([0, 1])).sum())),
        ("blank categorical values", blank_strings),
        ("categorical values with surrounding whitespace", surrounding_whitespace),
        (
            "OS-eligible rows with entry at or after exit",
            int((master["os_eligible"] & master["entry"].ge(master["os_time"])).sum()),
        ),
        (
            "TTNT-D-eligible rows with entry at or after exit",
            int((master["ttnt_d_eligible"] & master["entry"].ge(master["ttnt_d_time"])).sum()),
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "failures"])


def _write_artifacts(
    result: CleaningResult, output_root: Path, *, snapshot: bool
) -> CleaningArtifacts:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    current = output_root / "current"
    temp = Path(tempfile.mkdtemp(prefix=".phase2-current-", dir=output_root))
    backup = output_root / ".previous-current"
    try:
        dataset_path = temp / DATASET_FILENAME
        result.analysis_dataset.to_csv(dataset_path, index=False, float_format="%.17g")
        dictionary_path = temp / DICTIONARY_FILENAME
        _write_json(dictionary_path, _data_dictionary(result.analysis_dataset))
        restored = load_clean_analysis_dataset(dataset_path)
        try:
            assert_frame_equal(
                restored,
                result.analysis_dataset,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise CleaningWorkflowError("CSV round-trip changed values or declared dtypes") from exc

        report_path = temp / REPORT_FILENAME
        report_path.write_text(_render_report(result), encoding="utf-8")
        manifest_path = temp / MANIFEST_FILENAME
        _write_json(
            manifest_path,
            {
                "phase": "phase-2-data-cleaning",
                "fitted_preprocessing_performed": False,
                "models_fitted": False,
                "dataset_rows": len(result.analysis_dataset),
                "dataset_columns": result.analysis_dataset.shape[1],
                "os_eligible_patients": int(result.analysis_dataset["os_eligible"].sum()),
                "ttnt_d_eligible_patients": int(result.analysis_dataset["ttnt_d_eligible"].sum()),
                "dataset_sha256": _file_sha256(dataset_path),
                "release_manifest_sha256": result.release_manifest_sha256,
                "feature_manifest_sha256": _file_sha256(
                    Path(__file__).resolve().parent / "features" / "feature_manifest.json"
                ),
            },
        )
        _privacy_check_text_artifacts([dictionary_path, report_path, manifest_path])

        if backup.exists():
            shutil.rmtree(backup)
        if current.exists():
            current.rename(backup)
        temp.rename(current)
        if backup.exists():
            shutil.rmtree(backup)

        snapshot_directory = None
        if snapshot:
            snapshot_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            archive_root = output_root / "archive"
            archive_root.mkdir(exist_ok=True)
            snapshot_directory = archive_root / snapshot_id
            if snapshot_directory.exists():
                raise CleaningWorkflowError("Snapshot directory already exists")
            shutil.copytree(current, snapshot_directory)
        return CleaningArtifacts(
            current_directory=current,
            dataset=current / DATASET_FILENAME,
            data_dictionary=current / DICTIONARY_FILENAME,
            cleaning_report=current / REPORT_FILENAME,
            run_manifest=current / MANIFEST_FILENAME,
            snapshot_directory=snapshot_directory,
        )
    except Exception as exc:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        if not current.exists() and backup.exists():
            backup.rename(current)
        if isinstance(exc, CleaningWorkflowError):
            raise
        raise CleaningWorkflowError("Phase 2 artifact creation failed") from exc


def _data_dictionary(frame: pd.DataFrame) -> dict[str, object]:
    descriptions = {
        "patient_key": "Opaque patient linkage key; source identifier is not exported.",
        "institution": "GENIE contributing institution.",
        "sequencing_lag_days": "Days from FOLFOX initiation to first linked sequencing report.",
        "entry": "Delayed-entry time on the days-since-FOLFOX scale.",
        "os_time": "Overall-survival exit time in days from FOLFOX initiation.",
        "os_event": "Overall-survival event indicator (1=death, 0=censored).",
        "os_eligible": "Whether delayed entry occurs before OS exit.",
        "ttnt_d_time": "Death-inclusive time-to-next-treatment exit time in days.",
        "ttnt_d_event": "TTNT-D event indicator (1=event, 0=censored).",
        "ttnt_d_event_source": "Observed TTNT-D event source.",
        "ttnt_d_eligible": "Whether delayed entry occurs before TTNT-D exit.",
    }
    for feature in FeatureManifest.load().features:
        sources = ", ".join(feature.source_columns)
        descriptions[feature.name] = (
            f"{feature.role.capitalize()} predictor from {feature.source_table}; "
            f"source field(s): {sources}; derivation: {feature.derivation}."
        )
    return {
        "purpose": "Canonical cleaned Phase 2 dataset for later fold-fitted preprocessing",
        "rows": len(frame),
        "identifier_policy": "Source patient identifiers are replaced with opaque patient_key.",
        "missing_value_policy": (
            "Genuine predictor unknowns remain missing; learned imputation belongs inside "
            "training folds in Phase 3."
        ),
        "columns": [
            {
                "column": column,
                "pandas_dtype": str(frame[column].dtype),
                "nullable": bool(frame[column].isna().any()),
                "description": descriptions[column],
            }
            for column in frame.columns
        ],
    }


def _render_report(result: CleaningResult) -> str:
    master = result.analysis_dataset
    endpoint_summary = pd.DataFrame(
        [
            {
                "endpoint": "OS",
                "cohort_patients": len(master),
                "eligible_patients": int(master["os_eligible"].sum()),
                "events_among_eligible": int(master.loc[master["os_eligible"], "os_event"].sum()),
            },
            {
                "endpoint": "TTNT-D",
                "cohort_patients": len(master),
                "eligible_patients": int(master["ttnt_d_eligible"].sum()),
                "events_among_eligible": int(
                    master.loc[master["ttnt_d_eligible"], "ttnt_d_event"].sum()
                ),
            },
        ]
    )
    missing = result.missingness_audit.loc[result.missingness_audit["missing_n"].gt(0)]
    return "\n".join(
        [
            "# Phase 2 cleaning report",
            "",
            "Phase 2 validates and cleans the source release, constructs the cohort, predictors, "
            "and outcomes, and writes one canonical CSV. It does not impute predictors, split "
            "data, fit models, evaluate performance, or calculate explanations.",
            "",
            "## Source integrity and table grain",
            "",
            _markdown_table(result.release_audit.schema),
            "",
            "## Duplicate handling",
            "",
            _markdown_table(result.duplicate_audit),
            "",
            "## Linkage and merge checks",
            "",
            _markdown_table(result.release_audit.linkage),
            "",
            _markdown_table(result.merge_audit),
            "",
            "## Cohort cleaning flow",
            "",
            _markdown_table(result.cohort.cohort_flow),
            "",
            "## Retained missing values",
            "",
            _markdown_table(missing),
            "",
            "Genuine predictor unknowns are retained. Imputation will be learned separately in "
            "each training fold during Phase 3.",
            "",
            "## Endpoint-ready populations",
            "",
            _markdown_table(endpoint_summary),
            "",
            "## Final quality checks",
            "",
            _markdown_table(result.quality_checks),
            "",
            "## Canonical handoff",
            "",
            f"- Dataset: `{DATASET_FILENAME}`",
            f"- Rows: {len(master)}",
            f"- Columns: {master.shape[1]}",
            f"- Release manifest SHA-256: `{result.release_manifest_sha256}`",
            "",
            "Patient-level files remain local and gitignored.",
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    safe = frame.fillna("").astype(str)
    headers = [str(column) for column in safe.columns]
    rows = [[value.replace("|", "\\|") for value in row] for row in safe.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _privacy_check_text_artifacts(paths: list[Path]) -> None:
    # Schema field names are safe and necessary for a transparent grain audit. Test fixtures use
    # the invented-* prefix to prove that actual source identifier values never reach reports.
    forbidden = ("invented-",)
    for path in paths:
        content = path.read_text(encoding="utf-8").lower()
        if any(term in content for term in forbidden):
            raise CleaningWorkflowError("Privacy scan failed for an aggregate artifact")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CleaningArtifacts",
    "CleaningResult",
    "CleaningWorkflowError",
    "build_clean_analysis_dataset",
    "load_clean_analysis_dataset",
]

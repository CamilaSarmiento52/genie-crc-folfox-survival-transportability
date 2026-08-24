"""Canonical preprocessing workflow split and fold-fitted preprocessing workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from crc_folfox.features import FeatureManifest
from crc_folfox.pipeline import load_clean_analysis_dataset
from crc_folfox.preprocessing.contracts import (
    EndpointView,
    PreprocessingContractError,
    build_endpoint_view,
)
from crc_folfox.preprocessing.splits import (
    INNER_REPEATS,
    N_FOLDS,
    POOLED_REPEATS,
    ROOT_SEED,
    build_iecv_splits,
    build_inner_splits,
    build_pooled_splits,
)
from crc_folfox.preprocessing.transform import RARE_MIN_COUNT, FittedPreprocessor

REPORT_FILENAME = "preprocessing_report.md"
MANIFEST_FILENAME = "preprocessing_manifest.json"
FOLD_SUMMARY_FILENAME = "fold_summary.csv"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


class PreprocessingWorkflowError(RuntimeError):
    """Aggregate-safe failure raised by the preprocessing workflow."""


@dataclass(frozen=True)
class PreprocessingArtifacts:
    directory: Path
    preprocessing_report: Path
    preprocessing_manifest: Path
    fold_summary: Path
    feature_schema: Path


@dataclass(frozen=True)
class PreprocessingResult:
    endpoint_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    preprocessing_summary: pd.DataFrame
    feature_schema: dict[str, object]
    input_hashes: dict[str, str]
    artifacts: PreprocessingArtifacts | None = None


def run_preprocessing_workflow(
    phase2_directory: str | Path,
    output_directory: str | Path | None = None,
) -> PreprocessingResult:
    """Run preprocessing workflow without fitting models or persisting patient-level derivatives."""
    phase2 = Path(phase2_directory).expanduser().resolve()
    if not phase2.is_dir():
        raise PreprocessingWorkflowError("Canonical Phase 2 directory is unavailable")
    try:
        frame, phase2_manifest, input_hashes = _load_verified_inputs(phase2)
        feature_manifest = FeatureManifest.load()
        views = tuple(build_endpoint_view(frame, endpoint) for endpoint in ("OS", "TTNT-D"))
        _reconcile_endpoint_counts(views, phase2_manifest)
        endpoint_summary = _endpoint_summary(views)
        fold_rows: list[dict[str, object]] = []
        schema_entries: list[dict[str, object]] = []
        preprocessing_rows: list[dict[str, object]] = []

        for view in views:
            outer_splits = build_iecv_splits(view)
            for outer in outer_splits:
                fold_rows.append(outer.aggregate_row(view))
                _fit_split_feature_sets(
                    view,
                    outer,
                    feature_manifest,
                    schema_entries,
                    preprocessing_rows,
                )
                for inner in build_inner_splits(view, outer):
                    fold_rows.append(inner.aggregate_row(view))
                    _fit_split_feature_sets(
                        view,
                        inner,
                        feature_manifest,
                        schema_entries,
                        preprocessing_rows,
                    )

            for pooled in build_pooled_splits(view):
                fold_rows.append(pooled.aggregate_row(view))
                _fit_split_feature_sets(
                    view,
                    pooled,
                    feature_manifest,
                    schema_entries,
                    preprocessing_rows,
                )

        fold_summary = pd.DataFrame(fold_rows).sort_values(
            ["endpoint", "design", "held_out_institution", "repeat", "fold"],
            kind="stable",
        )
        fold_summary = fold_summary.reset_index(drop=True)
        preprocessing_summary = pd.DataFrame(preprocessing_rows).sort_values(
            ["endpoint", "design", "held_out_institution", "repeat", "fold", "feature_set"],
            kind="stable",
        )
        preprocessing_summary = preprocessing_summary.reset_index(drop=True)
        feature_schema: dict[str, object] = {
            "phase": "phase-3-fold-fitted-preprocessing",
            "feature_manifest_version": feature_manifest.version,
            "fit_states": schema_entries,
        }
        result = PreprocessingResult(
            endpoint_summary=endpoint_summary,
            fold_summary=fold_summary,
            preprocessing_summary=preprocessing_summary,
            feature_schema=feature_schema,
            input_hashes=input_hashes,
        )
        if output_directory is None:
            output_path = phase2 / "preprocessing"
        else:
            output_path = Path(output_directory).expanduser().resolve()
        artifacts = _write_artifacts(result, output_path, patient_keys=frame["patient_key"])
        _assert_inputs_unchanged(input_hashes)
        return PreprocessingResult(**{**result.__dict__, "artifacts": artifacts})
    except (PreprocessingWorkflowError, PreprocessingContractError):
        raise
    except Exception as exc:
        raise PreprocessingWorkflowError("Phase 3 preprocessing workflow failed") from exc


def _load_verified_inputs(
    phase2: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    dataset = phase2 / "analysis_dataset.csv"
    dictionary = phase2 / "data_dictionary.json"
    run_manifest = phase2 / "run_manifest.json"
    if not dataset.is_file() or not dictionary.is_file() or not run_manifest.is_file():
        raise PreprocessingWorkflowError("Canonical Phase 2 bundle is incomplete")
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if manifest.get("phase") != "phase-2-data-cleaning":
        raise PreprocessingWorkflowError("Input manifest is not the Phase 2 cleaning handoff")
    dataset_hash = _file_sha256(dataset)
    if dataset_hash != manifest.get("dataset_sha256"):
        raise PreprocessingWorkflowError("Phase 2 dataset hash does not match its manifest")
    feature_manifest_path = (
        Path(__file__).resolve().parents[1] / "features" / "feature_manifest.json"
    )
    feature_hash = _file_sha256(feature_manifest_path)
    if feature_hash != manifest.get("feature_manifest_sha256"):
        raise PreprocessingWorkflowError("Feature manifest hash differs from the Phase 2 handoff")
    frame = load_clean_analysis_dataset(dataset)
    if len(frame) != int(manifest.get("dataset_rows", -1)):
        raise PreprocessingWorkflowError("Phase 2 row total does not match its manifest")
    hashes = {
        str(dataset): dataset_hash,
        str(dictionary): _file_sha256(dictionary),
        str(run_manifest): _file_sha256(run_manifest),
        str(feature_manifest_path): feature_hash,
    }
    return frame, manifest, hashes


def _reconcile_endpoint_counts(
    views: tuple[EndpointView, ...], phase2_manifest: dict[str, object]
) -> None:
    expected_keys = {"OS": "os_eligible_patients", "TTNT-D": "ttnt_d_eligible_patients"}
    for view in views:
        expected = int(phase2_manifest.get(expected_keys[view.endpoint], -1))
        if len(view.data) != expected:
            raise PreprocessingWorkflowError("Endpoint population differs from Phase 2 manifest")


def _endpoint_summary(views: tuple[EndpointView, ...]) -> pd.DataFrame:
    rows = []
    for view in views:
        for institution, group in view.data.groupby("institution", sort=True):
            rows.append(
                {
                    "endpoint": view.endpoint,
                    "institution": str(institution),
                    "patients": len(group),
                    "events": int(group["event"].sum()),
                }
            )
        rows.append(
            {
                "endpoint": view.endpoint,
                "institution": "All",
                "patients": len(view.data),
                "events": int(view.data["event"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _fit_split_feature_sets(
    view: EndpointView,
    split,
    manifest: FeatureManifest,
    schema_entries: list[dict[str, object]],
    preprocessing_rows: list[dict[str, object]],
) -> None:
    development, assessment = split.frames(view)
    for feature_set in ("baseline", "core"):
        fitted = FittedPreprocessor.fit(
            development,
            endpoint=view.endpoint,
            feature_set=feature_set,
            fit_scope=split.split_id,
            manifest=manifest,
        )
        development_result = fitted.transform(development)
        assessment_result = fitted.transform(assessment)
        if fitted.fit_patient_keys != frozenset(development["patient_key"].astype(str)):
            raise PreprocessingWorkflowError("Fitted preprocessing rows differ from development")
        if fitted.fit_patient_keys & frozenset(assessment["patient_key"].astype(str)):
            raise PreprocessingWorkflowError("Assessment patient entered fitted preprocessing")
        if tuple(development_result.matrix.columns) != tuple(assessment_result.matrix.columns):
            raise PreprocessingWorkflowError("Development and assessment schemas differ")

        entry = fitted.aggregate_state()
        entry.update(
            {
                "design": split.design,
                "held_out_institution": split.held_out_institution,
                "repeat": split.repeat,
                "fold": split.fold,
                "assessment_patients": len(assessment),
                "assessment_unseen_level_counts": assessment_result.unseen_level_counts,
            }
        )
        schema_entries.append(entry)
        preprocessing_rows.append(
            {
                "endpoint": view.endpoint,
                "design": split.design,
                "split_id": split.split_id,
                "held_out_institution": split.held_out_institution or "",
                "repeat": split.repeat if split.repeat is not None else "",
                "fold": split.fold if split.fold is not None else "",
                "feature_set": feature_set,
                "training_patients": len(development),
                "assessment_patients": len(assessment),
                "input_features": len(fitted.feature_names),
                "output_columns": len(fitted.output_columns),
                "imputed_features": len(fitted.imputation_values),
                "pooled_source_levels": sum(
                    value == "__RARE__"
                    for mapping in fitted.level_maps.values()
                    for value in mapping.values()
                ),
                "unseen_assessment_values": sum(assessment_result.unseen_level_counts.values()),
                "removed_zero_variance": len(fitted.removed_zero_variance),
            }
        )


def _write_artifacts(
    result: PreprocessingResult,
    output_directory: Path,
    *,
    patient_keys: pd.Series,
) -> PreprocessingArtifacts:
    output = output_directory.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".phase3-preprocessing-", dir=output.parent))
    backup = output.parent / f".{output.name}-previous"
    try:
        fold_path = temp / FOLD_SUMMARY_FILENAME
        result.fold_summary.to_csv(fold_path, index=False)
        schema_path = temp / FEATURE_SCHEMA_FILENAME
        _write_json(schema_path, result.feature_schema)
        report_path = temp / REPORT_FILENAME
        report_path.write_text(_render_report(result), encoding="utf-8")

        manifest_payload: dict[str, object] = {
            "phase": "phase-3-fold-fitted-preprocessing",
            "models_fitted": False,
            "performance_evaluated": False,
            "patient_level_artifacts_written": False,
            "endpoints": ["OS", "TTNT-D"],
            "feature_sets": ["baseline", "core"],
            "root_seed": ROOT_SEED,
            "seed_algorithm": "sha256-first-8-hex-of-pipe-delimited-canonical-tuple",
            "inner_repeats": INNER_REPEATS,
            "pooled_repeats": POOLED_REPEATS,
            "folds": N_FOLDS,
            "rare_minimum_count": RARE_MIN_COUNT,
            "software_versions": {
                package: version(package) for package in ("numpy", "pandas", "scikit-learn")
            },
            "input_hashes": {
                Path(path).name: digest for path, digest in result.input_hashes.items()
            },
            "artifact_hashes": {
                FOLD_SUMMARY_FILENAME: _file_sha256(fold_path),
                FEATURE_SCHEMA_FILENAME: _file_sha256(schema_path),
                REPORT_FILENAME: _file_sha256(report_path),
            },
        }
        canonical = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"))
        manifest_payload["manifest_payload_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        manifest_path = temp / MANIFEST_FILENAME
        _write_json(manifest_path, manifest_payload)
        _privacy_check([fold_path, schema_path, report_path, manifest_path], patient_keys)

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        temp.rename(output)
        if backup.exists():
            shutil.rmtree(backup)
        return PreprocessingArtifacts(
            directory=output,
            preprocessing_report=output / REPORT_FILENAME,
            preprocessing_manifest=output / MANIFEST_FILENAME,
            fold_summary=output / FOLD_SUMMARY_FILENAME,
            feature_schema=output / FEATURE_SCHEMA_FILENAME,
        )
    except Exception as exc:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        if not output.exists() and backup.exists():
            backup.rename(output)
        if isinstance(exc, (PreprocessingWorkflowError, PreprocessingContractError)):
            raise
        raise PreprocessingWorkflowError("Phase 3 artifact creation failed") from exc


def _render_report(result: PreprocessingResult) -> str:
    fold_overview = (
        result.fold_summary.groupby(["endpoint", "design"], sort=True)
        .agg(
            partitions=("split_id", "count"),
            minimum_development_patients=("development_patients", "min"),
            minimum_assessment_patients=("assessment_patients", "min"),
        )
        .reset_index()
    )
    preprocessing_overview = (
        result.preprocessing_summary.groupby(["endpoint", "design", "feature_set"], sort=True)
        .agg(
            fits=("split_id", "count"),
            minimum_output_columns=("output_columns", "min"),
            maximum_output_columns=("output_columns", "max"),
            pooled_source_levels=("pooled_source_levels", "sum"),
            unseen_assessment_values=("unseen_assessment_values", "sum"),
            zero_variance_removals=("removed_zero_variance", "sum"),
        )
        .reset_index()
    )
    return "\n".join(
        [
            "# Phase 3 preprocessing report",
            "",
            "Phase 3 selects endpoint-eligible rows, creates IECV and paired pooled partitions, "
            "and fits every preprocessing operation on development rows only. No survival model "
            "is fitted and no performance is evaluated.",
            "",
            "## Verified endpoint populations",
            "",
            _markdown_table(result.endpoint_summary),
            "",
            "## Partition overview",
            "",
            _markdown_table(fold_overview),
            "",
            "## Fold-fitted preprocessing overview",
            "",
            _markdown_table(preprocessing_overview),
            "",
            "Detailed aggregate missingness, category support, imputation values, pooling maps, "
            "reference levels, scaling parameters, output schemas, unseen-level counts, and "
            f"filtering decisions are in `{FEATURE_SCHEMA_FILENAME}`.",
            "",
            "## Leakage and privacy boundary",
            "",
            "Assessment rows do not influence imputation, category pooling, encoding, scaling, "
            "or variance filtering. Split assignments and transformed patient matrices remain "
            "in memory and are not written. All persisted outputs are aggregate-only.",
            "",
            "## Phase 4 handoff",
            "",
            "Phase 4 may reconstruct these deterministic partitions and refit the same "
            "preprocessing recipes inside each model-development scope. Phase 3 produced no "
            "model, prediction, metric, explanation, or sensitivity result.",
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


def _privacy_check(paths: list[Path], patient_keys: pd.Series) -> None:
    contents = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    if any(str(key) in contents for key in patient_keys.astype(str)):
        raise PreprocessingWorkflowError("Privacy scan found a patient key in an artifact")


def _assert_inputs_unchanged(input_hashes: dict[str, str]) -> None:
    if any(_file_sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise PreprocessingWorkflowError("Phase 3 mutated a canonical input")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FEATURE_SCHEMA_FILENAME",
    "FOLD_SUMMARY_FILENAME",
    "MANIFEST_FILENAME",
    "REPORT_FILENAME",
    "PreprocessingArtifacts",
    "PreprocessingResult",
    "PreprocessingWorkflowError",
    "run_preprocessing_workflow",
]

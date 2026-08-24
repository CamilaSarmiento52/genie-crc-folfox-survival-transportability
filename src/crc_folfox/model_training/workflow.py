"""Atomic, assessment-blind model-training workflow (four required Cox candidates).

This orchestration boundary fits exactly `REQUIRED_MODEL_NAMES` across every IECV and pooled
development scope, publishes the result atomically with a tested rollback path, and supports
checkpointed resumption so an interrupted run only recomputes missing deterministic fit
identifiers. It records an already-decided forest/boosting admission outcome into the manifest
(`conditional_model_admissions`, see `run_model_training_workflow`) but never runs an admission
gate and never fits a conditional model itself -- that decision is made once, separately, by
`crc_folfox.model_training.admission`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.model_training.admission import AdmissionRecord
from crc_folfox.model_training.bundles import (
    RegistryRecord,
    extract_cox_bundle_state,
    predict_from_bundle_state,
    registry_record_from_state,
    save_bundle,
    verify_registry_load,
)
from crc_folfox.model_training.contracts import (
    ELASTIC_NET_GRID,
    REQUIRED_MODEL_NAMES,
    ModelScope,
    ModelTrainingContractError,
)
from crc_folfox.model_training.cox import apply_spline_state, fit_cox, fit_spline_state
from crc_folfox.model_training.selection import (
    ElasticNetSelection,
    ElasticNetTuningError,
    TuningDiagnostics,
    design_diagnostics,
    tune_elastic_net_cox,
)
from crc_folfox.preprocessing.contracts import EndpointView, build_endpoint_view
from crc_folfox.preprocessing.splits import (
    build_iecv_splits,
    build_inner_splits,
    build_pooled_splits,
)
from crc_folfox.preprocessing.transform import FittedPreprocessor
from crc_folfox.preprocessing.workflow import _load_verified_inputs

REPORT = "training_report.md"
MANIFEST = "training_manifest.json"
MODEL_TRAINING_WORKFLOW_IDENTIFIER = "phase-4-delayed-entry-model-training"
FIT_SUMMARY = "fit_summary.csv"
TUNING_SUMMARY = "tuning_summary.csv"
REGISTRY = "model_registry.json"

_WARNING_CLASSES = (
    "convergence",
    "ill_conditioned_or_singular",
    "extreme_coefficient_or_overflow",
    "low_variance",
    "adapter_or_curve",
    "unexpected",
)


class ModelTrainingWorkflowError(RuntimeError):
    """Aggregate-safe failure of the model-training workflow."""


@dataclass(frozen=True)
class ModelTrainingArtifacts:
    directory: Path
    training_report: Path
    training_manifest: Path
    fit_summary: Path
    tuning_summary: Path
    model_registry: Path


@dataclass(frozen=True)
class ModelTrainingResult:
    fit_summary: pd.DataFrame
    tuning_summary: pd.DataFrame
    artifacts: ModelTrainingArtifacts | None = None


def run_model_training_workflow(
    phase2_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    run_id: str | None = None,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    pooled_repeats: int | None = None,
    conditional_model_admissions: tuple[AdmissionRecord, ...] = (),
) -> ModelTrainingResult:
    """Fit and publish the required four-Cox inventory.

    `run_id` makes the run resumable: passing the same `run_id` twice reuses the same on-disk
    checkpoint and only (re)computes model identifiers missing from it, rather than starting over.
    Omit it (the default) for a normal one-shot run, which always starts from a fresh temporary
    directory. `endpoints`/`pooled_repeats` narrow the structural inventory for fast synthetic
    exercises of this boundary; the notebook must call this with both defaults left untouched.

    `conditional_model_admissions` records the already-decided, frozen forest/boosting admission
    outcome (`admission.decide_forest_admission`/`decide_boosting_admission`, run once in isolated
    clean runtimes per the spec's ordered gates) into the manifest. This function never runs an
    admission gate itself and never fits a conditional model -- passing an "admitted" record here
    does not add it to the fit loop; that wiring is out of scope until a gate actually passes.
    """
    phase2 = Path(phase2_directory).resolve()
    try:
        frame, _manifest, input_hashes = _load_verified_inputs(phase2)
        _verify_phase3(phase2 / "preprocessing", frame)
        temporary, resuming = _acquire_temporary_directory(phase2.parent, run_id)
        checkpoint = _checkpoint_path(phase2.parent, run_id) if run_id else None
        rows, registry = _load_checkpoint(checkpoint) if resuming else ([], {})
        completed_ids = {row["model_id"] for row in rows}
        tuning_rows = [row for row in rows if row.get("_is_tuning_row")]
        rows = [row for row in rows if not row.get("_is_tuning_row")]
        try:
            models = temporary / "models"
            models.mkdir(exist_ok=True)
            for endpoint in endpoints:
                view = build_endpoint_view(frame, endpoint)
                splits = (*build_iecv_splits(view), *_pooled_splits(view, pooled_repeats))
                for split in splits:
                    scope = ModelScope(split)
                    development, _ = split.frames(view)
                    for model_name in REQUIRED_MODEL_NAMES:
                        model_id = scope.model_id(model_name)
                        if model_id in completed_ids:
                            continue
                        row, tuning_row, record = _fit_required_model(
                            view, scope, development, model_name, models, input_hashes
                        )
                        rows.append(row)
                        registry[model_id] = record.to_json()
                        if tuning_row is not None:
                            tuning_rows.append(tuning_row)
                        if checkpoint is not None:
                            _append_checkpoint(checkpoint, row, tuning_row, record)
            fit_summary = (
                pd.DataFrame(rows).sort_values("model_id", kind="stable").reset_index(drop=True)
            )
            if tuning_rows:
                tuning_summary = (
                    pd.DataFrame(tuning_rows)
                    .sort_values("model_id", kind="stable")
                    .reset_index(drop=True)
                )
            else:
                tuning_summary = pd.DataFrame(columns=list(_TUNING_COLUMNS))
            output = (
                Path(output_directory).resolve() if output_directory else phase2 / "model_training"
            )
            artifacts = _publish(
                output,
                temporary,
                fit_summary,
                tuning_summary,
                registry,
                input_hashes,
                conditional_model_admissions,
            )
            _assert_unchanged(input_hashes)
            if checkpoint is not None:
                checkpoint.unlink(missing_ok=True)
            return ModelTrainingResult(fit_summary, tuning_summary, artifacts)
        except Exception:
            if run_id is None:
                shutil.rmtree(temporary, ignore_errors=True)
            raise
    except (ModelTrainingContractError, ModelTrainingWorkflowError):
        raise
    except Exception as exc:
        raise ModelTrainingWorkflowError("Phase 4 model training failed") from exc


def _pooled_splits(view: EndpointView, pooled_repeats: int | None):
    if pooled_repeats is None:
        return build_pooled_splits(view)
    return build_pooled_splits(view, repeats=pooled_repeats)


def _acquire_temporary_directory(parent: Path, run_id: str | None) -> tuple[Path, bool]:
    """Return the fit-staging directory and whether it (and its checkpoint) already existed.

    The checkpoint file itself lives outside this directory (see `_checkpoint_path`) so it can
    never be swept into the published output by `_publish`'s directory rename.
    """
    if run_id is None:
        return Path(tempfile.mkdtemp(prefix=".phase4-model-training-", dir=parent)), False
    directory = parent / f".phase4-model-training-{run_id}"
    resuming = directory.is_dir() and _checkpoint_path(parent, run_id).is_file()
    if not directory.is_dir():
        directory.mkdir(parents=True)
    return directory, resuming


def _checkpoint_path(parent: Path, run_id: str) -> Path:
    return parent / f".phase4-model-training-{run_id}.checkpoint.jsonl"


def _load_checkpoint(
    checkpoint: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    registry: dict[str, dict[str, object]] = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            rows.append(entry["row"])
            if entry.get("tuning_row") is not None:
                tuning_row = dict(entry["tuning_row"])
                tuning_row["_is_tuning_row"] = True
                rows.append(tuning_row)
            registry[entry["row"]["model_id"]] = entry["registry"]
    return rows, registry


def _append_checkpoint(
    checkpoint: Path,
    row: dict[str, object],
    tuning_row: dict[str, object] | None,
    record: RegistryRecord,
) -> None:
    entry = {"row": row, "tuning_row": tuning_row, "registry": record.to_json()}
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


_TUNING_COLUMNS = (
    "model_id",
    "model_name",
    "selected_penalizer",
    "selected_l1_ratio",
    "required_candidate_count",
    "eligible_candidate_count",
    "required_fold_count",
    "tie_break_used",
    "selection_status",
    # the correction workstream correction (correction-spec.md, "Failure diagnostics").
    "tuning_stage",
    "candidates_attempted",
    "candidates_failed",
    "candidates_nonfinite",
    "candidates_complete",
    "failure_exception_classes",
    "failure_penalizers",
    "failure_l1_ratios",
    "failure_inner_folds",
    "transformed_column_count",
    "zero_variance_column_count",
    "design_rank_deficit",
    "final_refit_failed",
)


def _classify_warning(message: str) -> str:
    lowered = message.lower()
    if "converg" in lowered:
        return "convergence"
    if "singular" in lowered or "ill-condition" in lowered or "collinear" in lowered:
        return "ill_conditioned_or_singular"
    if "overflow" in lowered or "delta" in lowered or "extreme" in lowered:
        return "extreme_coefficient_or_overflow"
    if "variance" in lowered or "constant" in lowered:
        return "low_variance"
    return "unexpected"


def _fit_required_model(
    view: EndpointView,
    scope: ModelScope,
    development: pd.DataFrame,
    model_name: str,
    models: Path,
    input_hashes: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object] | None, RegistryRecord]:
    feature_set = "baseline" if model_name == "minimal_cox" else "core"
    model_id = scope.model_id(model_name)
    split = scope.split
    development_patients = len(development)
    development_events = int(development["event"].sum())
    tuning_row: dict[str, object] | None = None
    warning_counts = dict.fromkeys(_WARNING_CLASSES, 0)
    failure_exception_class = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model_name == "elastic_net_cox":
                try:
                    selection = tune_elastic_net_cox(view, split)
                except ElasticNetTuningError as exc:
                    # The corrected per-candidate failure policy (correction-spec.md, "The
                    # corrected failure policy") means this branch now fires only for a
                    # scope-level tuning failure -- no eligible candidate, or the selected
                    # configuration's final refit failing -- never for one candidate's fit
                    # failing inside the grid loop, which `tune_elastic_net_cox` now isolates
                    # internally. `exc.diagnostics` carries the full failure breakdown.
                    tuning_row = _failed_tuning_row(model_id, exc.diagnostics)
                    raise
                except Exception:
                    # A structural failure before the candidate loop even starts (for example,
                    # an unsupported parent-split design) has no per-candidate diagnostics to
                    # report; still leave a `tuning_summary.csv` row for this parent scope
                    # (Block D, Step 10 precedent).
                    tuning_row = _failed_tuning_row(model_id, None)
                    raise
                tuning_row = _tuning_row(model_id, selection)
                fitter = selection.model.fitter
                preprocessor = selection.preprocessor
                penalizer, l1_ratio = selection.penalizer, selection.l1_ratio
                spline_state = None
                matrix = preprocessor.transform(development).matrix
            else:
                fitted = FittedPreprocessor.fit(
                    development,
                    endpoint=split.endpoint,
                    feature_set=feature_set,
                    fit_scope=model_id,
                )
                transformed = fitted.transform(development)
                matrix = transformed.matrix
                spline_state = None
                if model_name == "spline_cox":
                    spline_state = fit_spline_state(matrix)
                    matrix = apply_spline_state(matrix, spline_state)
                penalizer, l1_ratio = (
                    (0.0, 0.0) if model_name in {"minimal_cox", "core_cox"} else (0.01, 0.0)
                )
                fitter = fit_cox(
                    matrix, transformed.metadata, penalizer=penalizer, l1_ratio=l1_ratio
                )
                preprocessor = fitted
        for entry in caught:
            warning_counts[_classify_warning(str(entry.message))] += 1

        column_count, zero_variance_count, rank_deficit = design_diagnostics(matrix)
        cleared = replace(preprocessor, fit_patient_keys=frozenset())
        state = extract_cox_bundle_state(
            fitter,
            model_id=model_id,
            endpoint=split.endpoint,
            design=split.design,
            held_out_institution=split.held_out_institution,
            repeat=split.repeat,
            fold=split.fold,
            model_name=model_name,
            feature_set=feature_set,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            preprocessor=cleared,
            spline_state=spline_state,
            development_patient_count=development_patients,
            development_event_count=development_events,
            input_hashes=input_hashes,
        )
        relative_name = f"{_safe_name(model_id)}.joblib"
        bundle_path = models / relative_name
        save_bundle(state, bundle_path)
        digest = _sha256(bundle_path)
        record = registry_record_from_state(
            state, bundle_relative_path=relative_name, per_file_sha256=digest
        )
        status, reason = "fit_succeeded", ""
        convergence_status = (
            "converged_with_warning" if warning_counts["convergence"] else "converged"
        )
        output_dof = len(state.feature_columns)
        input_dof = len(cleared.aggregate_state()["input_features"])
        final_refit_failed = False
    except Exception as exc:
        record = RegistryRecord(
            model_id=model_id,
            fit_status="fit_failed",
            bundle_relative_path=None,
            per_file_sha256=None,
            canonical_state_hash=None,
            feature_schema_hash=None,
            format="joblib",
            runtime="python",
            endpoint=split.endpoint,
            development_scope=split.design,
            model_family=model_name,
        )
        digest = ""
        status, reason = "fit_failed", "numerical_or_adapter_failure"
        convergence_status = "not_converged"
        output_dof, input_dof = 0, 0
        column_count, zero_variance_count, rank_deficit = 0, 0, 0
        penalizer = l1_ratio = float("nan")
        failure_exception_class = type(exc).__name__
        final_refit_failed = True

    row = {
        "model_id": model_id,
        "endpoint": split.endpoint,
        "design": split.design,
        "held_out_institution": split.held_out_institution or "",
        "repeat": split.repeat if split.repeat is not None else "",
        "fold": split.fold if split.fold is not None else "",
        "model_name": model_name,
        "feature_set": feature_set,
        "development_patients": development_patients,
        "development_events": development_events,
        "input_degrees_of_freedom": input_dof,
        "output_degrees_of_freedom": output_dof,
        "selected_penalizer": penalizer,
        "selected_l1_ratio": l1_ratio,
        "convergence_status": convergence_status,
        "warning_counts": json.dumps(warning_counts, sort_keys=True),
        "fit_status": status,
        "reason": reason,
        "bundle_sha256": digest,
        # the correction workstream correction (correction-spec.md, "Failure diagnostics") -- added
        # to all four required candidates, not only elastic net, so a future `core_cox`-style
        # failure is diagnosable from the artifacts alone.
        "failure_exception_class": failure_exception_class,
        "transformed_column_count": column_count,
        "zero_variance_column_count": zero_variance_count,
        "design_rank_deficit": rank_deficit,
        "final_refit_failed": final_refit_failed,
    }
    return row, tuning_row, record


def _tuning_row(model_id: str, selection: ElasticNetSelection) -> dict[str, object]:
    diagnostics = selection.diagnostics
    return {
        "model_id": model_id,
        "model_name": "elastic_net_cox",
        "selected_penalizer": selection.penalizer,
        "selected_l1_ratio": selection.l1_ratio,
        "required_candidate_count": len(ELASTIC_NET_GRID),
        "eligible_candidate_count": len(selection.mean_scores),
        "required_fold_count": selection.inner_fold_count,
        "tie_break_used": selection.tie_break_used,
        "selection_status": "selected",
        **_diagnostics_fields(diagnostics),
    }


def _failed_tuning_row(model_id: str, diagnostics: TuningDiagnostics | None) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_name": "elastic_net_cox",
        "selected_penalizer": "",
        "selected_l1_ratio": "",
        "required_candidate_count": len(ELASTIC_NET_GRID),
        "eligible_candidate_count": 0,
        "required_fold_count": "",
        "tie_break_used": False,
        "selection_status": "fit_failed",
        **_diagnostics_fields(diagnostics),
    }


def _diagnostics_fields(diagnostics: TuningDiagnostics | None) -> dict[str, object]:
    if diagnostics is None:
        return {
            "tuning_stage": "",
            "candidates_attempted": 0,
            "candidates_failed": 0,
            "candidates_nonfinite": 0,
            "candidates_complete": 0,
            "failure_exception_classes": "",
            "failure_penalizers": "",
            "failure_l1_ratios": "",
            "failure_inner_folds": "",
            "transformed_column_count": 0,
            "zero_variance_column_count": 0,
            "design_rank_deficit": 0,
            "final_refit_failed": False,
        }
    return {
        "tuning_stage": diagnostics.tuning_stage,
        "candidates_attempted": diagnostics.candidates_attempted,
        "candidates_failed": diagnostics.candidates_failed,
        "candidates_nonfinite": diagnostics.candidates_nonfinite,
        "candidates_complete": diagnostics.candidates_complete,
        "failure_exception_classes": ";".join(diagnostics.failure_exception_classes),
        "failure_penalizers": ";".join(str(v) for v in diagnostics.failure_penalizers),
        "failure_l1_ratios": ";".join(str(v) for v in diagnostics.failure_l1_ratios),
        "failure_inner_folds": ";".join(str(v) for v in diagnostics.failure_inner_folds),
        "transformed_column_count": diagnostics.transformed_column_count,
        "zero_variance_column_count": diagnostics.zero_variance_column_count,
        "design_rank_deficit": diagnostics.design_rank_deficit,
        "final_refit_failed": diagnostics.final_refit_failed,
    }


def _verify_phase3(directory: Path, frame: pd.DataFrame) -> None:
    required = [
        "preprocessing_report.md",
        "preprocessing_manifest.json",
        "fold_summary.csv",
        "feature_schema.json",
    ]
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise ModelTrainingWorkflowError("Verified Phase 3 handoff is unavailable")
    manifest = json.loads((directory / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != "phase-3-fold-fitted-preprocessing":
        raise ModelTrainingWorkflowError("Phase 3 manifest has an unexpected phase identifier")
    for name, digest in manifest.get("artifact_hashes", {}).items():
        if _sha256(directory / name) != digest:
            raise ModelTrainingWorkflowError("Phase 3 artifact hash mismatch")
    _reconcile_phase3_splits(directory, frame)


def _reconcile_phase3_splits(directory: Path, frame: pd.DataFrame) -> None:
    fold_summary = pd.read_csv(directory / "fold_summary.csv")
    for endpoint in ("OS", "TTNT-D"):
        view = build_endpoint_view(frame, endpoint)
        expected_ids = {split.split_id for split in build_iecv_splits(view)}
        for outer in build_iecv_splits(view):
            expected_ids |= {split.split_id for split in build_inner_splits(view, outer)}
        expected_ids |= {split.split_id for split in build_pooled_splits(view)}
        recorded_ids = set(
            fold_summary.loc[fold_summary["endpoint"] == endpoint, "split_id"].astype(str)
        )
        if recorded_ids != expected_ids:
            raise ModelTrainingWorkflowError(
                "Rebuilt Phase 3 split identifiers do not reconcile with fold_summary.csv"
            )


def _publish(
    output: Path,
    temporary: Path,
    fit_summary: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    registry: dict[str, dict[str, object]],
    input_hashes: dict[str, str],
    conditional_model_admissions: tuple[AdmissionRecord, ...] = (),
) -> ModelTrainingArtifacts:
    fit_summary.to_csv(temporary / FIT_SUMMARY, index=False)
    tuning_summary.to_csv(temporary / TUNING_SUMMARY, index=False)
    (temporary / REGISTRY).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _render_report(fit_summary)
    (temporary / REPORT).write_text(report, encoding="utf-8")
    hashes = {
        name: _sha256(temporary / name) for name in (REPORT, FIT_SUMMARY, TUNING_SUMMARY, REGISTRY)
    }
    payload: dict[str, object] = {
        "phase": MODEL_TRAINING_WORKFLOW_IDENTIFIER,
        "input_hashes": {Path(path).name: digest for path, digest in input_hashes.items()},
        "software_versions": {"lifelines": version("lifelines"), "joblib": version("joblib")},
        "artifact_hashes": hashes,
        "conditional_model_admissions": {
            record.candidate: record.to_json() for record in conditional_model_admissions
        },
        "required_fit_count": int(len(fit_summary)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (temporary / MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.parent / f".{output.name}-previous"
    try:
        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        temporary.rename(output)
    except Exception as exc:
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise ModelTrainingWorkflowError("Phase 4 atomic publication failed") from exc

    try:
        _verify_published_bundles(output, registry)
    except Exception as exc:
        rolled_back = output.parent / f".{output.name}-rolledback"
        if rolled_back.exists():
            shutil.rmtree(rolled_back)
        output.rename(rolled_back)
        if backup.exists():
            backup.rename(output)
        shutil.rmtree(rolled_back, ignore_errors=True)
        raise ModelTrainingWorkflowError("Phase 4 post-publication verification failed") from exc

    if backup.exists():
        shutil.rmtree(backup)
    return ModelTrainingArtifacts(
        output,
        output / REPORT,
        output / MANIFEST,
        output / FIT_SUMMARY,
        output / TUNING_SUMMARY,
        output / REGISTRY,
    )


def _verify_published_bundles(output: Path, registry: dict[str, dict[str, object]]) -> None:
    """Reload every successful bundle from the just-published directory and re-verify it."""
    models_directory = output / "models"
    for payload in registry.values():
        if payload.get("fit_status") != "fit_succeeded":
            continue
        record = RegistryRecord(**payload)
        state = verify_registry_load(models_directory, record)
        matrix = pd.DataFrame([state.norm_mean], columns=list(state.feature_columns))
        prediction = predict_from_bundle_state(state, matrix)
        if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
            raise ModelTrainingWorkflowError("Post-publication synthetic prediction check failed")


def _render_report(fit_summary: pd.DataFrame) -> str:
    succeeded = int((fit_summary["fit_status"] == "fit_succeeded").sum()) if len(fit_summary) else 0
    failed = int((fit_summary["fit_status"] == "fit_failed").sum()) if len(fit_summary) else 0
    return (
        "# Phase 4 model-training report\n\n"
        "Training artifacts are aggregate-safe. No assessment predictions or performance metrics "
        "were generated.\n\n"
        f"## Fit inventory\n\nRequired fits recorded: {len(fit_summary)} "
        f"(succeeded: {succeeded}, failed: {failed}).\n\n"
        "Forest and boosting admission status is recorded separately once decided (Step 7); this "
        "run covers the four required Cox candidates only.\n\n"
        "## Phase 5 handoff\n\nPhase 5 must reconstruct untouched assessment rows from verified "
        "inputs; no assessment prediction or performance metric was generated here.\n"
    )


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_unchanged(hashes: dict[str, str]) -> None:
    if any(_sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ModelTrainingWorkflowError("Phase 4 mutated a canonical input")

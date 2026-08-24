"""The exploratory-optimization workstream orchestration: fit B2-B6 across every development scope.

Builds the 56 development scopes (53 OS -- 3 IECV outer + 50 pooled -- and 3 TTNT-D IECV-outer-
only, per optimization-spec.md's "Designs and endpoint hierarchy"), tunes and fits the five new
candidates for each, and publishes two registries: the unchanged model-training workflow
`model_registry.json`
format for B4/B5 (plain `lifelines.CoxPHFitter` fits, reusing `extract_cox_bundle_state`/
`save_bundle`/`registry_record_from_state` unchanged), and the separate `r_model_registry.json`
format (`crc_folfox.optimize.r_registry`) for the R-backed B2/B3/B6. B1 (`minimal_cox`) is not fit
here -- Workstream B loads it from the frozen model-training registry at evaluation time.

Deliberately simpler than `crc_folfox.model_training.workflow`: no checkpointed resumption and no
atomic rollback-on-failure publish. Both are engineering conveniences, not correctness
requirements, and are left as a documented simplification (see optimization-spec.md's "Workstream
B tests" gap) rather than reimplemented under this workflow's time budget.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import pandas as pd

from crc_folfox.features import FeatureManifest
from crc_folfox.model_training.bundles import (
    RegistryRecord,
    extract_cox_bundle_state,
    registry_record_from_state,
    save_bundle,
)
from crc_folfox.model_training.contracts import HORIZONS
from crc_folfox.model_training.workflow import _verify_phase3
from crc_folfox.optimize.candidate_selection import (
    CandidateTuningError,
    tune_flexsurv_cox,
    tune_penalized_nonlinear_cox,
    tune_reduced_core_cox,
)
from crc_folfox.optimize.contracts import ANALYSIS_STATUS, Phase5AContractError
from crc_folfox.optimize.glmnet_selection import (
    GlmnetTuningError,
    tune_glmnet_cox,
    tune_glmnet_enet_cox,
)
from crc_folfox.optimize.r_adapter import FlexsurvAdapter, GlmnetAdapter
from crc_folfox.optimize.r_registry import (
    RRegistryRecord,
    r_registry_record_failed,
    r_registry_record_succeeded,
)
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits, build_pooled_splits
from crc_folfox.preprocessing.workflow import _load_verified_inputs

FIT_SUMMARY = "fit_summary.csv"
MANIFEST = "optimization_development_manifest.json"
MODEL_REGISTRY = "model_registry.json"
R_MODEL_REGISTRY = "r_model_registry.json"
OPTIMIZATION_DIRNAME = "phase5a/optimization"

R_CANDIDATE_NAMES = ("glmnet_ridge_core", "glmnet_enet_core", "flexsurv_rp_spline")
PYTHON_CANDIDATE_NAMES = ("penalized_nonlinear_cox", "reduced_core_penalized_cox")
ALL_NEW_CANDIDATES = (
    "glmnet_ridge_core",
    "glmnet_enet_core",
    "penalized_nonlinear_cox",
    "reduced_core_penalized_cox",
    "flexsurv_rp_spline",
)


class WorkstreamBWorkflowError(RuntimeError):
    """Aggregate-safe failure of the exploratory-optimization workstream development workflow."""


@dataclass(frozen=True)
class WorkstreamBDevelopmentArtifacts:
    directory: Path
    fit_summary: Path
    manifest: Path
    model_registry: Path
    r_model_registry: Path


@dataclass(frozen=True)
class WorkstreamBDevelopmentResult:
    fit_summary: pd.DataFrame
    artifacts: WorkstreamBDevelopmentArtifacts | None = None


def _model_id(split: SplitDefinition, candidate: str) -> str:
    parts = [f"endpoint={split.endpoint}", f"design={split.design}"]
    if split.held_out_institution:
        parts.append(f"holdout={split.held_out_institution}")
    if split.repeat is not None:
        parts.append(f"repeat={split.repeat}")
    if split.fold is not None:
        parts.append(f"fold={split.fold}")
    parts.append(f"model={candidate}")
    return "|".join(parts)


def _safe_name(model_id: str) -> str:
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_preprocessor_sidecar(preprocessor, models_dir: Path, model_id: str) -> tuple[str, str]:
    """Persist an R candidate's `FittedPreprocessor` alongside its `.rds` bundle.

    An R bundle has no imputation/encoding/scaling state of its own, unlike a Python
    `CoxBundleState`, which embeds its preprocessor -- evaluation needs this sidecar to transform
    new assessment rows into the matrix the R model expects. Patient keys are cleared first,
    exactly as model-training workflow clears them before saving any bundle.
    """
    cleared = replace(preprocessor, fit_patient_keys=frozenset())
    relative_name = f"{_safe_name(model_id)}.preprocessor.joblib"
    destination = models_dir / relative_name
    joblib.dump(cleared, destination)
    return relative_name, _sha256(destination)


def development_scopes(
    view: EndpointView, pooled_repeats: int | None = None
) -> tuple[SplitDefinition, ...]:
    """The development scopes for one endpoint: 3 IECV outer + 50 pooled for OS, 3 IECV outer
    only for TTNT-D (the pooled comparator is prospectively omitted for the supportive endpoint;
    see optimization-spec.md, "Designs and endpoint hierarchy")."""
    iecv = build_iecv_splits(view)
    if view.endpoint == "TTNT-D":
        return iecv
    pooled = (
        build_pooled_splits(view)
        if pooled_repeats is None
        else build_pooled_splits(view, repeats=pooled_repeats)
    )
    return (*iecv, *pooled)


def _fit_row(
    *,
    model_id: str,
    split: SplitDefinition,
    candidate: str,
    feature_set: str,
    development_patients: int,
    development_events: int,
    status: str,
    reason: str,
    config: dict[str, float],
    diagnostics_dict: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "endpoint": split.endpoint,
        "design": split.design,
        "held_out_institution": split.held_out_institution or "",
        "repeat": split.repeat if split.repeat is not None else "",
        "fold": split.fold if split.fold is not None else "",
        "candidate": candidate,
        "feature_set": feature_set,
        "development_patients": development_patients,
        "development_events": development_events,
        "fit_status": status,
        "reason": reason,
        "config": json.dumps(config, sort_keys=True),
        "tuning_stage": (diagnostics_dict or {}).get("tuning_stage", ""),
        "candidates_attempted": (diagnostics_dict or {}).get("candidates_attempted", 0),
        "candidates_failed": (diagnostics_dict or {}).get("candidates_failed", 0),
        "candidates_complete": (diagnostics_dict or {}).get("candidates_complete", 0),
    }


def _diag_dict(diagnostics: object) -> dict[str, object]:
    return {
        "tuning_stage": diagnostics.tuning_stage,
        "candidates_attempted": diagnostics.candidates_attempted,
        "candidates_failed": diagnostics.candidates_failed,
        "candidates_complete": diagnostics.candidates_complete,
    }


def _fit_glmnet_family(
    view: EndpointView,
    split: SplitDefinition,
    candidate: str,
    *,
    models_dir: Path,
    tmp_root: Path,
    manifest: FeatureManifest | None,
    adapter: GlmnetAdapter,
) -> tuple[dict[str, object], RRegistryRecord]:
    development, _ = split.frames(view)
    n_patients, n_events = len(development), int(development["event"].sum())
    model_id = _model_id(split, candidate)
    bundles_dir = tmp_root / _safe_name(model_id)
    try:
        if candidate == "glmnet_ridge_core":
            selection = tune_glmnet_cox(
                view, split, alpha=0.0, bundles_dir=bundles_dir, manifest=manifest, adapter=adapter
            )
        else:
            selection = tune_glmnet_enet_cox(
                view, split, bundles_dir=bundles_dir, manifest=manifest, adapter=adapter
            )
    except GlmnetTuningError as exc:
        shutil.rmtree(bundles_dir, ignore_errors=True)
        record = r_registry_record_failed(
            model_id=model_id,
            candidate=candidate,
            endpoint=split.endpoint,
            design=split.design,
            held_out_institution=split.held_out_institution,
            repeat=split.repeat,
            fold=split.fold,
            feature_set="core",
            development_patient_count=n_patients,
            development_event_count=n_events,
        )
        row = _fit_row(
            model_id=model_id,
            split=split,
            candidate=candidate,
            feature_set="core",
            development_patients=n_patients,
            development_events=n_events,
            status="fit_failed",
            reason=str(exc),
            config={},
            diagnostics_dict=_diag_dict(exc.diagnostics),
        )
        return row, record

    config = {"alpha": selection.alpha, "lambda": selection.lambda_value}
    relative_name = f"{_safe_name(model_id)}.rds"
    destination = models_dir / relative_name
    shutil.copy2(selection.result.bundle, destination)
    preprocessor_relative_name, preprocessor_digest = _save_preprocessor_sidecar(
        selection.preprocessor, models_dir, model_id
    )
    shutil.rmtree(bundles_dir, ignore_errors=True)
    digest = _sha256(destination)
    record = r_registry_record_succeeded(
        model_id=model_id,
        candidate=candidate,
        bundle_relative_path=relative_name,
        per_file_sha256=digest,
        preprocessor_relative_path=preprocessor_relative_name,
        preprocessor_sha256=preprocessor_digest,
        endpoint=split.endpoint,
        design=split.design,
        held_out_institution=split.held_out_institution,
        repeat=split.repeat,
        fold=split.fold,
        feature_set="core",
        config=config,
        development_patient_count=n_patients,
        development_event_count=n_events,
    )
    row = _fit_row(
        model_id=model_id,
        split=split,
        candidate=candidate,
        feature_set="core",
        development_patients=n_patients,
        development_events=n_events,
        status="fit_succeeded",
        reason="",
        config=config,
        diagnostics_dict=_diag_dict(selection.diagnostics),
    )
    return row, record


def _fit_flexsurv(
    view: EndpointView,
    split: SplitDefinition,
    *,
    models_dir: Path,
    tmp_root: Path,
    manifest: FeatureManifest | None,
    adapter: FlexsurvAdapter,
) -> tuple[dict[str, object], RRegistryRecord]:
    development, _ = split.frames(view)
    n_patients, n_events = len(development), int(development["event"].sum())
    candidate = "flexsurv_rp_spline"
    model_id = _model_id(split, candidate)
    bundles_dir = tmp_root / _safe_name(model_id)
    try:
        selection = tune_flexsurv_cox(
            view, split, bundles_dir=bundles_dir, manifest=manifest, adapter=adapter
        )
    except CandidateTuningError as exc:
        shutil.rmtree(bundles_dir, ignore_errors=True)
        record = r_registry_record_failed(
            model_id=model_id,
            candidate=candidate,
            endpoint=split.endpoint,
            design=split.design,
            held_out_institution=split.held_out_institution,
            repeat=split.repeat,
            fold=split.fold,
            feature_set="baseline",
            development_patient_count=n_patients,
            development_event_count=n_events,
        )
        row = _fit_row(
            model_id=model_id,
            split=split,
            candidate=candidate,
            feature_set="baseline",
            development_patients=n_patients,
            development_events=n_events,
            status="fit_failed",
            reason=str(exc),
            config={},
            diagnostics_dict=_diag_dict(exc.diagnostics),
        )
        return row, record

    config = {"k": float(selection.k)}
    relative_name = f"{_safe_name(model_id)}.rds"
    destination = models_dir / relative_name
    shutil.copy2(selection.result.bundle, destination)
    preprocessor_relative_name, preprocessor_digest = _save_preprocessor_sidecar(
        selection.preprocessor, models_dir, model_id
    )
    shutil.rmtree(bundles_dir, ignore_errors=True)
    digest = _sha256(destination)
    record = r_registry_record_succeeded(
        model_id=model_id,
        candidate=candidate,
        bundle_relative_path=relative_name,
        per_file_sha256=digest,
        preprocessor_relative_path=preprocessor_relative_name,
        preprocessor_sha256=preprocessor_digest,
        endpoint=split.endpoint,
        design=split.design,
        held_out_institution=split.held_out_institution,
        repeat=split.repeat,
        fold=split.fold,
        feature_set="baseline",
        config=config,
        development_patient_count=n_patients,
        development_event_count=n_events,
    )
    row = _fit_row(
        model_id=model_id,
        split=split,
        candidate=candidate,
        feature_set="baseline",
        development_patients=n_patients,
        development_events=n_events,
        status="fit_succeeded",
        reason="",
        config=config,
        diagnostics_dict=_diag_dict(selection.diagnostics),
    )
    return row, record


def _fit_python_cox_family(
    view: EndpointView,
    split: SplitDefinition,
    candidate: str,
    *,
    models_dir: Path,
    manifest: FeatureManifest | None,
    input_hashes: dict[str, str],
) -> tuple[dict[str, object], RegistryRecord]:
    development, _ = split.frames(view)
    n_patients, n_events = len(development), int(development["event"].sum())
    model_id = _model_id(split, candidate)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if candidate == "penalized_nonlinear_cox":
                selection = tune_penalized_nonlinear_cox(view, split, manifest=manifest)
                spline_state = selection.spline_state
                penalizer = selection.penalizer
                config = {"penalizer": penalizer}
            else:
                selection = tune_reduced_core_cox(view, split, manifest=manifest)
                spline_state = None
                penalizer = selection.penalizer
                config = {"penalizer": penalizer}
        feature_set = "core"
        cleared_preprocessor = replace(selection.preprocessor, fit_patient_keys=frozenset())
        state = extract_cox_bundle_state(
            selection.fitter,
            model_id=model_id,
            endpoint=split.endpoint,
            design=split.design,
            held_out_institution=split.held_out_institution,
            repeat=split.repeat,
            fold=split.fold,
            model_name=candidate,
            feature_set=feature_set,
            penalizer=penalizer,
            l1_ratio=0.0,
            preprocessor=cleared_preprocessor,
            spline_state=spline_state,
            development_patient_count=n_patients,
            development_event_count=n_events,
            input_hashes=input_hashes,
        )
    except CandidateTuningError as exc:
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
            model_family=candidate,
        )
        row = _fit_row(
            model_id=model_id,
            split=split,
            candidate=candidate,
            feature_set="core",
            development_patients=n_patients,
            development_events=n_events,
            status="fit_failed",
            reason=str(exc),
            config={},
            diagnostics_dict=_diag_dict(exc.diagnostics),
        )
        return row, record

    relative_name = f"{_safe_name(model_id)}.joblib"
    destination = models_dir / relative_name
    save_bundle(state, destination)
    digest = _sha256(destination)
    record = registry_record_from_state(
        state, bundle_relative_path=relative_name, per_file_sha256=digest
    )
    row = _fit_row(
        model_id=model_id,
        split=split,
        candidate=candidate,
        feature_set="core",
        development_patients=n_patients,
        development_events=n_events,
        status="fit_succeeded",
        reason="",
        config=config,
        diagnostics_dict=_diag_dict(selection.diagnostics),
    )
    return row, record


def run_workstream_b_development(
    phase2_directory: str | Path,
    output_directory: str | Path,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    pooled_repeats: int | None = None,
    manifest: FeatureManifest | None = None,
) -> WorkstreamBDevelopmentResult:
    """Fit B2-B6 across every development scope and publish two registries.

    `pooled_repeats` narrows the OS pooled inventory for fast synthetic exercises of this
    boundary; the real run must call this with it left at the default.
    """
    phase2 = Path(phase2_directory).resolve()
    output = Path(output_directory).resolve()
    try:
        frame, _phase2_manifest, input_hashes = _load_verified_inputs(phase2)
        _verify_phase3(phase2 / "preprocessing", frame)

        models_dir = output / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = output / ".tmp_bundles"
        tmp_root.mkdir(parents=True, exist_ok=True)

        glmnet_adapter = GlmnetAdapter()
        flexsurv_adapter = FlexsurvAdapter()

        rows: list[dict[str, object]] = []
        python_registry: dict[str, dict[str, object]] = {}
        r_registry: dict[str, dict[str, object]] = {}

        for endpoint in endpoints:
            view = build_endpoint_view(frame, endpoint)
            for split in development_scopes(view, pooled_repeats):
                for candidate in ("glmnet_ridge_core", "glmnet_enet_core"):
                    row, record = _fit_glmnet_family(
                        view,
                        split,
                        candidate,
                        models_dir=models_dir,
                        tmp_root=tmp_root,
                        manifest=manifest,
                        adapter=glmnet_adapter,
                    )
                    rows.append(row)
                    r_registry[record.model_id] = record.to_json()

                row, record = _fit_flexsurv(
                    view,
                    split,
                    models_dir=models_dir,
                    tmp_root=tmp_root,
                    manifest=manifest,
                    adapter=flexsurv_adapter,
                )
                rows.append(row)
                r_registry[record.model_id] = record.to_json()

                for candidate in PYTHON_CANDIDATE_NAMES:
                    row, record = _fit_python_cox_family(
                        view,
                        split,
                        candidate,
                        models_dir=models_dir,
                        manifest=manifest,
                        input_hashes=input_hashes,
                    )
                    rows.append(row)
                    python_registry[record.model_id] = record.to_json()

        shutil.rmtree(tmp_root, ignore_errors=True)

        fit_summary = (
            pd.DataFrame(rows).sort_values("model_id", kind="stable").reset_index(drop=True)
        )
        fit_summary["analysis_status"] = ANALYSIS_STATUS
        fit_summary_path = output / FIT_SUMMARY
        fit_summary.to_csv(fit_summary_path, index=False)
        model_registry_path = output / MODEL_REGISTRY
        model_registry_path.write_text(
            json.dumps(python_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        r_model_registry_path = output / R_MODEL_REGISTRY
        r_model_registry_path.write_text(
            json.dumps(r_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload = {
            "phase": "phase-5a-workstream-b-development",
            "input_hashes": {Path(path).name: digest for path, digest in input_hashes.items()},
            "required_fit_count": int(len(fit_summary)),
            "horizons": list(HORIZONS),
        }
        manifest_path = output / MANIFEST
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        artifacts = WorkstreamBDevelopmentArtifacts(
            output, fit_summary_path, manifest_path, model_registry_path, r_model_registry_path
        )
        return WorkstreamBDevelopmentResult(fit_summary, artifacts)
    except Phase5AContractError:
        raise
    except Exception as exc:
        raise WorkstreamBWorkflowError("Phase 5A Workstream B development failed") from exc

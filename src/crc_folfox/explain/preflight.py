"""Explainability preflight: verify every consumed input before any explanation value is computed.

The explainability workflow never regenerates, repairs, or overwrites an input; a failure here
stops the analysis rather than proceeding on a drifted basis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crc_folfox.explain.contracts import ExplainContractError
from crc_folfox.model_training.bundles import RegistryRecord as CoxRegistryRecord
from crc_folfox.model_training.bundles import canonical_state_hash, verify_registry_load
from crc_folfox.model_training.workflow import MODEL_TRAINING_WORKFLOW_IDENTIFIER, _verify_phase3
from crc_folfox.optimize.archive import verify_frozen_primary_unchanged
from crc_folfox.optimize.contracts import OptimizationContractError
from crc_folfox.optimize.correction import CORRECTION_DIRNAME
from crc_folfox.optimize.workflow import OPTIMIZATION_DIRNAME
from crc_folfox.preprocessing.workflow import _load_verified_inputs

CORRECTION_SUBDIR = CORRECTION_DIRNAME
OPTIMIZATION_SUBDIR = OPTIMIZATION_DIRNAME

# The two models loaded from both the frozen model-training registry and the corrected
# registry: their canonical state hashes must be identical, since Workstream A's correction touches
# only `elastic_net_cox`'s failure policy. See `explainability-spec.md`, Tier 2.
_UNTOUCHED_BY_CORRECTION = ("minimal_cox", "spline_cox")


class ExplainabilityPreflightError(ExplainContractError):
    """Raised when an input cannot be verified before any explanation is computed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_model_training_directory(
    directory: Path, expected_workflow_identifier: str
) -> dict[str, dict]:
    """Verify a model-training directory (either the frozen original or corrected re-run):
    required files present, manifest artifact hashes
    match, and every `fit_succeeded` bundle's per-file hash matches its registry entry."""
    required = ["training_manifest.json", "fit_summary.csv", "model_registry.json"]
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise ExplainabilityPreflightError(f"Verified input directory is unavailable: {directory}")
    manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != expected_workflow_identifier:
        raise ExplainabilityPreflightError(
            f"Unexpected model-training workflow identifier in manifest: {directory}"
        )
    for name, digest in manifest.get("artifact_hashes", {}).items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            raise ExplainabilityPreflightError(f"Input artifact hash mismatch: {path}")
    registry = json.loads((directory / "model_registry.json").read_text(encoding="utf-8"))
    for record in registry.values():
        if record.get("fit_status") == "fit_succeeded":
            bundle_path = directory / "models" / str(record["bundle_relative_path"])
            if not bundle_path.is_file() or _sha256(bundle_path) != record["per_file_sha256"]:
                raise ExplainabilityPreflightError(
                    f"Registered bundle hash mismatch: {bundle_path}"
                )
    return registry


def _verify_optimization_directory(directory: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Verify the Workstream B directory exists with both registries. Per-bundle hashes are
    verified lazily at load time via `verify_registry_load`/`verify_r_registry_load`, matching this
    directory's own established verification pattern (no manifest-level artifact-hash list is
    published for it, unlike the model-training directories)."""
    required = ["fit_summary.csv", "model_registry.json", "r_model_registry.json"]
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise ExplainabilityPreflightError(
            f"Verified Workstream B directory is unavailable: {directory}"
        )
    python_registry = json.loads((directory / "model_registry.json").read_text(encoding="utf-8"))
    r_registry = json.loads((directory / "r_model_registry.json").read_text(encoding="utf-8"))
    return python_registry, r_registry


def _assert_corrected_matches_frozen(
    frozen_registry: dict,
    frozen_models_dir: Path,
    corrected_registry: dict,
    corrected_models_dir: Path,
) -> None:
    """Preflight stop condition: `minimal_cox`/`spline_cox` canonical state hashes must be
    identical between the frozen model-training registry and the corrected registry, since
    the correction touches only elastic-net's failure policy. This is what makes Tier 1/Tier 2's
    mixed provenance (some models from the frozen bundle, `elastic_net_cox` from the corrected one)
    safe -- a mismatch means the two bundles are not the byte-for-byte-comparable basis the
    explainability workflow assumes. The analysis must stop rather than explain a drifted
    model."""
    checked = 0
    for model_id, frozen_record in frozen_registry.items():
        if frozen_record.get("fit_status") != "fit_succeeded":
            continue
        if not any(
            f"model={name}" == model_id.rsplit("|", 1)[-1] for name in _UNTOUCHED_BY_CORRECTION
        ):
            continue
        corrected_record = corrected_registry.get(model_id)
        if corrected_record is None or corrected_record.get("fit_status") != "fit_succeeded":
            raise ExplainabilityPreflightError(
                f"'{model_id}' is fit_succeeded in the frozen registry but not in the corrected one"
            )
        frozen_state = verify_registry_load(frozen_models_dir, CoxRegistryRecord(**frozen_record))
        corrected_state = verify_registry_load(
            corrected_models_dir, CoxRegistryRecord(**corrected_record)
        )
        if canonical_state_hash(frozen_state) != canonical_state_hash(corrected_state):
            raise ExplainabilityPreflightError(
                f"Corrected-vs-frozen hash mismatch for an untouched model: '{model_id}'"
            )
        checked += 1
    if checked == 0:
        raise ExplainabilityPreflightError(
            "No untouched-model scope was available to prove corrected-vs-frozen hash equality"
        )


@dataclass(frozen=True)
class VerifiedInputs:
    frame: pd.DataFrame
    input_hashes: dict[str, str]
    frozen_model_training_dir: Path
    frozen_model_training_registry: dict[str, dict]
    corrected_model_training_dir: Path
    corrected_model_training_registry: dict[str, dict]
    optimization_dir: Path
    optimization_python_registry: dict[str, dict]
    optimization_r_registry: dict[str, dict]


def verify_all_inputs(
    analysis_directory: str | Path, feature_manifest_path: str | Path
) -> VerifiedInputs:
    """Verify the cleaned dataset, preprocessing artifacts, frozen and corrected model-training
    bundles, and exploratory-optimization bundle. Then prove the frozen-primary preservation and
    corrected-vs-frozen hash-equality checks before returning."""
    analysis_root = Path(analysis_directory).resolve()
    try:
        verify_frozen_primary_unchanged(analysis_root, Path(feature_manifest_path))
    except OptimizationContractError as exc:
        raise ExplainabilityPreflightError(
            "Frozen primary verification failed -- if the pre-existing restricted_data suite has "
            "run since Workstream A's correction landed, restore analysis/out/current/"
            "model_training/ and analysis/out/current/evaluation/ from "
            "analysis/out/archive/phase5-frozen-primary/ and retry"
        ) from exc

    frame, _cleaning_manifest, input_hashes = _load_verified_inputs(analysis_root)
    _verify_phase3(analysis_root / "preprocessing", frame)

    frozen_model_training_dir = analysis_root / "model_training"
    frozen_registry = _verify_model_training_directory(
        frozen_model_training_dir, MODEL_TRAINING_WORKFLOW_IDENTIFIER
    )
    corrected_model_training_dir = analysis_root / CORRECTION_SUBDIR / "model_training"
    corrected_registry = _verify_model_training_directory(
        corrected_model_training_dir, MODEL_TRAINING_WORKFLOW_IDENTIFIER
    )
    _assert_corrected_matches_frozen(
        frozen_registry,
        frozen_model_training_dir / "models",
        corrected_registry,
        corrected_model_training_dir / "models",
    )

    optimization_dir = analysis_root / OPTIMIZATION_SUBDIR
    python_registry, r_registry = _verify_optimization_directory(optimization_dir)

    return VerifiedInputs(
        frame=frame,
        input_hashes=input_hashes,
        frozen_model_training_dir=frozen_model_training_dir,
        frozen_model_training_registry=frozen_registry,
        corrected_model_training_dir=corrected_model_training_dir,
        corrected_model_training_registry=corrected_registry,
        optimization_dir=optimization_dir,
        optimization_python_registry=python_registry,
        optimization_r_registry=r_registry,
    )


def reverify_frozen_primary_unchanged(
    analysis_directory: str | Path, feature_manifest_path: str | Path
) -> None:
    """Post-run preservation sentinel: called again after publication."""
    try:
        verify_frozen_primary_unchanged(Path(analysis_directory), Path(feature_manifest_path))
    except OptimizationContractError as exc:
        raise ExplainabilityPreflightError(
            "Explainability workflow mutated the frozen primary bundle"
        ) from exc


__all__ = [
    "CORRECTION_SUBDIR",
    "OPTIMIZATION_SUBDIR",
    "ExplainabilityPreflightError",
    "VerifiedInputs",
    "reverify_frozen_primary_unchanged",
    "verify_all_inputs",
]

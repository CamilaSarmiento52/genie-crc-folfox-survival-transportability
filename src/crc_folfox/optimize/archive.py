"""Optimization workflow Gate 1 -- preservation: archive and hash-verify the frozen model-training
and primary-evaluation bundles.

Nothing else in optimization workflow may run until this gate passes. See `correction-spec.md`,
"Preservation
gate".
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from crc_folfox.optimize.contracts import Phase5AContractError


class Phase5APreservationError(Phase5AContractError):
    """Raised when the frozen model-training and primary-evaluation primary analysis cannot be
    verified unchanged."""


# Recorded 2026-08-08 in validation.md and
# validation.md. These are regression constants, exactly like the
# cohort-reconstruction workflow cohort-flow numbers in tech-stack.md -- a mismatch means the frozen
# primary analysis has
# drifted, and optimization workflow must refuse to proceed rather than archive a silently different
# bundle.
FROZEN_PHASE2_RUN_MANIFEST_SHA256 = (
    "d11982f0e9cd7f740cae7432744fa23aba336d944f7236a2f10429e775b84bfd"
)
FROZEN_PHASE2_ANALYSIS_DATASET_SHA256 = (
    "6b9e28099f547150324b33be900d228d22b411b7dbe0d68e1e08551aaab6740e"
)
FROZEN_FEATURE_MANIFEST_SHA256 = "cc891bcdca2b06179dd33e4bc55160e0e048557059e57b0b02789c7b18dc0f73"
FROZEN_PHASE4_FIT_SUMMARY_SHA256 = (
    "dfb77a84220741ad94e1362a90eaaa8c2240980ddd7c59ef0ed039548d60ad9b"
)
FROZEN_PHASE4_MODEL_REGISTRY_SHA256 = (
    "2695e55ae3ed97a9e98d5ff419544dafb7ee21e9bb65c6e6ca4ae6bd3770596f"
)
FROZEN_PHASE5_MANIFEST_PAYLOAD_SHA256 = (
    "1e25224a40ce91b4f88d16410f658291f2440e83209e4330a70d274a2860ba15"
)

ARCHIVE_DIRNAME = "phase5-frozen-primary"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenPrimaryVerification:
    phase2_run_manifest_sha256: str
    phase2_analysis_dataset_sha256: str
    feature_manifest_sha256: str
    phase4_fit_summary_sha256: str
    phase4_model_registry_sha256: str
    phase5_manifest_payload_sha256: str


def verify_frozen_primary_unchanged(
    phase2_directory: Path, feature_manifest_path: Path
) -> FrozenPrimaryVerification:
    """Re-check the six named regression hashes against the frozen values.

    Raises `frozen-primary preservation error` -- never silently continues -- if the live
    data-cleaning, model-training, and primary-evaluation
    bundle no longer matches what the model-training workflow and primary-evaluation workflow
    validation protocols recorded. Called
    both before archiving and again after every optimization workflow run (the "preservation
    sentinel").
    """
    phase2 = Path(phase2_directory)
    model_training = phase2 / "model_training"
    evaluation = phase2 / "evaluation"

    checks = {
        "phase2_run_manifest": (phase2 / "run_manifest.json", FROZEN_PHASE2_RUN_MANIFEST_SHA256),
        "phase2_analysis_dataset": (
            phase2 / "analysis_dataset.csv",
            FROZEN_PHASE2_ANALYSIS_DATASET_SHA256,
        ),
        "feature_manifest": (Path(feature_manifest_path), FROZEN_FEATURE_MANIFEST_SHA256),
        "phase4_fit_summary": (
            model_training / "fit_summary.csv",
            FROZEN_PHASE4_FIT_SUMMARY_SHA256,
        ),
        "phase4_model_registry": (
            model_training / "model_registry.json",
            FROZEN_PHASE4_MODEL_REGISTRY_SHA256,
        ),
    }
    actual: dict[str, str] = {}
    for label, (path, expected) in checks.items():
        if not path.is_file():
            raise Phase5APreservationError(f"Frozen primary input missing: {label}")
        digest = _sha256(path)
        if digest != expected:
            raise Phase5APreservationError(f"Frozen primary input changed: {label}")
        actual[label] = digest

    manifest_path = evaluation / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise Phase5APreservationError("Frozen primary input missing: phase5_evaluation_manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8")).get("manifest_payload_sha256")
    if payload != FROZEN_PHASE5_MANIFEST_PAYLOAD_SHA256:
        raise Phase5APreservationError("Frozen primary input changed: phase5_manifest_payload")

    return FrozenPrimaryVerification(
        phase2_run_manifest_sha256=actual["phase2_run_manifest"],
        phase2_analysis_dataset_sha256=actual["phase2_analysis_dataset"],
        feature_manifest_sha256=actual["feature_manifest"],
        phase4_fit_summary_sha256=actual["phase4_fit_summary"],
        phase4_model_registry_sha256=actual["phase4_model_registry"],
        phase5_manifest_payload_sha256=payload,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def archive_frozen_primary(
    phase2_directory: Path,
    feature_manifest_path: Path,
    archive_root: Path,
) -> dict[str, str]:
    """Copy the frozen `model_training/` and `evaluation/` trees under `archive_root`, and hash
    every archived file. Raises before copying anything if the six named regression hashes do not
    match; raises after copying if the archive is not byte-identical to the source.

    Idempotent: if `archive_root / ARCHIVE_DIRNAME` already exists, this only re-verifies it
    against both the frozen hashes and the live source, rather than overwriting it -- optimization
    workflow must
    never silently replace an existing archive.
    """
    verify_frozen_primary_unchanged(phase2_directory, feature_manifest_path)

    phase2 = Path(phase2_directory)
    destination = Path(archive_root) / ARCHIVE_DIRNAME
    sources = {
        "model_training": phase2 / "model_training",
        "evaluation": phase2 / "evaluation",
    }

    if not destination.exists():
        destination.mkdir(parents=True)
        for name, source in sources.items():
            shutil.copytree(source, destination / name)

    hashes: dict[str, str] = {}
    for name, source in sources.items():
        source_hashes = _tree_hashes(source)
        archived_hashes = _tree_hashes(destination / name)
        if source_hashes != archived_hashes:
            raise Phase5APreservationError(f"Archived '{name}' does not match its live source")
        hashes.update({f"{name}/{relative}": digest for relative, digest in source_hashes.items()})
    return hashes

"""The correction workstream orchestration: the corrected re-run and its reconciliation.

Re-runs model-training workflow -- now using the corrected elastic-net failure policy in
`crc_folfox.model_training.selection` -- and then primary-evaluation workflow, into a separate
output root, and
builds the archived-vs-corrected reconciliation table. The grid, splits, endpoints, horizons,
metrics, precision targets, materiality margins, and classification rule are byte-identical to
the frozen originals: this module changes only which output directory the same production
workflows publish to, and adds the frozen-input preservation check before and after. See
correction-spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crc_folfox.evaluate.workflow import EvaluationResult, run_evaluation_workflow
from crc_folfox.model_training.workflow import ModelTrainingResult, run_model_training_workflow
from crc_folfox.optimize.archive import ARCHIVE_DIRNAME, verify_frozen_primary_unchanged
from crc_folfox.optimize.reconciliation import (
    assert_no_unexpected_movement,
    build_reconciliation,
    summarize_movement,
)

CORRECTION_DIRNAME = "phase5a/correction"


@dataclass(frozen=True)
class CorrectionResult:
    model_training: ModelTrainingResult
    evaluation: EvaluationResult
    reconciliation: pd.DataFrame
    movement_summary: dict[str, int]
    output_directory: Path


def run_workstream_a_correction(
    phase2_directory: str | Path,
    feature_manifest_path: str | Path,
    archive_root: str | Path,
    output_root: str | Path,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    pooled_repeats: int | None = None,
    bootstrap_resamples: int | None = None,
    min_admissible_pooled_repeats: int | None = None,
) -> CorrectionResult:
    """Execute Workstream A's corrected re-run and reconciliation end to end.

    Requires Gate 1 to already have archived and hash-verified the frozen model-training and
    primary-evaluation bundle at
    `archive_root / archive.ARCHIVE_DIRNAME` (call
    `crc_folfox.optimize.archive.archive_frozen_primary` first) -- this function re-verifies the
    live frozen bundle is still unchanged both before and after, but does not perform the initial
    archive itself.

    `pooled_repeats`/`bootstrap_resamples`/`min_admissible_pooled_repeats` narrow the workload for
    fast synthetic exercises of this boundary; the real corrected re-run must call this with all
    three left at their defaults, which forward to the same model-training and primary-evaluation
    defaults unchanged.
    """
    phase2 = Path(phase2_directory).resolve()
    feature_manifest = Path(feature_manifest_path)
    verify_frozen_primary_unchanged(phase2, feature_manifest)

    archived_evaluation_dir = Path(archive_root).resolve() / ARCHIVE_DIRNAME / "evaluation"
    archived_performance = pd.read_csv(archived_evaluation_dir / "performance_summary.csv")

    output_dir = Path(output_root).resolve() / CORRECTION_DIRNAME
    model_training_output = output_dir / "model_training"
    evaluation_output = output_dir / "evaluation"

    model_training_kwargs: dict[str, object] = {"endpoints": endpoints}
    if pooled_repeats is not None:
        model_training_kwargs["pooled_repeats"] = pooled_repeats
    training_result = run_model_training_workflow(
        phase2, output_directory=model_training_output, **model_training_kwargs
    )

    evaluation_kwargs: dict[str, object] = {
        "phase4_directory": model_training_output,
        "endpoints": endpoints,
    }
    if pooled_repeats is not None:
        evaluation_kwargs["pooled_repeats"] = pooled_repeats
    if bootstrap_resamples is not None:
        evaluation_kwargs["bootstrap_resamples"] = bootstrap_resamples
    if min_admissible_pooled_repeats is not None:
        evaluation_kwargs["min_admissible_pooled_repeats"] = min_admissible_pooled_repeats
    evaluation_result = run_evaluation_workflow(
        phase2, output_directory=evaluation_output, **evaluation_kwargs
    )

    reconciliation = build_reconciliation(
        archived_performance, evaluation_result.performance_summary
    )
    assert_no_unexpected_movement(reconciliation)

    output_dir.mkdir(parents=True, exist_ok=True)
    reconciliation.to_csv(output_dir / "reconciliation.csv", index=False)

    verify_frozen_primary_unchanged(phase2, feature_manifest)

    return CorrectionResult(
        model_training=training_result,
        evaluation=evaluation_result,
        reconciliation=reconciliation,
        movement_summary=summarize_movement(reconciliation),
        output_directory=output_dir,
    )

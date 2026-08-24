"""Forest-model-retry workflow tuning for `ltrccif`, on the locked model-training forest grid.

Every constant and the objective itself are **imported**, never redefined: the grid comes from
`forest_mtry_grid` and `FOREST_NODESIZE_GRID`, the horizons from `HORIZONS`, and the score from
`delayed_entry_ipcw_brier`. forest-model-retry workflow may not loosen a threshold or widen a range
that an earlier
workflow froze (`mission.md` non-negotiable 9).

Two mappings are worth stating explicitly, because they are where a silent divergence would hide:

* **`nodesize` -> `minbucket`.** model-training workflow's locked forest grid tunes `nodesize`, the
  terminal node
  size of an `ltrcrrf` tree. `ltrccif`'s equivalent is `minbucket` in `ctree_control`. The grid
  values are used unchanged.
* **Tuning scores the `unconditional` scale.** The IPCW Brier compares predicted survival against
  observed outcomes measured from FOLFOX initiation, so the prediction must be unconditional --
  the same scale `evaluation-spec.md` locks as primary. Scoring the entry-conditional prediction
  against unconditional outcomes would compare two different quantities.

These tuning scores are **not** held-out performance estimates and are never reported as
comparative results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from crc_folfox.forest.r_adapter import RConditionalForestAdapter
from crc_folfox.model_training.contracts import (
    FOREST_NODESIZE_GRID,
    HORIZONS,
    ModelTrainingContractError,
    forest_mtry_grid,
)
from crc_folfox.model_training.tuning import delayed_entry_ipcw_brier

#: The scale the tuning objective scores. Locked, not configurable.
TUNING_RISK_SCALE = "unconditional"


@dataclass(frozen=True)
class CandidateScore:
    """One `(mtry, minbucket)` combination's outcome. Failures are retained, not dropped."""

    mtry: int
    minbucket: int
    score: float | None
    status: str  # "scored" or "failed"
    reason: str = ""


@dataclass(frozen=True)
class TuningResult:
    selected_mtry: int | None
    selected_minbucket: int | None
    selected_score: float | None
    status: str  # "selected" or "no_eligible_candidate"
    reason: str
    candidates: tuple[CandidateScore, ...]


def cif_grid(predictor_count: int) -> tuple[tuple[int, int], ...]:
    """The locked model-training forest grid, as `(mtry, minbucket)` pairs -- at most nine."""
    return tuple(
        (mtry, minbucket)
        for mtry in forest_mtry_grid(predictor_count)
        for minbucket in FOREST_NODESIZE_GRID
    )


def _select(candidates: Sequence[CandidateScore]) -> tuple[CandidateScore | None, str]:
    """Smallest mean score wins; exact ties take the larger minbucket, then the smaller mtry.

    The tie-break is model-training workflow's, imported verbatim in intent: it favours the simpler
    forest.
    """
    scored = [c for c in candidates if c.status == "scored" and c.score is not None]
    if not scored:
        return None, "no_eligible_candidate: every grid combination failed"
    best = min(scored, key=lambda c: (c.score, -c.minbucket, c.mtry))
    return best, "selected"


def tune_ltrccif(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    seed: int,
    adapter: RConditionalForestAdapter | None = None,
    horizons: tuple[float, ...] = HORIZONS,
    grid: Sequence[tuple[int, int]] | None = None,
) -> TuningResult:
    """Score every locked grid combination across inner folds and select one.

    `folds` is a sequence of `(train_matrix, train_metadata, validation_matrix,
    validation_metadata)`. Every fit and prediction happens inside inner-training data only; the
    censoring estimator inside the objective is fitted on inner-training rows by construction.

    A failure disqualifies **only** the combination that failed, never the whole scope -- the
    correction workstream correction, applied here from the start rather than discovered later.
    """
    if not folds:
        raise ModelTrainingContractError("Tuning requires at least one inner fold")
    adapter = adapter or RConditionalForestAdapter()
    # `grid` is an override for reduced-inventory smoke tests only; a real run passes None and
    # gets the locked nine-point model-training grid.
    combinations = tuple(grid) if grid is not None else cif_grid(folds[0][0].shape[1])
    candidates: list[CandidateScore] = []

    with TemporaryDirectory(prefix="crc-phase9-tuning-") as directory:
        root = Path(directory)
        for mtry, minbucket in combinations:
            fold_scores: list[float] = []
            failure = ""
            for index, (train_x, train_meta, valid_x, valid_meta) in enumerate(folds):
                try:
                    fitted = adapter.fit(
                        train_x,
                        train_meta,
                        mtry=mtry,
                        minbucket=minbucket,
                        seed=seed,
                        destination=root / f"m{mtry}-b{minbucket}-f{index}.rds",
                    )
                    predicted = adapter.predict(fitted.bundle, valid_x, valid_meta, times=horizons)[
                        TUNING_RISK_SCALE
                    ]
                    fold_scores.append(
                        delayed_entry_ipcw_brier(
                            train_meta, valid_meta, predicted, horizons=horizons
                        )
                    )
                except (ModelTrainingContractError, ValueError, KeyError) as exc:
                    failure = f"{type(exc).__name__} on inner fold {index}: {exc}"
                    break
            if failure:
                candidates.append(CandidateScore(mtry, minbucket, None, "failed", failure))
                continue
            mean = float(np.mean(fold_scores))
            if not np.isfinite(mean):
                candidates.append(
                    CandidateScore(mtry, minbucket, None, "failed", "non_finite_mean_score")
                )
                continue
            candidates.append(CandidateScore(mtry, minbucket, mean, "scored"))

    best, status = _select(candidates)
    if best is None:
        return TuningResult(None, None, None, "no_eligible_candidate", status, tuple(candidates))
    return TuningResult(
        best.mtry,
        best.minbucket,
        best.score,
        "selected",
        f"smallest mean delayed-entry IPCW Brier across {len(folds)} inner fold(s)",
        tuple(candidates),
    )


__all__ = [
    "TUNING_RISK_SCALE",
    "CandidateScore",
    "TuningResult",
    "cif_grid",
    "tune_ltrccif",
]

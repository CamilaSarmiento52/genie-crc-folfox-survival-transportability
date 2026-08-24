"""Pooled out-of-fold prediction aggregation across admissible repeats.

Per `evaluation-spec.md`, "Pooled out-of-fold aggregation": a repeat is admissible only when all
five of its fold-scope model-training workflow fits succeeded; within an admissible repeat, every
eligible patient
has exactly one out-of-fold prediction (from the fold that held them out); the patient's pooled risk
is the arithmetic mean, on the risk scale, over all admissible repeats. A pooled cell requires at
least `MIN_ADMISSIBLE_POOLED_REPEATS` (5 of 10) admissible repeats.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.contracts import (
    MIN_ADMISSIBLE_POOLED_REPEATS,
    EvaluationContractError,
)
from crc_folfox.evaluate.prediction import predict_for_frame
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.model_training.contracts import HORIZONS, ModelScope
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition


@dataclass(frozen=True)
class PooledAggregationResult:
    status: str  # "ok" or "insufficient_admissible_repeats"
    admissible_repeats: int
    total_repeats: int
    patient_key: tuple[str, ...]
    institution: tuple[str, ...]
    entry: np.ndarray
    exit: np.ndarray
    event: np.ndarray
    unconditional: np.ndarray | None  # (n_patients, n_horizons), mean over admissible repeats
    entry_conditional: np.ndarray | None
    linear_predictor: np.ndarray | None  # (n_patients,), mean over admissible repeats
    per_repeat_unconditional: tuple[np.ndarray, ...]  # for dispersion diagnostics
    per_repeat_admissible: tuple[int, ...]  # the admissible repeat indices


def _centre_frame(view: EndpointView, centre: str) -> pd.DataFrame:
    frame = view.data.loc[view.data["institution"].eq(centre)].copy()
    if frame.empty:
        raise EvaluationContractError("Centre has no endpoint-eligible patients")
    return frame.sort_values("patient_key", kind="stable").reset_index(drop=True)


def _repeat_admissible(
    repeat_splits: tuple[SplitDefinition, ...],
    model_name: str,
    registry: Mapping[str, Mapping[str, object]],
) -> bool:
    for split in repeat_splits:
        model_id = ModelScope(split).model_id(model_name)
        record = registry.get(model_id)
        if record is None or record.get("fit_status") != "fit_succeeded":
            return False
    return True


def _repeat_prediction(
    view: EndpointView,
    repeat_splits: tuple[SplitDefinition, ...],
    model_name: str,
    registry: Mapping[str, Mapping[str, object]],
    models_directory: Path,
    centre_keys: tuple[str, ...],
    horizons: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict every centre patient using whichever fold held them out in this repeat."""
    key_index = {key: index for index, key in enumerate(centre_keys)}
    unconditional = np.full((len(centre_keys), len(horizons)), np.nan)
    conditional = np.full((len(centre_keys), len(horizons)), np.nan)
    lp = np.full(len(centre_keys), np.nan)
    seen: set[str] = set()
    for split in repeat_splits:
        fold_keys = tuple(key for key in split.assessment_keys if key in key_index)
        if not fold_keys:
            continue
        model_id = ModelScope(split).model_id(model_name)
        record = RegistryRecord(**registry[model_id])
        state = verify_registry_load(models_directory, record)
        _, fold_assessment = split.frames(view)
        subset = fold_assessment.loc[fold_assessment["patient_key"].astype(str).isin(fold_keys)]
        predictions = predict_for_frame(state, subset, horizons)
        for row_index, key in enumerate(predictions.patient_key):
            position = key_index[key]
            unconditional[position] = predictions.unconditional[row_index]
            conditional[position] = predictions.entry_conditional[row_index]
            lp[position] = predictions.linear_predictor[row_index]
            seen.add(key)
    if seen != set(centre_keys):
        raise EvaluationContractError(
            "Pooled repeat does not cover every centre patient exactly once"
        )
    return unconditional, conditional, lp


def aggregate_pooled_predictions(
    view: EndpointView,
    centre: str,
    model_name: str,
    pooled_splits: tuple[SplitDefinition, ...],
    registry: Mapping[str, Mapping[str, object]],
    models_directory: Path,
    horizons: tuple[float, ...] = HORIZONS,
    min_admissible_repeats: int = MIN_ADMISSIBLE_POOLED_REPEATS,
) -> PooledAggregationResult:
    centre_frame = _centre_frame(view, centre)
    centre_keys = tuple(centre_frame["patient_key"].astype(str))

    repeats: dict[int, list[SplitDefinition]] = {}
    for split in pooled_splits:
        repeats.setdefault(split.repeat, []).append(split)
    total_repeats = len(repeats)

    per_repeat_unconditional: list[np.ndarray] = []
    per_repeat_conditional: list[np.ndarray] = []
    per_repeat_lp: list[np.ndarray] = []
    admissible_repeat_indices: list[int] = []
    for repeat in sorted(repeats):
        repeat_splits = tuple(
            sorted(repeats[repeat], key=lambda split: split.fold if split.fold is not None else -1)
        )
        if not _repeat_admissible(repeat_splits, model_name, registry):
            continue
        unconditional, conditional, lp = _repeat_prediction(
            view, repeat_splits, model_name, registry, models_directory, centre_keys, horizons
        )
        per_repeat_unconditional.append(unconditional)
        per_repeat_conditional.append(conditional)
        per_repeat_lp.append(lp)
        admissible_repeat_indices.append(repeat)

    admissible_count = len(admissible_repeat_indices)
    base = dict(
        admissible_repeats=admissible_count,
        total_repeats=total_repeats,
        patient_key=centre_keys,
        institution=tuple(centre_frame["institution"].astype(str)),
        entry=centre_frame["entry"].to_numpy(dtype=float),
        exit=centre_frame["exit"].to_numpy(dtype=float),
        event=centre_frame["event"].to_numpy(dtype=int),
        per_repeat_unconditional=tuple(per_repeat_unconditional),
        per_repeat_admissible=tuple(admissible_repeat_indices),
    )
    if admissible_count < min_admissible_repeats:
        return PooledAggregationResult(
            status="insufficient_admissible_repeats",
            unconditional=None,
            entry_conditional=None,
            linear_predictor=None,
            **base,
        )
    unconditional_mean = np.nanmean(np.stack(per_repeat_unconditional), axis=0)
    conditional_mean = np.nanmean(np.stack(per_repeat_conditional), axis=0)
    lp_mean = np.nanmean(np.stack(per_repeat_lp), axis=0)
    return PooledAggregationResult(
        status="ok",
        unconditional=unconditional_mean,
        entry_conditional=conditional_mean,
        linear_predictor=lp_mean,
        **base,
    )


__all__ = ["PooledAggregationResult", "aggregate_pooled_predictions"]

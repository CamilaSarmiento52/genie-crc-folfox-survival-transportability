"""Model-training workflow public contracts and deterministic model inventory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from crc_folfox.preprocessing.contracts import SplitDefinition
from crc_folfox.preprocessing.splits import ROOT_SEED, derive_seed

HORIZONS = (365.0, 730.0)

# The four Cox candidates are always required. `ltrc_rrf` and `ltrc_boosting` are admitted
# only after their own pre-result compatibility gate passes (model-training-spec.md); they are
# never part of the required registry. `MODEL_NAMES` is kept as an alias of `REQUIRED_MODEL_NAMES`
# for existing callers that have not yet migrated to the split names.
REQUIRED_MODEL_NAMES = ("minimal_cox", "core_cox", "elastic_net_cox", "spline_cox")
CONDITIONAL_MODEL_NAMES = ("ltrc_rrf", "ltrc_boosting")
MODEL_NAMES = REQUIRED_MODEL_NAMES

TUNABLE_MODELS = frozenset({"elastic_net_cox", "ltrc_rrf"})
ELASTIC_NET_GRID = tuple(
    (penalizer, l1_ratio) for penalizer in (0.001, 0.01, 0.1, 1.0) for l1_ratio in (0.0, 0.5, 1.0)
)
FOREST_NODESIZE_GRID = (10, 20, 40)

# Frozen 2026-08-07 (Block C, Step 7): neither `ltrc_rrf` nor `ltrc_boosting` passed its pre-
# result admission gate (see implement.md), so the required final-fit inventory is the Cox-only
# baseline -- 53 development scopes (3 IECV outer + 50 pooled) x 4 required models x 2 endpoints.
# This constant must be updated, in the same dated commit as the admission decision, only if a
# later amendment admits a conditional candidate.
DEVELOPMENT_SCOPES_PER_ENDPOINT = 53
REQUIRED_FINAL_FIT_COUNT = DEVELOPMENT_SCOPES_PER_ENDPOINT * len(REQUIRED_MODEL_NAMES) * 2


class ModelTrainingContractError(ValueError):
    """Raised for a privacy-safe model-training workflow contract violation."""


class FittedDelayedEntryModel(Protocol):
    """A fitted candidate: predicts bounded, nonincreasing survival at fixed horizons."""

    def predict_survival(
        self, matrix: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
    ) -> np.ndarray: ...


class DelayedEntryAdapter(Protocol):
    """Language-neutral contract every candidate family (Python or R) must satisfy.

    Every adapter accepts one ordered finite numeric predictor matrix plus aligned
    entry/exit/event vectors and a deterministic seed and frozen config, and returns a fitted
    model exposing `predict_survival`. Structural (Protocol), not an ABC: the eventual R adapter
    crosses a subprocess boundary and cannot subclass a Python base.
    """

    def fit(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        seed: int,
        config: Mapping[str, object],
    ) -> FittedDelayedEntryModel: ...


@dataclass(frozen=True)
class ModelScope:
    """A single assessment-blind development scope."""

    split: SplitDefinition

    @property
    def model_prefix(self) -> str:
        split = self.split
        parts = [f"endpoint={split.endpoint}", f"design={split.design}"]
        if split.held_out_institution:
            parts.append(f"holdout={split.held_out_institution}")
        if split.repeat is not None:
            parts.append(f"repeat={split.repeat}")
        if split.fold is not None:
            parts.append(f"fold={split.fold}")
        return "|".join(parts)

    def model_id(self, model_name: str) -> str:
        if model_name not in REQUIRED_MODEL_NAMES and model_name not in CONDITIONAL_MODEL_NAMES:
            raise ModelTrainingContractError("Unknown model candidate")
        return f"{self.model_prefix}|model={model_name}"


def child_seed(
    *,
    endpoint: str,
    design: str,
    held_out_institution: str | None = None,
    repeat: int | None = None,
    fold: int | None = None,
    model_name: str = "",
    candidate: str = "",
    purpose: str,
) -> int:
    """Derive a workflow-4 seed, extending the workflow-3 canonical seed tuple."""
    base = derive_seed(
        endpoint=endpoint,
        design=design,
        held_out_institution=held_out_institution,
        repeat=repeat,
        fold=fold,
        purpose=purpose,
        root_seed=ROOT_SEED,
    )
    fields = (str(base), model_name, candidate, purpose)
    return int(hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:8], 16)


def r_seed(seed: int) -> int:
    """Map an unsigned child seed to R's positive integer range."""
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ModelTrainingContractError("Child seed is outside the unsigned 32-bit range")
    return 1 + seed % 2147483646


def forest_mtry_grid(predictor_count: int) -> tuple[int, ...]:
    """Return the frozen clipped square-root forest grid."""
    if predictor_count < 1:
        raise ModelTrainingContractError("Forest requires at least one predictor")
    root = int(np.floor(np.sqrt(predictor_count)))
    return tuple(
        sorted({max(1, min(predictor_count, value)) for value in (root // 2, root, 2 * root)})
    )


def validate_survival_inputs(
    matrix: np.ndarray, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray
) -> None:
    """Validate the language-neutral delayed-entry model input contract."""
    if matrix.ndim != 2 or not len(matrix):
        raise ModelTrainingContractError("Predictor matrix must be nonempty and two-dimensional")
    if not np.isfinite(matrix).all():
        raise ModelTrainingContractError("Predictor matrix contains nonfinite values")
    if not (len(entry) == len(exit_) == len(event) == len(matrix)):
        raise ModelTrainingContractError("Survival vectors are not aligned with predictors")
    if (
        not np.isfinite(entry).all()
        or not np.isfinite(exit_).all()
        or not set(event).issubset({0, 1})
    ):
        raise ModelTrainingContractError("Survival inputs are invalid")
    if np.any(entry < 0) or np.any(entry >= exit_):
        raise ModelTrainingContractError("Delayed-entry ordering is invalid")

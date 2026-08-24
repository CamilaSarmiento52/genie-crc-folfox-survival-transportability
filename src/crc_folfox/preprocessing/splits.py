"""Deterministic preprocessing workflow IECV, inner, and pooled partitions."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from crc_folfox.preprocessing.contracts import (
    INSTITUTIONS,
    EndpointView,
    PreprocessingContractError,
    SplitDefinition,
)

ROOT_SEED = 42
N_FOLDS = 5
INNER_REPEATS = 5
POOLED_REPEATS = 10


def derive_seed(
    *,
    endpoint: str,
    design: str,
    held_out_institution: str | None = None,
    repeat: int | None = None,
    fold: int | None = None,
    purpose: str,
    root_seed: int = ROOT_SEED,
) -> int:
    """Derive the contract's stable unsigned 32-bit child seed."""
    fields = (
        str(root_seed),
        endpoint,
        design,
        held_out_institution or "",
        "" if repeat is None else str(repeat),
        "" if fold is None else str(fold),
        purpose,
    )
    digest = hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_iecv_splits(view: EndpointView) -> tuple[SplitDefinition, ...]:
    """Create the three literal leave-one-institution-out partitions."""
    splits = []
    for held_out in INSTITUTIONS:
        assessment = view.data.loc[view.data["institution"].eq(held_out), "patient_key"]
        development = view.data.loc[view.data["institution"].ne(held_out), "patient_key"]
        if assessment.empty or development.empty:
            raise PreprocessingContractError("IECV partition has an empty side")
        splits.append(
            SplitDefinition(
                split_id=f"iecv|holdout={held_out}",
                endpoint=view.endpoint,
                design="iecv_outer",
                development_keys=tuple(development.astype(str)),
                assessment_keys=tuple(assessment.astype(str)),
                held_out_institution=held_out,
            )
        )
    _validate_iecv_coverage(view, splits)
    return tuple(splits)


def build_inner_splits(
    view: EndpointView,
    outer_split: SplitDefinition,
    *,
    repeats: int = INNER_REPEATS,
    folds: int = N_FOLDS,
) -> tuple[SplitDefinition, ...]:
    """Create repeated stratified inner partitions inside one IECV development set."""
    development, _ = outer_split.frames(view)
    return _build_repeated_splits(
        development,
        endpoint=view.endpoint,
        design="iecv_inner",
        repeats=repeats,
        folds=folds,
        held_out_institution=outer_split.held_out_institution,
    )


def build_pooled_splits(
    view: EndpointView,
    *,
    repeats: int = POOLED_REPEATS,
    folds: int = N_FOLDS,
) -> tuple[SplitDefinition, ...]:
    """Create repeated pooled mixed-institution out-of-fold partitions."""
    return _build_repeated_splits(
        view.data,
        endpoint=view.endpoint,
        design="pooled",
        repeats=repeats,
        folds=folds,
    )


def _build_repeated_splits(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    design: str,
    repeats: int,
    folds: int,
    held_out_institution: str | None = None,
) -> tuple[SplitDefinition, ...]:
    if repeats < 1 or folds < 2:
        raise PreprocessingContractError("Repeated split dimensions are invalid")
    ordered = frame.sort_values("patient_key", kind="stable").reset_index(drop=True)
    strata = ordered["institution"].astype(str) + "|event=" + ordered["event"].astype(str)
    counts = strata.value_counts()
    if counts.empty or int(counts.min()) < folds:
        raise PreprocessingContractError(
            "Institution-by-event stratum has fewer rows than the required folds"
        )

    result = []
    for repeat in range(repeats):
        seed = derive_seed(
            endpoint=endpoint,
            design=design,
            held_out_institution=held_out_institution,
            repeat=repeat,
            purpose="partition",
        )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (development_index, assessment_index) in enumerate(
            splitter.split(ordered, strata)
        ):
            held = f"|holdout={held_out_institution}" if held_out_institution else ""
            result.append(
                SplitDefinition(
                    split_id=f"{design}{held}|repeat={repeat}|fold={fold}",
                    endpoint=endpoint,
                    design=design,
                    development_keys=tuple(
                        ordered.iloc[development_index]["patient_key"].astype(str)
                    ),
                    assessment_keys=tuple(
                        ordered.iloc[assessment_index]["patient_key"].astype(str)
                    ),
                    held_out_institution=held_out_institution,
                    repeat=repeat,
                    fold=fold,
                    seed=seed,
                )
            )
    _validate_repeated_coverage(ordered, result, repeats=repeats, folds=folds)
    return tuple(result)


def _validate_iecv_coverage(view: EndpointView, splits: list[SplitDefinition]) -> None:
    expected = set(view.data["patient_key"].astype(str))
    seen: list[str] = []
    for split in splits:
        development = set(split.development_keys)
        assessment = set(split.assessment_keys)
        if development & assessment or development | assessment != expected:
            raise PreprocessingContractError("IECV coverage or disjointness failed")
        seen.extend(split.assessment_keys)
    if len(seen) != len(expected) or set(seen) != expected:
        raise PreprocessingContractError("IECV assessment coverage failed")


def _validate_repeated_coverage(
    frame: pd.DataFrame,
    splits: list[SplitDefinition],
    *,
    repeats: int,
    folds: int,
) -> None:
    expected = set(frame["patient_key"].astype(str))
    if len(splits) != repeats * folds:
        raise PreprocessingContractError("Repeated split count is incorrect")
    for repeat in range(repeats):
        repeated = [split for split in splits if split.repeat == repeat]
        seen: list[str] = []
        for split in repeated:
            development = set(split.development_keys)
            assessment = set(split.assessment_keys)
            if development & assessment or development | assessment != expected:
                raise PreprocessingContractError("Repeated split coverage or disjointness failed")
            seen.extend(split.assessment_keys)
        if len(seen) != len(expected) or set(seen) != expected:
            raise PreprocessingContractError("Repeated assessment coverage failed")


__all__ = [
    "INNER_REPEATS",
    "N_FOLDS",
    "POOLED_REPEATS",
    "ROOT_SEED",
    "build_iecv_splits",
    "build_inner_splits",
    "build_pooled_splits",
    "derive_seed",
]

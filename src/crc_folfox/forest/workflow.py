"""Forest-model-retry workflow fitting orchestration for `ltrccif`, with per-scope checkpointing.

The run is long — roughly 63 hours serial across 53 OS development scopes and ~11,978 fits — so
**checkpointing is a correctness requirement, not a convenience.** explainability workflow lost a
~38-hour execution
twice to causes outside the code (a harness restart, then an overnight sleep event); the per-model
split it built afterwards is what let five of seven models' results survive. The forest-model-retry
workflow applies that
lesson from the start: one checkpoint line per completed scope, and resuming skips exactly
the scopes already recorded.

Scope boundary. This module **fits and tunes only**. It computes no performance metric, no
transportability gap, and no comparison against `minimal_cox` — those use primary-evaluation
workflow's imported
machinery and belong to the evaluation step. Everything published here carries
`analysis_status = post_hoc_exploratory`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from crc_folfox.forest.r_adapter import CIF_NTREE, RConditionalForestAdapter
from crc_folfox.forest.tuning import TuningResult, cif_grid, tune_ltrccif
from crc_folfox.model_training.contracts import ModelTrainingContractError
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition, build_endpoint_view
from crc_folfox.preprocessing.splits import (
    build_iecv_splits,
    build_inner_splits,
    build_pooled_splits,
    derive_seed,
)
from crc_folfox.preprocessing.transform import FittedPreprocessor

#: The forest uses the preprocessing workflow **core** matrix, per `forest-retry-spec.md` section 4.
FEATURE_SET = "core"

#: The candidate name every artifact row carries.
CANDIDATE = "ltrc_cif"

#: Every forest-model-retry workflow row is labelled this. Non-negotiable 9.
ANALYSIS_STATUS = "post_hoc_exploratory"


class ForestWorkflowError(RuntimeError):
    """Forest-model-retry workflow orchestration failed."""


@dataclass(frozen=True)
class ForestTrainingResult:
    fit_summary: pd.DataFrame
    tuning_summary: pd.DataFrame
    output: Path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _scope_id(split: SplitDefinition) -> str:
    parts = [f"endpoint={split.endpoint}", f"design={split.design}"]
    if split.held_out_institution:
        parts.append(f"holdout={split.held_out_institution}")
    if split.repeat is not None:
        parts.append(f"repeat={split.repeat}")
    if split.fold is not None:
        parts.append(f"fold={split.fold}")
    parts.append(f"model={CANDIDATE}")
    return "|".join(parts)


def _development_scopes(view: EndpointView) -> tuple[SplitDefinition, ...]:
    return (*build_iecv_splits(view), *build_pooled_splits(view))


def _inner_folds(
    view: EndpointView, outer: SplitDefinition, scope_id: str, limit: int | None
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Preprocess each inner fold on inner-training rows only.

    The preprocessor is fitted on the inner-training frame and applied to the inner-validation
    frame, so no validation row influences imputation, pooling, encoding, or scaling.
    """
    if outer.design == "iecv_outer":
        inner_splits = build_inner_splits(view, outer)
    elif outer.design == "pooled":
        from crc_folfox.model_training.selection import build_nested_pooled_splits

        inner_splits = build_nested_pooled_splits(view, outer)
    else:
        raise ModelTrainingContractError(f"Unsupported parent design: {outer.design}")
    if limit is not None:
        inner_splits = tuple(inner_splits)[:limit]

    folds = []
    for index, inner in enumerate(inner_splits):
        train, valid = inner.frames(view)
        preprocessor = FittedPreprocessor.fit(
            train,
            endpoint=view.endpoint,
            feature_set=FEATURE_SET,
            fit_scope=f"{scope_id}|inner={index}",
        )
        fitted_train = preprocessor.transform(train)
        fitted_valid = preprocessor.transform(valid)
        folds.append(
            (
                fitted_train.matrix,
                fitted_train.metadata,
                fitted_valid.matrix,
                fitted_valid.metadata,
            )
        )
    return folds


def _tuning_rows(scope_id: str, split: SplitDefinition, result: TuningResult) -> list[dict]:
    """Every grid combination is retained, scored or failed. Failures are results, not gaps."""
    return [
        {
            "scope_id": scope_id,
            "endpoint": split.endpoint,
            "design": split.design,
            "held_out_institution": split.held_out_institution or "",
            "candidate": CANDIDATE,
            "mtry": candidate.mtry,
            "minbucket": candidate.minbucket,
            "score": candidate.score if candidate.score is not None else "",
            "candidate_status": candidate.status,
            "reason": candidate.reason,
            "selected": bool(
                candidate.status == "scored"
                and candidate.mtry == result.selected_mtry
                and candidate.minbucket == result.selected_minbucket
            ),
            "analysis_status": ANALYSIS_STATUS,
        }
        for candidate in result.candidates
    ]


def _load_checkpoint(path: Path) -> tuple[list[dict], list[dict], set[str]]:
    if not path.is_file():
        return [], [], set()
    fit_rows, tuning_rows = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        fit_rows.append(payload["fit"])
        tuning_rows.extend(payload["tuning"])
    return fit_rows, tuning_rows, {row["scope_id"] for row in fit_rows}


def _append_checkpoint(path: Path, fit_row: dict, tuning: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"fit": fit_row, "tuning": tuning}) + "\n")
        handle.flush()


def run_forest_training_workflow(
    phase2_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    run_id: str,
    endpoints: tuple[str, ...] = ("OS",),
    scope_limit: int | None = None,
    inner_limit: int | None = None,
    grid_limit: int | None = None,
    adapter: RConditionalForestAdapter | None = None,
    progress: bool = False,
) -> ForestTrainingResult:
    """Tune and fit `ltrccif` across every development scope, checkpointing after each.

    `endpoints` defaults to OS only, per the dated 2026-08-15 amendment. `scope_limit`,
    `inner_limit`, and `grid_limit` exist **only** to exercise this boundary on a reduced
    inventory before committing to a multi-day run; a real run leaves all three at `None`.
    """
    phase2 = Path(phase2_directory).resolve()
    output = Path(output_directory).resolve() if output_directory else phase2 / "phase9"
    adapter = adapter or RConditionalForestAdapter()
    checkpoint = output / f"checkpoint-{run_id}.jsonl"
    bundles = output / "models"
    bundles.mkdir(parents=True, exist_ok=True)

    from crc_folfox.pipeline import load_clean_analysis_dataset

    frame = load_clean_analysis_dataset(phase2 / "analysis_dataset.csv")
    fit_rows, tuning_rows, completed = _load_checkpoint(checkpoint)
    if completed and progress:
        print(f"[{_now()}] resuming: {len(completed)} scope(s) already complete", flush=True)

    for endpoint in endpoints:
        view = build_endpoint_view(frame, endpoint)
        scopes = _development_scopes(view)
        if scope_limit is not None:
            scopes = scopes[:scope_limit]
        for position, split in enumerate(scopes, start=1):
            scope_id = _scope_id(split)
            if scope_id in completed:
                continue
            if progress:
                print(
                    f"[{_now()}] scope {position}/{len(scopes)}  {scope_id}",
                    flush=True,
                )
            development, _ = split.frames(view)
            seed = derive_seed(
                endpoint=split.endpoint,
                design=split.design,
                held_out_institution=split.held_out_institution,
                repeat=split.repeat,
                fold=split.fold,
                purpose=f"{CANDIDATE}_fit",
            )
            try:
                folds = _inner_folds(view, split, scope_id, inner_limit)
                grid = cif_grid(folds[0][0].shape[1])
                if grid_limit is not None:
                    grid = grid[:grid_limit]
                tuning = tune_ltrccif(folds, seed=seed, adapter=adapter, grid=grid)
                if tuning.status != "selected":
                    fit_row = _failed_row(scope_id, split, development, tuning.reason)
                else:
                    preprocessor = FittedPreprocessor.fit(
                        development,
                        endpoint=split.endpoint,
                        feature_set=FEATURE_SET,
                        fit_scope=scope_id,
                    )
                    transformed = preprocessor.transform(development)
                    fitted = adapter.fit(
                        transformed.matrix,
                        transformed.metadata,
                        mtry=tuning.selected_mtry,
                        minbucket=tuning.selected_minbucket,
                        seed=seed,
                        destination=bundles / f"{_safe(scope_id)}.rds",
                    )
                    fit_row = {
                        "scope_id": scope_id,
                        "endpoint": split.endpoint,
                        "design": split.design,
                        "held_out_institution": split.held_out_institution or "",
                        "candidate": CANDIDATE,
                        "development_patients": len(development),
                        "development_events": int(development["event"].sum()),
                        "ntree": CIF_NTREE,
                        "selected_mtry": tuning.selected_mtry,
                        "selected_minbucket": tuning.selected_minbucket,
                        "selected_score": tuning.selected_score,
                        "fit_status": "fit_succeeded",
                        "reason": "",
                        "bundle": str(fitted.bundle.relative_to(output)),
                        "analysis_status": ANALYSIS_STATUS,
                        "completed_at": _now(),
                    }
            except Exception as exc:  # noqa: BLE001 -- one scope's failure is a recorded result
                fit_row = _failed_row(scope_id, split, development, f"{type(exc).__name__}: {exc}")
                tuning = None

            scope_tuning = _tuning_rows(scope_id, split, tuning) if tuning is not None else []
            fit_rows.append(fit_row)
            tuning_rows.extend(scope_tuning)
            _append_checkpoint(checkpoint, fit_row, scope_tuning)

    fit_summary = pd.DataFrame(fit_rows).sort_values("scope_id", kind="stable")
    tuning_summary = (
        pd.DataFrame(tuning_rows).sort_values(["scope_id", "mtry", "minbucket"], kind="stable")
        if tuning_rows
        else pd.DataFrame()
    )
    _publish(output, fit_summary, tuning_summary, run_id)
    return ForestTrainingResult(
        fit_summary.reset_index(drop=True), tuning_summary.reset_index(drop=True), output
    )


def _failed_row(
    scope_id: str, split: SplitDefinition, development: pd.DataFrame, reason: str
) -> dict[str, Any]:
    return {
        "scope_id": scope_id,
        "endpoint": split.endpoint,
        "design": split.design,
        "held_out_institution": split.held_out_institution or "",
        "candidate": CANDIDATE,
        "development_patients": len(development),
        "development_events": int(development["event"].sum()),
        "ntree": CIF_NTREE,
        "selected_mtry": "",
        "selected_minbucket": "",
        "selected_score": "",
        "fit_status": "fit_failed",
        "reason": reason,
        "bundle": "",
        "analysis_status": ANALYSIS_STATUS,
        "completed_at": _now(),
    }


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _publish(output: Path, fit: pd.DataFrame, tuning: pd.DataFrame, run_id: str) -> None:
    """Write the aggregate frames atomically. Patient-level content never reaches disk."""
    output.mkdir(parents=True, exist_ok=True)
    staging = output / f".staging-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    fit.to_csv(staging / "fit_summary.csv", index=False)
    if not tuning.empty:
        tuning.to_csv(staging / "tuning_summary.csv", index=False)
    manifest = {
        "phase": 9,
        "candidate": CANDIDATE,
        "analysis_status": ANALYSIS_STATUS,
        "ntree": CIF_NTREE,
        "feature_set": FEATURE_SET,
        "run_id": run_id,
        "published_at": _now(),
        "scopes": int(len(fit)),
        "fit_succeeded": int((fit["fit_status"] == "fit_succeeded").sum()) if len(fit) else 0,
        "fit_failed": int((fit["fit_status"] == "fit_failed").sum()) if len(fit) else 0,
    }
    (staging / "phase9_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for item in staging.iterdir():
        item.replace(output / item.name)
    shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "ANALYSIS_STATUS",
    "CANDIDATE",
    "FEATURE_SET",
    "ForestTrainingResult",
    "ForestWorkflowError",
    "run_forest_training_workflow",
]

"""Forest-model-retry workflow evaluation: score `ltrc_cif` with primary-evaluation workflow's
machinery, unchanged.

Nothing here reimplements a metric, an interval, or a classification. The primary-evaluation
workflow's estimators,
bootstrap index matrices, reporting-status rule, and complexity-value classification are **imported
and called**, so the comparison against `minimal_cox` is exactly paired and no threshold can drift.
sensitivity-analysis workflow set the precedent for importing `evaluate.workflow`'s private helpers
rather than copying
them; the same choice is made here, deliberately, because a copy is what would let the two
implementations diverge.

`minimal_cox` is loaded from the **frozen model-training registry** by primary-evaluation workflow's
own predictor and is never
refitted.

Reconstructing the preprocessor
-------------------------------
A Cox bundle carries its fitted preprocessor inside `CoxBundleState`; the forest bundle is an RDS
holding only the fitted ensemble. The preprocessor is therefore **reconstructed** at evaluation time
by refitting `FittedPreprocessor` on the same development frame — deterministic given the same rows,
so it reproduces the object used at fit time exactly. This refits a *transformation*, never a model.

A silent mismatch is not possible: the fitted forest carries its own R formula, so a column set that
differs from the one it was trained on fails loudly inside R rather than predicting from the wrong
columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.bootstrap import bootstrap_index_matrix
from crc_folfox.evaluate.contracts import (
    BOOTSTRAP_RESAMPLES,
    CENTRES,
    IBS_GRID,
    MIN_ADMISSIBLE_POOLED_REPEATS,
    EvaluationContractError,
)
from crc_folfox.evaluate.workflow import (
    ScoredMetric,
    _CellData,
    _complexity_rows,
    _predict_iecv,
    _predict_pooled,
    _score_cell,
    _transportability_rows,
    _verify_phase4,
)
from crc_folfox.forest.r_adapter import RConditionalForestAdapter
from crc_folfox.forest.workflow import ANALYSIS_STATUS, CANDIDATE, FEATURE_SET, _scope_id
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits, build_pooled_splits
from crc_folfox.preprocessing.transform import FittedPreprocessor

BENCHMARK = "minimal_cox"

#: Index of the 730-day horizon within `IBS_GRID` -- the primary horizon.
_PRIMARY_HORIZON_INDEX = IBS_GRID.index(730.0)


def _ranking(risk_grid: np.ndarray) -> np.ndarray:
    """The scalar risk ranking concordance is computed from.

    primary-evaluation workflow passes a Cox model's **linear predictor** here. A forest has none,
    so the choice is a
    real decision rather than a detail, and it is made before any forest concordance value exists:

    * Concordance is rank-based, so any monotone transform of a ranking gives the same value.
    * Under proportional hazards, a Cox model's absolute risk at any horizon is a monotone
      transform of its linear predictor. So for `minimal_cox` the 730-day risk ranking and the
      linear predictor produce **identical** concordance — the benchmark is unaffected by this
      choice, and the two models stay directly comparable.
    * For the forest no such cross-horizon equivalence holds, so its concordance is a
      **730-day-specific** ranking quantity. That is a genuine difference in meaning from the
      Cox model's horizon-free one, and it is disclosed rather than glossed.

    The alternative — leaving concordance `not_computable` for the forest — was rejected because
    `truncation_aware_concordance` accepts a risk vector and is fully delayed-entry-aware, so this
    is a well-defined LTRC quantity, not a right-censored approximation of one.
    """
    return risk_grid[:, _PRIMARY_HORIZON_INDEX]


@dataclass(frozen=True)
class ForestEvaluationResult:
    performance: pd.DataFrame
    transportability: pd.DataFrame
    complexity: pd.DataFrame
    output: Path


def _load_forest_registry(phase9: Path) -> dict[str, dict]:
    summary = pd.read_csv(phase9 / "fit_summary.csv")
    return {str(row["scope_id"]): dict(row) for _, row in summary.iterrows()}


def _forest_risk_grid(
    adapter: RConditionalForestAdapter,
    bundle: Path,
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    endpoint: str,
    scope_id: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Absolute risk over `IBS_GRID` on both scales, plus the aligned assessment metadata."""
    preprocessor = FittedPreprocessor.fit(
        development, endpoint=endpoint, feature_set=FEATURE_SET, fit_scope=scope_id
    )
    transformed = preprocessor.transform(assessment)
    survival = adapter.predict(bundle, transformed.matrix, transformed.metadata, times=IBS_GRID)
    # The forest returns survival; primary-evaluation workflow's cells hold absolute risk.
    return (
        1.0 - survival["unconditional"],
        1.0 - survival["entry_conditional"],
        transformed.metadata,
    )


def _forest_iecv_cell(
    view: EndpointView,
    split: SplitDefinition,
    registry: dict[str, dict],
    models: Path,
    adapter: RConditionalForestAdapter,
) -> _CellData:
    scope_id = _scope_id(split)
    record = registry.get(scope_id)
    if record is None or record.get("fit_status") != "fit_succeeded":
        return _CellData("not_computable", "fit_failed", None, None, None, None, None, None, None)
    development, assessment = split.frames(view)
    unconditional, conditional, metadata = _forest_risk_grid(
        adapter,
        models / Path(str(record["bundle"])).name,
        development,
        assessment,
        split.endpoint,
        scope_id,
    )
    return _CellData(
        "ok",
        "",
        tuple(metadata["patient_key"].astype(str)),
        metadata["entry"].to_numpy(float),
        metadata["exit"].to_numpy(float),
        metadata["event"].to_numpy(int),
        unconditional,
        conditional,
        _ranking(unconditional),
    )


def _forest_pooled_cell(
    view: EndpointView,
    centre: str,
    pooled_splits: tuple[SplitDefinition, ...],
    registry: dict[str, dict],
    models: Path,
    adapter: RConditionalForestAdapter,
    min_admissible_repeats: int,
) -> _CellData:
    """Out-of-fold pooled aggregation, following primary-evaluation workflow's locked rule.

    A repeat is admissible only when every one of its fold scopes fitted; a patient's pooled risk is
    the arithmetic mean, on the risk scale, over admissible repeats.
    """
    centre_frame = (
        view.data.loc[view.data["institution"].eq(centre)]
        .sort_values("patient_key", kind="stable")
        .reset_index(drop=True)
    )
    if centre_frame.empty:
        raise EvaluationContractError("Centre has no endpoint-eligible patients")
    centre_keys = tuple(centre_frame["patient_key"].astype(str))
    key_index = {key: position for position, key in enumerate(centre_keys)}

    repeats: dict[int, list[SplitDefinition]] = {}
    for split in pooled_splits:
        repeats.setdefault(split.repeat, []).append(split)

    per_repeat_u: list[np.ndarray] = []
    per_repeat_c: list[np.ndarray] = []
    total_repeats = len(repeats)

    for repeat in sorted(repeats):
        folds = sorted(repeats[repeat], key=lambda s: s.fold if s.fold is not None else -1)
        if not all(
            (registry.get(_scope_id(s)) or {}).get("fit_status") == "fit_succeeded" for s in folds
        ):
            continue
        unconditional = np.full((len(centre_keys), len(IBS_GRID)), np.nan)
        conditional = np.full((len(centre_keys), len(IBS_GRID)), np.nan)
        seen: set[str] = set()
        for split in folds:
            fold_keys = [key for key in split.assessment_keys if key in key_index]
            if not fold_keys:
                continue
            scope_id = _scope_id(split)
            development, assessment = split.frames(view)
            risk_u, risk_c, metadata = _forest_risk_grid(
                adapter,
                models / Path(str(registry[scope_id]["bundle"])).name,
                development,
                assessment,
                split.endpoint,
                scope_id,
            )
            fold_position = {
                key: index for index, key in enumerate(metadata["patient_key"].astype(str))
            }
            for key in fold_keys:
                unconditional[key_index[key]] = risk_u[fold_position[key]]
                conditional[key_index[key]] = risk_c[fold_position[key]]
                seen.add(key)
        if seen != set(centre_keys):
            raise EvaluationContractError(
                "Pooled repeat does not cover every centre patient exactly once"
            )
        per_repeat_u.append(unconditional)
        per_repeat_c.append(conditional)

    admissible = len(per_repeat_u)
    if admissible < min_admissible_repeats:
        return _CellData(
            "not_computable",
            f"insufficient_admissible_pooled_repeats_{admissible}_of_{total_repeats}",
            centre_keys,
            centre_frame["entry"].to_numpy(float),
            centre_frame["exit"].to_numpy(float),
            centre_frame["event"].to_numpy(int),
            None,
            None,
            None,
            admissible,
            total_repeats,
        )
    return _CellData(
        "ok",
        "",
        centre_keys,
        centre_frame["entry"].to_numpy(float),
        centre_frame["exit"].to_numpy(float),
        centre_frame["event"].to_numpy(int),
        np.mean(per_repeat_u, axis=0),
        np.mean(per_repeat_c, axis=0),
        _ranking(np.mean(per_repeat_u, axis=0)),
        admissible,
        total_repeats,
    )


def run_forest_evaluation_workflow(
    phase2_directory: str | Path,
    *,
    phase9_directory: str | Path | None = None,
    phase4_directory: str | Path | None = None,
    endpoint: str = "OS",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    min_admissible_pooled_repeats: int = MIN_ADMISSIBLE_POOLED_REPEATS,
    centres: tuple[str, ...] = CENTRES,
    adapter: RConditionalForestAdapter | None = None,
    progress: bool = False,
) -> ForestEvaluationResult:
    """Score the forest and the frozen benchmark on identical patients and resamples."""
    from crc_folfox.pipeline import load_clean_analysis_dataset

    phase2 = Path(phase2_directory).resolve()
    phase9 = Path(phase9_directory).resolve() if phase9_directory else phase2 / "phase9"
    phase4 = Path(phase4_directory).resolve() if phase4_directory else phase2 / "model_training"
    adapter = adapter or RConditionalForestAdapter()

    frame = load_clean_analysis_dataset(phase2 / "analysis_dataset.csv")
    cox_registry, _fit_summary, _hashes = _verify_phase4(phase4)
    forest_registry = _load_forest_registry(phase9)
    cox_models, forest_models = phase4 / "models", phase9 / "models"

    view = build_endpoint_view(frame, endpoint)
    iecv_by_centre = {s.held_out_institution: s for s in build_iecv_splits(view)}
    pooled_splits = build_pooled_splits(view)

    performance_rows: list[dict] = []
    transport_rows: list[dict] = []
    complexity_rows: list[dict] = []

    for centre in centres:
        if progress:
            print(f"  scoring {centre} ...", flush=True)
        cells: dict[tuple[str, str], _CellData] = {
            ("iecv", BENCHMARK): _predict_iecv(
                view, iecv_by_centre[centre], BENCHMARK, cox_registry, cox_models
            ),
            ("pooled", BENCHMARK): _predict_pooled(
                view,
                centre,
                BENCHMARK,
                pooled_splits,
                cox_registry,
                cox_models,
                min_admissible_pooled_repeats,
            ),
            ("iecv", CANDIDATE): _forest_iecv_cell(
                view, iecv_by_centre[centre], forest_registry, forest_models, adapter
            ),
            ("pooled", CANDIDATE): _forest_pooled_cell(
                view,
                centre,
                pooled_splits,
                forest_registry,
                forest_models,
                adapter,
                min_admissible_pooled_repeats,
            ),
        }

        n_patients = next(
            (len(cell.patient_key) for cell in cells.values() if cell.patient_key),
            int((view.data["institution"] == centre).sum()),
        )
        # Deterministic in (endpoint, centre) alone -- the same matrix primary-evaluation workflow
        # used, so the paired
        # difference is exactly paired rather than merely similarly constructed.
        index_matrix = bootstrap_index_matrix(
            endpoint, centre, n_patients, resamples=bootstrap_resamples
        )
        pair_indices = np.triu_indices(n_patients, k=1) if n_patients >= 2 else None

        scored_by_key: dict[tuple, ScoredMetric] = {}
        for (design, model_name), cell in cells.items():
            rows, scored = _score_cell(
                endpoint, design, centre, model_name, cell, index_matrix, pair_indices
            )
            performance_rows.extend(rows)
            scored_by_key.update(scored)

        transport_rows.extend(_transportability_rows(endpoint, centre, CANDIDATE, scored_by_key))
        for design in ("iecv", "pooled"):
            complexity_rows.extend(
                _complexity_rows(endpoint, design, centre, CANDIDATE, scored_by_key)
            )

    performance = pd.DataFrame(performance_rows).assign(analysis_status=ANALYSIS_STATUS)
    transportability = pd.DataFrame(transport_rows).assign(analysis_status=ANALYSIS_STATUS)
    complexity = pd.DataFrame(complexity_rows).assign(analysis_status=ANALYSIS_STATUS)

    performance.to_csv(phase9 / "performance_summary.csv", index=False)
    transportability.to_csv(phase9 / "transportability_summary.csv", index=False)
    complexity.to_csv(phase9 / "complexity_value_summary.csv", index=False)
    manifest = json.loads((phase9 / "phase9_manifest.json").read_text(encoding="utf-8"))
    manifest["evaluation"] = {
        "endpoint": endpoint,
        "centres": list(centres),
        "bootstrap_resamples": bootstrap_resamples,
        "benchmark": BENCHMARK,
        "performance_rows": int(len(performance)),
        "complexity_rows": int(len(complexity)),
    }
    (phase9 / "phase9_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ForestEvaluationResult(performance, transportability, complexity, phase9)


__all__ = ["BENCHMARK", "ForestEvaluationResult", "run_forest_evaluation_workflow"]

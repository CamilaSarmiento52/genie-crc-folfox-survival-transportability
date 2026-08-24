"""The exploratory-optimization workstream evaluation: reuse primary-evaluation workflow's
metric/bootstrap/reporting-status machinery
verbatim via its model-agnostic `_CellData`/`_score_cell` boundary
(`crc_folfox.evaluate.workflow`), rather than reimplementing any metric, bootstrap, or
reporting-status logic.

Both the IECV and pooled (OS only) designs are covered. The pooled design's out-of-fold
aggregation reuses primary-evaluation workflow's own admissibility rule and risk-scale averaging
convention
(`crc_folfox.evaluate.pooling.aggregate_pooled_predictions`'s docstring), but that function itself
is hardcoded to `CoxBundleState`/`predict_for_frame`, so it cannot run the three R-backed
candidates. `aggregate_workstream_b_pooled_predictions` below generalizes it by reusing
`build_iecv_cell`'s existing per-candidate dispatch per fold (a pooled repeat's fold split is
structurally identical to an IECV split), rather than reimplementing per-candidate prediction a
second time. This was a disclosed, deferred gap in the workflow's first implementation pass; see
`optimization-spec.md`'s 2026-08-10 amendment and `implement.md`.

Predictions come from three sources depending on candidate:

- B1 (`minimal_cox`): the frozen model-training registry, never refit -- reuses
  `crc_folfox.model_training.bundles.verify_registry_load` and
  `crc_folfox.evaluate.prediction.predict_for_frame` unchanged.
- B4/B5: Workstream B's own model-training-format Python registry (plain `lifelines.CoxPHFitter`
  bundles) -- the same two functions, pointed at the Workstream B `model_registry.json`/`models/`
  instead of model-training workflow's.
- B2/B3/B6: Workstream B's separate R registry. The R bundle has no encoding/scaling state of its
  own, so the persisted preprocessor sidecar (`workflow.py`'s `_save_preprocessor_sidecar`)
  transforms assessment rows first; predictions come from `GlmnetAdapter.predict`/
  `FlexsurvAdapter.predict`. Because glmnet/flexsurv return the survival curve directly rather
  than a baseline-curve-plus-linear-predictor decomposition, entry-conditional risk is computed by
  requesting predictions at a combined grid (`IBS_GRID` union every distinct assessment-row entry
  time) and evaluating `1 - S(t)/S(entry)` from that one batched response.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crc_folfox.evaluate.bootstrap import bootstrap_index_matrix
from crc_folfox.evaluate.contracts import (
    BOOTSTRAP_RESAMPLES,
    CENTRES,
    IBS_GRID,
    MIN_ADMISSIBLE_POOLED_REPEATS,
)
from crc_folfox.evaluate.prediction import (
    entry_conditional_risk,
    linear_predictor,
    predict_for_frame,
    unconditional_risk,
)
from crc_folfox.evaluate.workflow import (  # model-agnostic reuse boundary
    _CellData,
    _complexity_rows,
    _entry_support_rows,
    _score_cell,
    _transportability_rows,
)
from crc_folfox.model_training.bundles import CoxBundleState, RegistryRecord, verify_registry_load
from crc_folfox.model_training.contracts import ModelScope
from crc_folfox.optimize.candidates import transform_for_penalized_nonlinear_cox
from crc_folfox.optimize.contracts import ANALYSIS_STATUS, Phase5AContractError
from crc_folfox.optimize.r_adapter import FlexsurvAdapter, GlmnetAdapter
from crc_folfox.optimize.r_registry import (
    RRegistryRecord,
    verify_r_preprocessor_path,
    verify_r_registry_load,
)
from crc_folfox.optimize.workflow import PYTHON_CANDIDATE_NAMES, R_CANDIDATE_NAMES, _model_id
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits, build_pooled_splits
from crc_folfox.preprocessing.workflow import _load_verified_inputs

CANDIDATE_ORDER = (
    "minimal_cox",
    "glmnet_ridge_core",
    "glmnet_enet_core",
    "penalized_nonlinear_cox",
    "reduced_core_penalized_cox",
    "flexsurv_rp_spline",
)
FULLER_CANDIDATES = tuple(c for c in CANDIDATE_ORDER if c != "minimal_cox")

_EMPTY_CELL = _CellData("not_computable", "fit_failed", None, None, None, None, None, None, None)


def _load_json_registry(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise Phase5AContractError(f"Registry file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_r_candidate_cell(
    record: RRegistryRecord, models_directory: Path, assessment_frame: pd.DataFrame
) -> _CellData:
    """Build a `_CellData` for one R-backed candidate's IECV assessment set, on both risk scales,
    with no refitting."""
    bundle_path = verify_r_registry_load(models_directory, record)
    preprocessor_path = verify_r_preprocessor_path(models_directory, record)
    preprocessor = joblib.load(preprocessor_path)
    transformed = preprocessor.transform(assessment_frame)
    matrix, metadata = transformed.matrix, transformed.metadata
    entry = metadata["entry"].to_numpy(dtype=float)

    unique_entries = sorted({float(value) for value in entry})
    combined_grid = tuple(sorted(set(IBS_GRID) | set(unique_entries)))
    index_of = {time_value: index for index, time_value in enumerate(combined_grid)}

    if record.candidate in ("glmnet_ridge_core", "glmnet_enet_core"):
        result = GlmnetAdapter().predict(bundle_path, matrix, horizons=combined_grid)
        survival = result.survival[0]
    elif record.candidate == "flexsurv_rp_spline":
        result = FlexsurvAdapter().predict(bundle_path, matrix, horizons=combined_grid)
        survival = result.survival
    else:
        raise Phase5AContractError(f"Unknown R candidate '{record.candidate}'")

    horizon_index = [index_of[h] for h in IBS_GRID]
    unconditional = 1.0 - survival[:, horizon_index]
    entry_index = np.array([index_of[float(value)] for value in entry])
    s_entry = survival[np.arange(len(entry)), entry_index]
    s_horizon = survival[:, horizon_index]
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = 1.0 - (s_horizon / s_entry[:, None])
    eligible = entry[:, None] <= np.asarray(IBS_GRID, dtype=float)[None, :]
    conditional = np.where(eligible, conditional, np.nan)

    if (
        not np.isfinite(unconditional).all()
        or np.any(unconditional < -1e-9)
        or np.any(unconditional > 1 + 1e-9)
    ):
        raise Phase5AContractError("R candidate unconditional risk is nonfinite or out of [0, 1]")
    finite_where_eligible = np.isfinite(conditional) | ~eligible
    if not finite_where_eligible.all():
        raise Phase5AContractError("R candidate entry-conditional risk is nonfinite where eligible")

    # No closed-form linear predictor exists for a penalized/spline model the way it does for an
    # unpenalized Cox fit; the horizon-independent prognostic-index slope is therefore not
    # computed for R-backed candidates (`linear_predictor=None`) -- `_score_cell` treats a missing
    # `lp` input to `truncation_aware_concordance` as a contract error, so concordance is reported
    # `not_computable` for these three candidates rather than approximated.
    return _CellData(
        "ok",
        "",
        tuple(metadata["patient_key"].astype(str)),
        entry,
        metadata["exit"].to_numpy(dtype=float),
        metadata["event"].to_numpy(dtype=int),
        unconditional,
        conditional,
        np.full(len(entry), np.nan),
    )


def predict_penalized_nonlinear_cell(state: CoxBundleState, frame: pd.DataFrame) -> _CellData:
    """B4 (`penalized_nonlinear_cox`) prediction path.

    B4's fitted design adds `add_presentation_age_spline_interaction` after the spline basis
    (`candidates.py`'s `fit_penalized_nonlinear_cox`), so `state.feature_columns` includes the
    interaction columns. primary-evaluation workflow's frozen `transform_for_bundle` only knows
    about `spline_state`,
    not this extra step, so it cannot reconstruct a matching design matrix and `predict_for_frame`
    raises `EvaluationContractError` on a real schema mismatch. Rather than patch frozen
    primary-evaluation workflow
    code, this reproduces `predict_for_frame`'s own steps with `transform_for_penalized_nonlinear_
    cox` (B4's own, already-tested transform) in place of `transform_for_bundle`, then calls the
    same lower-level risk functions primary-evaluation workflow itself uses.
    """
    required = {"patient_key", "institution", "entry", "exit", "event"}
    if missing := required - set(frame.columns):
        raise Phase5AContractError(f"Assessment frame is missing {len(missing)} required columns")
    if state.spline_state is None:
        raise Phase5AContractError("penalized_nonlinear_cox bundle is missing its spline state")
    transformed = state.preprocessor.transform(frame)
    matrix, metadata = transformed.matrix, transformed.metadata
    design = transform_for_penalized_nonlinear_cox(matrix, state.spline_state)
    entry = metadata["entry"].to_numpy(dtype=float)
    unconditional = unconditional_risk(state, design, IBS_GRID)
    conditional = entry_conditional_risk(state, design, entry, IBS_GRID)
    lp = linear_predictor(state, design)
    return _CellData(
        "ok",
        "",
        tuple(metadata["patient_key"].astype(str)),
        entry,
        metadata["exit"].to_numpy(dtype=float),
        metadata["event"].to_numpy(dtype=int),
        unconditional,
        conditional,
        lp,
    )


def build_iecv_cell(
    candidate: str,
    view: EndpointView,
    split: SplitDefinition,
    *,
    phase4_models_dir: Path,
    phase4_registry: dict[str, dict[str, object]],
    ws_b_python_models_dir: Path,
    ws_b_python_registry: dict[str, dict[str, object]],
    ws_b_r_models_dir: Path,
    ws_b_r_registry: dict[str, dict[str, object]],
) -> _CellData:
    """Dispatch to the correct registry/prediction path for one `(candidate, split)` IECV cell."""
    _, assessment = split.frames(view)
    if candidate == "minimal_cox":
        model_id = ModelScope(split).model_id("minimal_cox")
        record_json = phase4_registry.get(model_id)
        if record_json is None or record_json.get("fit_status") != "fit_succeeded":
            return _EMPTY_CELL
        state = verify_registry_load(phase4_models_dir, RegistryRecord(**record_json))
        predictions = predict_for_frame(state, assessment, IBS_GRID)
        return _CellData(
            "ok",
            "",
            predictions.patient_key,
            predictions.entry,
            predictions.exit,
            predictions.event,
            predictions.unconditional,
            predictions.entry_conditional,
            predictions.linear_predictor,
        )
    if candidate in PYTHON_CANDIDATE_NAMES:
        model_id = _model_id(split, candidate)
        record_json = ws_b_python_registry.get(model_id)
        if record_json is None or record_json.get("fit_status") != "fit_succeeded":
            return _EMPTY_CELL
        state = verify_registry_load(ws_b_python_models_dir, RegistryRecord(**record_json))
        if candidate == "penalized_nonlinear_cox":
            return predict_penalized_nonlinear_cell(state, assessment)
        predictions = predict_for_frame(state, assessment, IBS_GRID)
        return _CellData(
            "ok",
            "",
            predictions.patient_key,
            predictions.entry,
            predictions.exit,
            predictions.event,
            predictions.unconditional,
            predictions.entry_conditional,
            predictions.linear_predictor,
        )
    if candidate in R_CANDIDATE_NAMES:
        model_id = _model_id(split, candidate)
        record_json = ws_b_r_registry.get(model_id)
        if record_json is None or record_json.get("fit_status") != "fit_succeeded":
            return _EMPTY_CELL
        record = RRegistryRecord.from_json(record_json)
        return predict_r_candidate_cell(record, ws_b_r_models_dir, assessment)
    raise Phase5AContractError(f"Unknown Workstream B candidate '{candidate}'")


def _candidate_fit_succeeded(
    split: SplitDefinition,
    candidate: str,
    *,
    phase4_registry: dict[str, dict[str, object]],
    ws_b_python_registry: dict[str, dict[str, object]],
    ws_b_r_registry: dict[str, dict[str, object]],
) -> bool:
    """Fit-status lookup only, no bundle load -- used for pooled repeat admissibility checks."""
    if candidate == "minimal_cox":
        record_json = phase4_registry.get(ModelScope(split).model_id("minimal_cox"))
    elif candidate in PYTHON_CANDIDATE_NAMES:
        record_json = ws_b_python_registry.get(_model_id(split, candidate))
    elif candidate in R_CANDIDATE_NAMES:
        record_json = ws_b_r_registry.get(_model_id(split, candidate))
    else:
        raise Phase5AContractError(f"Unknown Workstream B candidate '{candidate}'")
    return record_json is not None and record_json.get("fit_status") == "fit_succeeded"


def aggregate_workstream_b_pooled_predictions(
    view: EndpointView,
    centre: str,
    candidate: str,
    pooled_splits: tuple[SplitDefinition, ...],
    *,
    phase4_models_dir: Path,
    phase4_registry: dict[str, dict[str, object]],
    ws_b_python_models_dir: Path,
    ws_b_python_registry: dict[str, dict[str, object]],
    ws_b_r_models_dir: Path,
    ws_b_r_registry: dict[str, dict[str, object]],
    min_admissible_repeats: int = MIN_ADMISSIBLE_POOLED_REPEATS,
) -> _CellData:
    """Pooled out-of-fold aggregation for one Workstream B candidate.

    Generalizes `crc_folfox.evaluate.pooling.aggregate_pooled_predictions` (hardcoded to
    `CoxBundleState`/`predict_for_frame`, so it cannot run the R-backed candidates) across all six
    candidate types. A pooled repeat's fold split is structurally identical to an IECV split
    (`build_pooled_splits` returns the same `SplitDefinition` shape, just `design="pooled"` with
    `repeat`/`fold` set), so `build_iecv_cell`'s existing per-candidate dispatch is reused per fold
    rather than reimplementing prediction dispatch a second time -- only the repeat-admissibility
    check and the risk-scale averaging are new here, both following primary-evaluation workflow's
    own locked rule: a
    repeat is admissible only when every one of its fold-scope fits succeeded, and a patient's
    pooled risk is the arithmetic mean, on the risk scale, over all admissible repeats.
    """
    centre_frame = (
        view.data.loc[view.data["institution"].eq(centre)]
        .sort_values("patient_key", kind="stable")
        .reset_index(drop=True)
    )
    if centre_frame.empty:
        raise Phase5AContractError("Centre has no endpoint-eligible patients")
    centre_keys = tuple(centre_frame["patient_key"].astype(str))
    key_index = {key: index for index, key in enumerate(centre_keys)}

    repeats: dict[int, list[SplitDefinition]] = {}
    for split in pooled_splits:
        repeats.setdefault(split.repeat, []).append(split)

    per_repeat_unconditional: list[np.ndarray] = []
    per_repeat_conditional: list[np.ndarray] = []
    per_repeat_lp: list[np.ndarray] = []
    admissible_repeat_indices: list[int] = []

    for repeat in sorted(repeats):
        repeat_splits = sorted(
            repeats[repeat], key=lambda split: split.fold if split.fold is not None else -1
        )
        if not all(
            _candidate_fit_succeeded(
                split,
                candidate,
                phase4_registry=phase4_registry,
                ws_b_python_registry=ws_b_python_registry,
                ws_b_r_registry=ws_b_r_registry,
            )
            for split in repeat_splits
        ):
            continue

        unconditional = np.full((len(centre_keys), len(IBS_GRID)), np.nan)
        conditional = np.full((len(centre_keys), len(IBS_GRID)), np.nan)
        lp = np.full(len(centre_keys), np.nan)
        seen: set[str] = set()
        for split in repeat_splits:
            fold_keys = tuple(key for key in split.assessment_keys if key in key_index)
            if not fold_keys:
                continue
            cell = build_iecv_cell(
                candidate,
                view,
                split,
                phase4_models_dir=phase4_models_dir,
                phase4_registry=phase4_registry,
                ws_b_python_models_dir=ws_b_python_models_dir,
                ws_b_python_registry=ws_b_python_registry,
                ws_b_r_models_dir=ws_b_r_models_dir,
                ws_b_r_registry=ws_b_r_registry,
            )
            if cell.status != "ok":
                raise Phase5AContractError(
                    "Pooled fold marked fit_succeeded in the registry but its cell is "
                    "not_computable -- registry/prediction inconsistency"
                )
            fold_key_index = {key: index for index, key in enumerate(cell.patient_key)}
            for key in fold_keys:
                position, fold_position = key_index[key], fold_key_index[key]
                unconditional[position] = cell.risk_grid_unconditional[fold_position]
                conditional[position] = cell.risk_grid_conditional[fold_position]
                lp[position] = cell.linear_predictor[fold_position]
                seen.add(key)
        if seen != set(centre_keys):
            raise Phase5AContractError(
                "Pooled repeat does not cover every centre patient exactly once"
            )
        per_repeat_unconditional.append(unconditional)
        per_repeat_conditional.append(conditional)
        per_repeat_lp.append(lp)
        admissible_repeat_indices.append(repeat)

    admissible_count = len(admissible_repeat_indices)
    entry = centre_frame["entry"].to_numpy(dtype=float)
    exit_ = centre_frame["exit"].to_numpy(dtype=float)
    event = centre_frame["event"].to_numpy(dtype=int)
    if admissible_count < min_admissible_repeats:
        return _CellData(
            "not_computable",
            "insufficient_admissible_repeats",
            centre_keys,
            entry,
            exit_,
            event,
            None,
            None,
            None,
            admissible_count,
            len(repeats),
        )
    return _CellData(
        "ok",
        "",
        centre_keys,
        entry,
        exit_,
        event,
        np.nanmean(np.stack(per_repeat_unconditional), axis=0),
        np.nanmean(np.stack(per_repeat_conditional), axis=0),
        np.nanmean(np.stack(per_repeat_lp), axis=0),
        admissible_count,
        len(repeats),
    )


def workstream_b_entry_support_summary(
    phase2_directory: str | Path, *, endpoints: tuple[str, ...] = ("OS", "TTNT-D")
) -> pd.DataFrame:
    """Entry support at 84/365/730 days per endpoint/centre, for the notebook's design section.

    Reuses primary-evaluation workflow's `_entry_support_rows` unchanged -- it depends only on each
    endpoint's full
    `EndpointView` (entry/exit/event per patient), not on any candidate or fitted bundle, so it is
    identical for Workstream A/B and the frozen primary evaluation by construction.
    """
    phase2 = Path(phase2_directory).resolve()
    frame, _manifest, _hashes = _load_verified_inputs(phase2)
    rows: list[dict] = []
    for endpoint in endpoints:
        view = build_endpoint_view(frame, endpoint)
        rows.extend(_entry_support_rows(view, endpoint))
    return pd.DataFrame(rows)


def run_workstream_b_evaluation(
    phase2_directory: str | Path,
    workstream_b_directory: str | Path,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    bootstrap_resamples: int | None = None,
    min_admissible_pooled_repeats: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every `(endpoint, centre, candidate)` cell -- IECV for both endpoints, pooled for OS
    only (`optimization-spec.md`'s prospective omission of pooled for the supportive TTNT-D
    endpoint) -- and return `(performance_summary, transportability_summary,
    complexity_value_summary)`.

    `bootstrap_resamples`/`min_admissible_pooled_repeats` narrow the workload for fast synthetic
    exercises of this boundary; the real run must leave both at the primary-evaluation defaults
    (`BOOTSTRAP_RESAMPLES` 2000, `MIN_ADMISSIBLE_POOLED_REPEATS`).
    """
    phase2 = Path(phase2_directory).resolve()
    ws_b = Path(workstream_b_directory).resolve()

    frame, _manifest, _hashes = _load_verified_inputs(phase2)

    phase4_models_dir = phase2 / "model_training" / "models"
    phase4_registry = _load_json_registry(phase2 / "model_training" / "model_registry.json")
    ws_b_python_models_dir = ws_b / "models"
    ws_b_python_registry = _load_json_registry(ws_b / "model_registry.json")
    ws_b_r_models_dir = ws_b / "models"
    ws_b_r_registry = _load_json_registry(ws_b / "r_model_registry.json")

    resamples = bootstrap_resamples if bootstrap_resamples is not None else BOOTSTRAP_RESAMPLES
    min_admissible = (
        min_admissible_pooled_repeats
        if min_admissible_pooled_repeats is not None
        else MIN_ADMISSIBLE_POOLED_REPEATS
    )

    all_rows: list[dict] = []
    transport_rows: list[dict] = []
    complexity_rows: list[dict] = []

    for endpoint in endpoints:
        view = build_endpoint_view(frame, endpoint)
        iecv_by_centre = {split.held_out_institution: split for split in build_iecv_splits(view)}
        pooled_splits = build_pooled_splits(view) if endpoint == "OS" else ()
        designs = ("iecv", "pooled") if endpoint == "OS" else ("iecv",)

        for centre in CENTRES:
            iecv_split = iecv_by_centre[centre]
            _, iecv_assessment = iecv_split.frames(view)
            n_patients = len(iecv_assessment)
            index_matrix = (
                bootstrap_index_matrix(endpoint, centre, n_patients, resamples=resamples)
                if n_patients > 0
                else np.empty((resamples, 0), dtype=int)
            )
            pair_indices = np.triu_indices(n_patients, k=1) if n_patients >= 2 else None

            scored_by_candidate: dict[str, dict[tuple, object]] = {}
            for candidate in CANDIDATE_ORDER:
                cells: dict[str, _CellData] = {
                    "iecv": build_iecv_cell(
                        candidate,
                        view,
                        iecv_split,
                        phase4_models_dir=phase4_models_dir,
                        phase4_registry=phase4_registry,
                        ws_b_python_models_dir=ws_b_python_models_dir,
                        ws_b_python_registry=ws_b_python_registry,
                        ws_b_r_models_dir=ws_b_r_models_dir,
                        ws_b_r_registry=ws_b_r_registry,
                    )
                }
                if endpoint == "OS":
                    cells["pooled"] = aggregate_workstream_b_pooled_predictions(
                        view,
                        centre,
                        candidate,
                        pooled_splits,
                        phase4_models_dir=phase4_models_dir,
                        phase4_registry=phase4_registry,
                        ws_b_python_models_dir=ws_b_python_models_dir,
                        ws_b_python_registry=ws_b_python_registry,
                        ws_b_r_models_dir=ws_b_r_models_dir,
                        ws_b_r_registry=ws_b_r_registry,
                        min_admissible_repeats=min_admissible,
                    )

                scored_by_key: dict[tuple, object] = {}
                for design in designs:
                    rows, scored = _score_cell(
                        endpoint,
                        design,
                        centre,
                        candidate,
                        cells[design],
                        index_matrix,
                        pair_indices,
                    )
                    all_rows.extend(rows)
                    scored_by_key.update(scored)
                scored_by_candidate[candidate] = scored_by_key

            if endpoint == "OS":
                for candidate in CANDIDATE_ORDER:
                    transport_rows.extend(
                        _transportability_rows(
                            endpoint, centre, candidate, scored_by_candidate[candidate]
                        )
                    )

            for design in designs:
                for fuller in FULLER_CANDIDATES:
                    fuller_scored = scored_by_candidate.get(fuller, {})
                    minimal_scored = scored_by_candidate.get("minimal_cox", {})
                    combined = {**minimal_scored, **fuller_scored}
                    complexity_rows.extend(
                        _complexity_rows(endpoint, design, centre, fuller, combined)
                    )

    performance_summary = pd.DataFrame(all_rows)
    transportability_summary = pd.DataFrame(transport_rows)
    complexity_value_summary = pd.DataFrame(complexity_rows)
    # `_score_cell`/`_transportability_rows`/`_complexity_rows` are reused verbatim from
    # primary-evaluation workflow,
    # which has no reason to label its own rows -- primary-evaluation workflow *is* the primary
    # analysis. Every
    # Workstream B row is post hoc exploratory (optimization-spec.md's "Epistemic status"), so
    # this module stamps the label primary-evaluation workflow's reused functions correctly don't
    # produce themselves.
    performance_summary["analysis_status"] = ANALYSIS_STATUS
    transportability_summary["analysis_status"] = ANALYSIS_STATUS
    complexity_value_summary["analysis_status"] = ANALYSIS_STATUS
    return performance_summary, transportability_summary, complexity_value_summary

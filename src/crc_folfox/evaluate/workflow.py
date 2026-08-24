"""Atomic held-out evaluation workflow: the one supported orchestration boundary.

Ties together preflight verification (data-cleaning, preprocessing, and model-training hashes and
inventory reconciliation), bundle-
verified prediction on both risk scales for all 48 evaluation scopes, the delayed-entry metric set
with paired bootstrap uncertainty, the reporting-status rule, the transportability gap, the
complexity-value table, and atomic publication of the aggregate artifact bundle.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.bootstrap import bootstrap_index_matrix, summarize_bootstrap
from crc_folfox.evaluate.calibration import (
    calibration_curve_groups,
    horizon_calibration,
    prognostic_index_slope,
)
from crc_folfox.evaluate.censoring import fit_censoring_km, horizon_weights
from crc_folfox.evaluate.comparisons import (
    classify_complexity_cell,
    compute_complexity_gain,
    compute_transportability_gap,
)
from crc_folfox.evaluate.contracts import (
    BOOTSTRAP_RESAMPLES,
    CENTRES,
    DCA_OMISSION_REASON,
    EVAL_MODEL_NAMES,
    IBS_GRID,
    METRIC_REGISTRY,
    MIN_ADMISSIBLE_POOLED_REPEATS,
    EvaluationContractError,
)
from crc_folfox.evaluate.discrimination import time_dependent_auc, truncation_aware_concordance
from crc_folfox.evaluate.pooling import PooledAggregationResult, aggregate_pooled_predictions
from crc_folfox.evaluate.prediction import PredictionSet, predict_for_frame
from crc_folfox.evaluate.prediction_error import (
    fast_brier_from_weights,
    fast_integrated_brier_score,
    horizon_brier_point,
    integrated_brier_score,
)
from crc_folfox.evaluate.reporting import ReportingStatus, classify_reporting_status
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.model_training.contracts import ModelScope
from crc_folfox.model_training.workflow import _verify_phase3
from crc_folfox.preprocessing.contracts import EndpointView, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits, build_pooled_splits
from crc_folfox.preprocessing.workflow import _load_verified_inputs

REPORT = "evaluation_report.md"
MANIFEST = "evaluation_manifest.json"
PERFORMANCE = "performance_summary.csv"
TRANSPORT = "transportability_summary.csv"
COMPLEXITY = "complexity_value_summary.csv"
CURVES = "calibration_curves.csv"
ENTRY_SUPPORT = "entry_support_summary.csv"
FIGURES = "figures"

FULLER_MODELS = ("core_cox", "elastic_net_cox", "spline_cox")
IDX_365, IDX_730 = 0, len(IBS_GRID) - 1
ENTRY_SUPPORT_DAYS = (84.0, 365.0, 730.0)


class EvaluationWorkflowError(RuntimeError):
    """Aggregate-safe failure of the primary-evaluation workflow."""


@dataclass(frozen=True)
class EvaluationArtifacts:
    directory: Path
    evaluation_report: Path
    evaluation_manifest: Path
    performance_summary: Path
    transportability_summary: Path
    complexity_value_summary: Path
    calibration_curves: Path
    entry_support_summary: Path
    figures: Path


@dataclass(frozen=True)
class EvaluationResult:
    performance_summary: pd.DataFrame
    transportability_summary: pd.DataFrame
    complexity_value_summary: pd.DataFrame
    calibration_curves: pd.DataFrame
    entry_support_summary: pd.DataFrame
    artifacts: EvaluationArtifacts | None = None


# --- Cell prediction --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CellData:
    status: str  # "ok" or "not_computable"
    reason: str
    patient_key: tuple[str, ...] | None
    entry: np.ndarray | None
    exit: np.ndarray | None
    event: np.ndarray | None
    risk_grid_unconditional: np.ndarray | None  # (n, len(IBS_GRID))
    risk_grid_conditional: np.ndarray | None
    linear_predictor: np.ndarray | None
    admissible_repeats: int | None = None
    total_repeats: int | None = None


def _predict_iecv(
    view: EndpointView, split, model_name: str, registry: dict, models_directory: Path
) -> _CellData:
    model_id = ModelScope(split).model_id(model_name)
    record_json = registry.get(model_id)
    if record_json is None or record_json.get("fit_status") != "fit_succeeded":
        return _CellData("not_computable", "fit_failed", None, None, None, None, None, None, None)
    state = verify_registry_load(models_directory, RegistryRecord(**record_json))
    _, assessment = split.frames(view)
    predictions: PredictionSet = predict_for_frame(state, assessment, IBS_GRID)
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


def _predict_pooled(
    view: EndpointView,
    centre: str,
    model_name: str,
    pooled_splits: tuple,
    registry: dict,
    models_directory: Path,
    min_admissible_repeats: int,
) -> _CellData:
    result: PooledAggregationResult = aggregate_pooled_predictions(
        view,
        centre,
        model_name,
        pooled_splits,
        registry,
        models_directory,
        horizons=IBS_GRID,
        min_admissible_repeats=min_admissible_repeats,
    )
    if result.status != "ok":
        return _CellData(
            "not_computable",
            result.status,
            result.patient_key,
            result.entry,
            result.exit,
            result.event,
            None,
            None,
            None,
            result.admissible_repeats,
            result.total_repeats,
        )
    return _CellData(
        "ok",
        "",
        result.patient_key,
        result.entry,
        result.exit,
        result.event,
        result.unconditional,
        result.entry_conditional,
        result.linear_predictor,
        result.admissible_repeats,
        result.total_repeats,
    )


# --- Metric raw computation --------------------------------------------------------------------


@dataclass(frozen=True)
class MetricRaw:
    value: float | None
    resamples: np.ndarray
    n_eligible: int
    n_events: int
    n_event_free: int
    reason: str


def _not_computable_raw(n_resamples: int, reason: str) -> MetricRaw:
    return MetricRaw(None, np.full(n_resamples, np.nan), 0, 0, 0, reason)


# Every bootstrap-resample array a cell needs, computed together in one pass over `index_matrix`
# so the censoring KM is fit once per resample and reused, instead of once per metric per resample.
# `evaluation-spec.md`'s "Uncertainty" section requires recomputing censoring/eligibility/weights
# inside every resample; it does not require refitting the *same* KM redundantly for each of a
# cell's ~13 metrics, and doing so was the dominant real-data runtime cost (see implement.md).
_RESAMPLE_KEYS = (
    "concordance",
    "ibs",
    "auc_365",
    "auc_730",
    "brier_365",
    "brier_730",
    "citl_365_unconditional",
    "citl_730_unconditional",
    "slope_365_unconditional",
    "slope_730_unconditional",
    "citl_365_entry_conditional",
    "citl_730_entry_conditional",
    "slope_365_entry_conditional",
    "slope_730_entry_conditional",
)


def _safe(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _bootstrap_cell(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_u: np.ndarray,
    risk_c: np.ndarray,
    lp: np.ndarray,
    index_matrix: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    n_resamples = index_matrix.shape[0]
    arrays: dict[str, np.ndarray] = {key: np.full(n_resamples, np.nan) for key in _RESAMPLE_KEYS}

    for row in range(n_resamples):
        idx = index_matrix[row]
        e, x, v = entry[idx], exit_[idx], event[idx]
        try:
            km_r = fit_censoring_km(e, x, v)
        except EvaluationContractError:
            continue

        arrays["concordance"][row] = _safe(
            truncation_aware_concordance(e, x, v, lp[idx], pair_indices=pair_indices).value
        )
        with contextlib.suppress(EvaluationContractError):
            arrays["ibs"][row] = _safe(fast_integrated_brier_score(km_r, e, x, v, risk_u[idx]))

        for horizon, col, suffix in ((365.0, IDX_365, "365"), (730.0, IDX_730, "730")):
            ru, rc = risk_u[idx, col], risk_c[idx, col]
            arrays[f"auc_{suffix}"][row] = _safe(
                time_dependent_auc(km_r, e, x, v, ru, horizon).value
            )
            try:
                weights = horizon_weights(km_r, e, x, v, horizon)
                arrays[f"brier_{suffix}"][row] = _safe(fast_brier_from_weights(weights, ru))
            except EvaluationContractError:
                pass
            calibration_u = horizon_calibration(km_r, e, x, v, ru, horizon)
            arrays[f"slope_{suffix}_unconditional"][row] = _safe(calibration_u.slope)
            arrays[f"citl_{suffix}_unconditional"][row] = _safe(calibration_u.citl)
            calibration_c = horizon_calibration(km_r, e, x, v, rc, horizon)
            arrays[f"slope_{suffix}_entry_conditional"][row] = _safe(calibration_c.slope)
            arrays[f"citl_{suffix}_entry_conditional"][row] = _safe(calibration_c.citl)

    return arrays


# --- Per-scope scoring -------------------------------------------------------------------------

_ROW_SPEC = (
    ("concordance", None, "unconditional"),
    ("ibs", None, "unconditional"),
    ("auc", 365.0, "unconditional"),
    ("auc", 730.0, "unconditional"),
    ("brier", 365.0, "unconditional"),
    ("brier", 730.0, "unconditional"),
    ("citl", 365.0, "unconditional"),
    ("citl", 730.0, "unconditional"),
    ("slope", 365.0, "unconditional"),
    ("slope", 730.0, "unconditional"),
    ("citl", 365.0, "entry_conditional"),
    ("citl", 730.0, "entry_conditional"),
    ("slope", 365.0, "entry_conditional"),
    ("slope", 730.0, "entry_conditional"),
)


@dataclass(frozen=True)
class ScoredMetric:
    raw: MetricRaw
    summary: object
    status: ReportingStatus


def _score(raw: MetricRaw, metric: str) -> ScoredMetric:
    summary = summarize_bootstrap(raw.value, raw.resamples)
    status = classify_reporting_status(
        value=raw.value,
        ci_lower=summary.ci_lower,
        ci_upper=summary.ci_upper,
        n_events=raw.n_events,
        n_event_free=raw.n_event_free,
        bootstrap_finite_fraction=summary.finite_fraction,
        precision_target=METRIC_REGISTRY[metric].precision_target,
        upstream_reason=raw.reason,
    )
    return ScoredMetric(raw, summary, status)


def _performance_row(
    endpoint: str,
    design: str,
    centre: str,
    model_name: str,
    metric: str,
    horizon: float | None,
    risk_scale: str,
    scored: ScoredMetric,
    prognostic_slope: float | None,
    prognostic_reason: str,
) -> dict:
    higher = METRIC_REGISTRY[metric].higher_is_better
    return {
        "endpoint": endpoint,
        "design": design,
        "centre": centre,
        "model": model_name,
        "metric": metric,
        "horizon": horizon if horizon is not None else "",
        "risk_scale": risk_scale,
        "estimate": scored.raw.value,
        "ci_lower": scored.summary.ci_lower,
        "ci_upper": scored.summary.ci_upper,
        "n_eligible": scored.raw.n_eligible,
        "n_events": scored.raw.n_events,
        "n_event_free": scored.raw.n_event_free,
        "bootstrap_finite_fraction": scored.summary.finite_fraction,
        "reporting_status": scored.status.status,
        "reason": ",".join(scored.status.reasons),
        "higher_is_better": higher if higher is not None else "",
        "prognostic_index_slope": prognostic_slope if prognostic_slope is not None else "",
        "prognostic_index_slope_reason": prognostic_reason,
    }


def _score_cell(
    endpoint: str,
    design: str,
    centre: str,
    model_name: str,
    cell: _CellData,
    index_matrix: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[list[dict], dict[tuple, ScoredMetric]]:
    scored_by_key: dict[tuple, ScoredMetric] = {}
    rows: list[dict] = []
    n_resamples = index_matrix.shape[0]

    if cell.status != "ok":
        for metric, horizon, risk_scale in _ROW_SPEC:
            raw = _not_computable_raw(n_resamples, cell.reason)
            scored = _score(raw, metric)
            scored_by_key[(design, model_name, metric, horizon, risk_scale)] = scored
            rows.append(
                _performance_row(
                    endpoint,
                    design,
                    centre,
                    model_name,
                    metric,
                    horizon,
                    risk_scale,
                    scored,
                    None,
                    "",
                )
            )
        return rows, scored_by_key

    entry, exit_, event = cell.entry, cell.exit, cell.event
    risk_u, risk_c, lp = (
        cell.risk_grid_unconditional,
        cell.risk_grid_conditional,
        cell.linear_predictor,
    )
    km = fit_censoring_km(entry, exit_, event)
    resamples = _bootstrap_cell(entry, exit_, event, risk_u, risk_c, lp, index_matrix, pair_indices)

    def _emit(
        metric: str,
        horizon: float | None,
        risk_scale: str,
        raw: MetricRaw,
        pis_value=None,
        pis_reason="",
    ) -> None:
        scored = _score(raw, metric)
        scored_by_key[(design, model_name, metric, horizon, risk_scale)] = scored
        rows.append(
            _performance_row(
                endpoint,
                design,
                centre,
                model_name,
                metric,
                horizon,
                risk_scale,
                scored,
                pis_value,
                pis_reason,
            )
        )

    conc_point = truncation_aware_concordance(entry, exit_, event, lp, pair_indices=pair_indices)
    n_events_overall, n_event_free_overall = int(event.sum()), int((event == 0).sum())
    conc_raw = MetricRaw(
        conc_point.value,
        resamples["concordance"],
        len(entry),
        n_events_overall,
        n_event_free_overall,
        "" if conc_point.value is not None else "no_comparable_pairs",
    )
    pis_value, pis_reason = prognostic_index_slope(entry, exit_, event, lp)
    _emit("concordance", None, "unconditional", conc_raw, pis_value, pis_reason)

    ibs_point = integrated_brier_score(entry, exit_, event, risk_u)
    ibs_weights = horizon_weights(km, entry, exit_, event, IBS_GRID[-1])
    ibs_raw = MetricRaw(
        ibs_point.value,
        resamples["ibs"],
        int(ibs_weights.eligible.sum()),
        int(ibs_weights.occurred.sum()),
        int(ibs_weights.survived.sum()),
        ibs_point.reason if ibs_point.value is None else "",
    )
    _emit("ibs", None, "unconditional", ibs_raw)

    for horizon, col, suffix in ((365.0, IDX_365, "365"), (730.0, IDX_730, "730")):
        weights = horizon_weights(km, entry, exit_, event, horizon)
        n_eligible, n_events, n_event_free = (
            int(weights.eligible.sum()),
            int(weights.occurred.sum()),
            int(weights.survived.sum()),
        )

        auc_point = time_dependent_auc(km, entry, exit_, event, risk_u[:, col], horizon)
        auc_raw = MetricRaw(
            auc_point.value,
            resamples[f"auc_{suffix}"],
            n_eligible,
            n_events,
            n_event_free,
            "" if auc_point.value is not None else "insufficient_case_or_control",
        )
        _emit("auc", horizon, "unconditional", auc_raw)

        brier_point = horizon_brier_point(entry, exit_, event, risk_u[:, col], horizon)
        brier_raw = MetricRaw(
            brier_point.score,
            resamples[f"brier_{suffix}"],
            n_eligible,
            n_events,
            n_event_free,
            brier_point.reason if brier_point.score is None else "",
        )
        _emit("brier", horizon, "unconditional", brier_raw)

        calibration_u = horizon_calibration(km, entry, exit_, event, risk_u[:, col], horizon)
        calibration_c = horizon_calibration(km, entry, exit_, event, risk_c[:, col], horizon)
        for which, calibration_result, risk_scale in (
            ("unconditional", calibration_u, "unconditional"),
            ("entry_conditional", calibration_c, "entry_conditional"),
        ):
            for metric_name, value in (
                ("slope", calibration_result.slope),
                ("citl", calibration_result.citl),
            ):
                raw = MetricRaw(
                    value,
                    resamples[f"{metric_name}_{suffix}_{which}"],
                    calibration_result.n_eligible,
                    calibration_result.n_events,
                    calibration_result.n_eligible - calibration_result.n_events,
                    calibration_result.reason if value is None else "",
                )
                _emit(metric_name, horizon, risk_scale, raw)

    return rows, scored_by_key


def _curve_rows(
    endpoint: str, design: str, centre: str, model_name: str, cell: _CellData
) -> list[dict]:
    if cell.status != "ok":
        return []
    rows = []
    for horizon, col in ((365.0, IDX_365), (730.0, IDX_730)):
        for risk_scale, risk_grid in (
            ("unconditional", cell.risk_grid_unconditional),
            ("entry_conditional", cell.risk_grid_conditional),
        ):
            result = calibration_curve_groups(
                cell.entry, cell.exit, cell.event, risk_grid[:, col], horizon
            )
            if result.status != "ok":
                continue
            for group in result.groups:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "design": design,
                        "centre": centre,
                        "model": model_name,
                        "horizon": horizon,
                        "risk_scale": risk_scale,
                        "group": group.group,
                        "n": group.n,
                        "n_events": group.n_events,
                        "mean_predicted_risk": group.mean_predicted_risk,
                        "observed_risk": group.observed_risk
                        if group.observed_risk is not None
                        else "",
                        "observed_risk_se": group.observed_risk_se
                        if group.observed_risk_se is not None
                        else "",
                        "low_support": group.low_support,
                    }
                )
    return rows


# --- Transportability and complexity ------------------------------------------------------------

_TRANSPORT_SPEC = (
    ("concordance", None),
    ("ibs", None),
    ("auc", 365.0),
    ("auc", 730.0),
    ("brier", 365.0),
    ("brier", 730.0),
    ("citl", 365.0),
    ("citl", 730.0),
    ("slope", 365.0),
    ("slope", 730.0),
)


def _transportability_rows(
    endpoint: str, centre: str, model_name: str, scored_by_key: dict[tuple, ScoredMetric]
) -> list[dict]:
    rows = []
    for metric, horizon in _TRANSPORT_SPEC:
        pooled = scored_by_key[("pooled", model_name, metric, horizon, "unconditional")]
        iecv = scored_by_key[("iecv", model_name, metric, horizon, "unconditional")]
        higher = METRIC_REGISTRY[metric].higher_is_better
        gap = compute_transportability_gap(
            pooled_point=pooled.raw.value,
            pooled_resamples=pooled.raw.resamples,
            iecv_point=iecv.raw.value,
            iecv_resamples=iecv.raw.resamples,
            higher_is_better=bool(higher) if higher is not None else False,
        )
        rows.append(
            {
                "endpoint": endpoint,
                "centre": centre,
                "model": model_name,
                "metric": metric,
                "horizon": horizon if horizon is not None else "",
                "higher_is_better": higher if higher is not None else "",
                "gap": gap.value,
                "gap_ci_lower": gap.ci_lower,
                "gap_ci_upper": gap.ci_upper,
                "gap_reason": gap.reason,
                "pooled_estimate": pooled.raw.value,
                "pooled_ci_lower": pooled.summary.ci_lower,
                "pooled_ci_upper": pooled.summary.ci_upper,
                "pooled_status": pooled.status.status,
                "pooled_n_eligible": pooled.raw.n_eligible,
                "pooled_n_events": pooled.raw.n_events,
                "iecv_estimate": iecv.raw.value,
                "iecv_ci_lower": iecv.summary.ci_lower,
                "iecv_ci_upper": iecv.summary.ci_upper,
                "iecv_status": iecv.status.status,
                "iecv_n_eligible": iecv.raw.n_eligible,
                "iecv_n_events": iecv.raw.n_events,
            }
        )
    return rows


_GAIN_SPEC = (
    ("auc", "auc", 365.0),
    ("auc", "auc", 730.0),
    ("ibs", "ibs", None),
    ("citl", "citl", 365.0),
    ("citl", "citl", 730.0),
    ("slope", "slope", 365.0),
    ("slope", "slope", 730.0),
)


def _complexity_rows(
    endpoint: str,
    design: str,
    centre: str,
    fuller_model: str,
    scored_by_key: dict[tuple, ScoredMetric],
) -> list[dict]:
    gains = {}
    for kind, metric, horizon in _GAIN_SPEC:
        fuller = scored_by_key[(design, fuller_model, metric, horizon, "unconditional")]
        minimal = scored_by_key[(design, "minimal_cox", metric, horizon, "unconditional")]
        gains[f"{kind}_{horizon}"] = compute_complexity_gain(
            kind,
            fuller_point=fuller.raw.value,
            fuller_resamples=fuller.raw.resamples,
            minimal_point=minimal.raw.value,
            minimal_resamples=minimal.raw.resamples,
        )
    classification = classify_complexity_cell(gains)
    rows = []
    for (kind, metric, horizon), (_gain_key, gain) in zip(_GAIN_SPEC, gains.items(), strict=True):
        fuller = scored_by_key[(design, fuller_model, metric, horizon, "unconditional")]
        minimal = scored_by_key[(design, "minimal_cox", metric, horizon, "unconditional")]
        rows.append(
            {
                "endpoint": endpoint,
                "design": design,
                "centre": centre,
                "fuller_model": fuller_model,
                "gain": kind,
                "horizon": horizon if horizon is not None else "",
                "value": gain.value,
                "ci_lower": gain.ci_lower,
                "ci_upper": gain.ci_upper,
                "reason": gain.reason,
                "fuller_estimate": fuller.raw.value,
                "minimal_estimate": minimal.raw.value,
                "net_benefit_gain": "not_applicable",
                "net_benefit_gain_reason": DCA_OMISSION_REASON,
                "classification": classification,
            }
        )
    return rows


# --- Entry support -------------------------------------------------------------------------------


def _entry_support_rows(view: EndpointView, endpoint: str) -> list[dict]:
    rows = []
    for group in (*CENTRES, "All"):
        data = view.data if group == "All" else view.data.loc[view.data["institution"].eq(group)]
        entry = data["entry"].to_numpy(dtype=float)
        exit_ = data["exit"].to_numpy(dtype=float)
        event = data["event"].to_numpy(dtype=int)
        for day in ENTRY_SUPPORT_DAYS:
            eligible = entry <= day
            n_eligible = int(eligible.sum())
            n_events = int(((exit_ <= day) & (event == 1) & eligible).sum())
            n_event_free = int(((exit_ > day) & eligible).sum())
            n_censored_before = n_eligible - n_events - n_event_free
            rows.append(
                {
                    "endpoint": endpoint,
                    "group": group,
                    "day": day,
                    "n_eligible": n_eligible,
                    "n_events": n_events,
                    "n_event_free": n_event_free,
                    "n_censored_before": n_censored_before,
                }
            )
    return rows


# --- Orchestration ---------------------------------------------------------------------------


def _pooled_splits(view: EndpointView, pooled_repeats: int | None) -> tuple:
    if pooled_repeats is None:
        return build_pooled_splits(view)
    return build_pooled_splits(view, repeats=pooled_repeats)


def run_evaluation_workflow(
    phase2_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    phase4_directory: str | Path | None = None,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    pooled_repeats: int | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    min_admissible_pooled_repeats: int = MIN_ADMISSIBLE_POOLED_REPEATS,
) -> EvaluationResult:
    """Evaluate every registered trained model on the untouched assessment rows of its scope.

    `pooled_repeats`/`bootstrap_resamples`/`min_admissible_pooled_repeats` narrow the workload for
    fast synthetic exercises of this boundary; the notebook must call this with all defaults left
    untouched.

    `model_training_directory` defaults to the canonical analysis directory's `model_training/`
    subdirectory, exactly as before this
    parameter existed. It exists only so the technical-correction workstream
    can evaluate a corrected model-training bundle from a separate output root
    without touching, moving, or shadowing the frozen primary `model_training/` directory -- the
    real primary evaluation always calls this with the default.
    """
    phase2 = Path(phase2_directory).resolve()
    try:
        frame, _phase2_manifest, input_hashes = _load_verified_inputs(phase2)
        _verify_phase3(phase2 / "preprocessing", frame)
        phase4_dir = (
            Path(phase4_directory).resolve() if phase4_directory else phase2 / "model_training"
        )
        registry, fit_summary, phase4_hashes = _verify_phase4(phase4_dir)
        input_hashes = {**input_hashes, **phase4_hashes}
        models_directory = phase4_dir / "models"

        performance_rows: list[dict] = []
        transport_rows: list[dict] = []
        complexity_rows: list[dict] = []
        curve_rows: list[dict] = []
        support_rows: list[dict] = []

        for endpoint in endpoints:
            view = build_endpoint_view(frame, endpoint)
            iecv_by_centre = {
                split.held_out_institution: split for split in build_iecv_splits(view)
            }
            pooled_splits = _pooled_splits(view, pooled_repeats)
            support_rows.extend(_entry_support_rows(view, endpoint))

            for centre in CENTRES:
                cells: dict[tuple[str, str], _CellData] = {}
                for model_name in EVAL_MODEL_NAMES:
                    cells[("iecv", model_name)] = _predict_iecv(
                        view, iecv_by_centre[centre], model_name, registry, models_directory
                    )
                    cells[("pooled", model_name)] = _predict_pooled(
                        view,
                        centre,
                        model_name,
                        pooled_splits,
                        registry,
                        models_directory,
                        min_admissible_pooled_repeats,
                    )

                n_patients = next(
                    (len(cell.patient_key) for cell in cells.values() if cell.patient_key),
                    int((view.data["institution"] == centre).sum()),
                )
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
                    curve_rows.extend(_curve_rows(endpoint, design, centre, model_name, cell))

                for model_name in EVAL_MODEL_NAMES:
                    transport_rows.extend(
                        _transportability_rows(endpoint, centre, model_name, scored_by_key)
                    )
                for design in ("iecv", "pooled"):
                    for fuller_model in FULLER_MODELS:
                        complexity_rows.extend(
                            _complexity_rows(endpoint, design, centre, fuller_model, scored_by_key)
                        )

        performance = pd.DataFrame(performance_rows)
        transportability = pd.DataFrame(transport_rows)
        complexity = pd.DataFrame(complexity_rows)
        curves = pd.DataFrame(curve_rows) if curve_rows else pd.DataFrame(columns=_CURVE_COLUMNS)
        entry_support = pd.DataFrame(support_rows)

        output = Path(output_directory).resolve() if output_directory else phase2 / "evaluation"
        artifacts = _publish(
            output,
            performance,
            transportability,
            complexity,
            curves,
            entry_support,
            input_hashes,
        )
        _assert_unchanged(input_hashes)
        return EvaluationResult(
            performance, transportability, complexity, curves, entry_support, artifacts
        )
    except (EvaluationContractError, EvaluationWorkflowError):
        raise
    except Exception as exc:
        raise EvaluationWorkflowError("Phase 5 evaluation workflow failed") from exc


_CURVE_COLUMNS = (
    "endpoint",
    "design",
    "centre",
    "model",
    "horizon",
    "risk_scale",
    "group",
    "n",
    "n_events",
    "mean_predicted_risk",
    "observed_risk",
    "observed_risk_se",
    "low_support",
)


def _verify_phase4(phase4_dir: Path) -> tuple[dict, pd.DataFrame, dict[str, str]]:
    required = [
        "training_report.md",
        "training_manifest.json",
        "fit_summary.csv",
        "model_registry.json",
    ]
    if not phase4_dir.is_dir() or any(not (phase4_dir / name).is_file() for name in required):
        raise EvaluationWorkflowError("Verified Phase 4 handoff is unavailable")
    manifest = json.loads((phase4_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != "phase-4-delayed-entry-model-training":
        raise EvaluationWorkflowError("Phase 4 manifest has an unexpected phase identifier")
    hashes: dict[str, str] = {}
    for name, digest in manifest.get("artifact_hashes", {}).items():
        path = phase4_dir / name
        actual = _sha256(path)
        if actual != digest:
            raise EvaluationWorkflowError("Phase 4 artifact hash mismatch")
        hashes[str(path)] = actual
    fit_summary = pd.read_csv(phase4_dir / "fit_summary.csv")
    registry = json.loads((phase4_dir / "model_registry.json").read_text(encoding="utf-8"))
    for record in registry.values():
        if record.get("fit_status") == "fit_succeeded":
            path = phase4_dir / "models" / str(record["bundle_relative_path"])
            if _sha256(path) != record["per_file_sha256"]:
                raise EvaluationWorkflowError("Registered Phase 4 bundle hash mismatch")
    return registry, fit_summary, hashes


def _publish(
    output: Path,
    performance: pd.DataFrame,
    transportability: pd.DataFrame,
    complexity: pd.DataFrame,
    curves: pd.DataFrame,
    entry_support: pd.DataFrame,
    input_hashes: dict[str, str],
) -> EvaluationArtifacts:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase5-evaluation-", dir=output.parent))
    backup = output.parent / f".{output.name}-previous"
    try:
        performance.to_csv(temporary / PERFORMANCE, index=False)
        transportability.to_csv(temporary / TRANSPORT, index=False)
        complexity.to_csv(temporary / COMPLEXITY, index=False)
        curves.to_csv(temporary / CURVES, index=False)
        entry_support.to_csv(temporary / ENTRY_SUPPORT, index=False)
        figures_dir = temporary / FIGURES
        figures_dir.mkdir()
        _render_figures(curves, figures_dir)

        report = _render_report(performance, transportability, complexity)
        (temporary / REPORT).write_text(report, encoding="utf-8")

        artifact_names = (REPORT, PERFORMANCE, TRANSPORT, COMPLEXITY, CURVES, ENTRY_SUPPORT)
        hashes = {name: _sha256(temporary / name) for name in artifact_names}
        payload: dict[str, object] = {
            "phase": "phase-5-held-out-evaluation",
            "input_hashes": {Path(path).name: digest for path, digest in input_hashes.items()},
            "software_versions": {
                "lifelines": version("lifelines"),
                "statsmodels": version("statsmodels"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
            },
            "artifact_hashes": hashes,
            "row_counts": {
                PERFORMANCE: int(len(performance)),
                TRANSPORT: int(len(transportability)),
                COMPLEXITY: int(len(complexity)),
                CURVES: int(len(curves)),
                ENTRY_SUPPORT: int(len(entry_support)),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        (temporary / MANIFEST).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        temporary.rename(output)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if not output.exists() and backup.exists():
            backup.rename(output)
        if isinstance(exc, EvaluationWorkflowError):
            raise
        raise EvaluationWorkflowError("Phase 5 atomic publication failed") from exc

    try:
        _verify_published(output)
    except Exception as exc:
        rolled_back = output.parent / f".{output.name}-rolledback"
        if rolled_back.exists():
            shutil.rmtree(rolled_back)
        output.rename(rolled_back)
        if backup.exists():
            backup.rename(output)
        shutil.rmtree(rolled_back, ignore_errors=True)
        raise EvaluationWorkflowError("Phase 5 post-publication verification failed") from exc

    if backup.exists():
        shutil.rmtree(backup)
    return EvaluationArtifacts(
        output,
        output / REPORT,
        output / MANIFEST,
        output / PERFORMANCE,
        output / TRANSPORT,
        output / COMPLEXITY,
        output / CURVES,
        output / ENTRY_SUPPORT,
        output / FIGURES,
    )


def _verify_published(output: Path) -> None:
    required = {
        REPORT,
        MANIFEST,
        PERFORMANCE,
        TRANSPORT,
        COMPLEXITY,
        CURVES,
        ENTRY_SUPPORT,
        FIGURES,
    }
    if {path.name for path in output.iterdir()} != required:
        raise EvaluationWorkflowError("Published Phase 5 bundle has unexpected contents")


def _render_figures(curves: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if curves.empty:
        return
    for (endpoint, design, horizon), group in curves.groupby(["endpoint", "design", "horizon"]):
        subset = group[group["risk_scale"] == "unconditional"]
        if subset.empty:
            continue
        centres = sorted(subset["centre"].unique())
        models = sorted(subset["model"].unique())
        fig, axes = plt.subplots(
            len(centres),
            len(models),
            figsize=(3.2 * len(models), 3.0 * len(centres)),
            squeeze=False,
        )
        for row_index, centre in enumerate(centres):
            for col_index, model in enumerate(models):
                ax = axes[row_index][col_index]
                cell = subset[(subset["centre"] == centre) & (subset["model"] == model)]
                if not cell.empty:
                    x = pd.to_numeric(cell["mean_predicted_risk"], errors="coerce")
                    y = pd.to_numeric(cell["observed_risk"], errors="coerce")
                    ax.scatter(x, y, s=15)
                    limit = max(1.0, float(np.nanmax([x.max(), y.max(), 0.1])))
                    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color="gray")
                ax.set_title(f"{centre}/{model}", fontsize=7)
                ax.set_xlabel("predicted", fontsize=6)
                ax.set_ylabel("observed", fontsize=6)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"{endpoint} {design} {int(horizon)}d calibration")
        fig.tight_layout()
        fig.savefig(figures_dir / f"calibration_{endpoint}_{design}_{int(horizon)}.png", dpi=110)
        plt.close(fig)


def _render_report(
    performance: pd.DataFrame, transportability: pd.DataFrame, complexity: pd.DataFrame
) -> str:
    status_counts = (
        performance["reporting_status"].value_counts().to_dict() if len(performance) else {}
    )
    classification_counts = (
        complexity.drop_duplicates(["endpoint", "design", "centre", "fuller_model"])[
            "classification"
        ]
        .value_counts()
        .to_dict()
        if len(complexity)
        else {}
    )
    return (
        "# Phase 5 evaluation report\n\n"
        "Evaluation artifacts are aggregate-safe. No model was refit, recalibrated, or "
        "explained.\n\n"
        "## Reporting-status counts (performance_summary.csv)\n\n"
        f"{status_counts}\n\n"
        "## Complexity-value classification counts\n\n"
        f"{classification_counts}\n\n"
        "## Decision-curve analysis\n\n"
        f"Prospectively omitted (`{DCA_OMISSION_REASON}`): no prespecified clinical action or "
        "threshold range is supplied, and the 84-day risk set has only four eligible deaths "
        "overall. `net_benefit_gain` is `not_applicable` in every complexity-value row.\n\n"
        "## Phase 6 handoff\n\nPhase 6 must reconstruct any patient-level prediction it needs from "
        "the verified inputs; no explanation, attribution, prediction-stability, recalibration, or "
        "sensitivity result was produced here.\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_unchanged(hashes: dict[str, str]) -> None:
    if any(_sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise EvaluationWorkflowError("Phase 5 mutated a canonical input")


__all__ = [
    "EvaluationArtifacts",
    "EvaluationResult",
    "EvaluationWorkflowError",
    "run_evaluation_workflow",
]

"""Atomic sensitivity-analysis workflow: the one supported orchestration boundary.

Reconstructs `minimal_cox`/`spline_cox` predictions exactly as primary-evaluation workflow's own
evaluation does (by
reusing its private cell-prediction functions, the same cross-module reuse pattern every later
workflow
already uses for `_load_verified_inputs`/`the preprocessing verifier`), then runs all six
sensitivities and
publishes one atomic aggregate bundle. No primary model is refit; every artifact is reported
alongside the frozen model-training and primary-evaluation result, never in place of it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from crc_folfox.evaluate.bootstrap import bootstrap_index_matrix
from crc_folfox.evaluate.contracts import (
    BOOTSTRAP_RESAMPLES,
    CENTRES,
    MIN_ADMISSIBLE_POOLED_REPEATS,
)
from crc_folfox.evaluate.workflow import (
    EvaluationWorkflowError,
    _CellData,
    _pooled_splits,
    _predict_iecv,
    _predict_pooled,
    _verify_phase4,
)
from crc_folfox.model_training.workflow import _verify_phase3
from crc_folfox.preprocessing.contracts import EndpointView, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits
from crc_folfox.preprocessing.workflow import _load_verified_inputs
from crc_folfox.sensitivity.calibration_drift import calibration_drift_steps, local_adaptation_rows
from crc_folfox.sensitivity.complete_case import (
    complete_case_comparison_rows,
    complete_case_flag_frame,
    complete_case_mask_for,
)
from crc_folfox.sensitivity.contracts import SENSITIVITY_MODEL_NAMES, SensitivityContractError
from crc_folfox.sensitivity.endpoint_robustness import (
    classify_concordance,
)
from crc_folfox.sensitivity.no_information import no_information_brier_rows, scaled_brier_rows
from crc_folfox.sensitivity.plots import (
    render_calibration_drift,
    render_complete_case_forest,
    render_endpoint_robustness,
    render_local_adaptation_curve,
    render_scaled_brier,
    render_sequencing_lag_headline,
    render_truncation_diagnostic,
)
from crc_folfox.sensitivity.sequencing_lag import (
    lag_km_curve_rows,
    lag_km_rows,
    lag_pooled_minimal_cox_rows,
    lag_support_rows,
)
from crc_folfox.sensitivity.truncation import (
    entry_at_or_after_exit_summary,
    entry_time_distribution_rows,
    km_by_entry_quartile_rows,
    truncation_kendall_tau_rows,
)

REPORT = "sensitivity_report.md"
MANIFEST = "sensitivity_manifest.json"
SEQUENCING_LAG_SUPPORT = "sequencing_lag_support.csv"
SEQUENCING_LAG_KM = "sequencing_lag_km.csv"
SEQUENCING_LAG_KM_CURVE = "sequencing_lag_km_curve.csv"
SEQUENCING_LAG_POOLED = "sequencing_lag_pooled_minimal_cox.csv"
TRUNCATION_TAU = "truncation_kendall_tau.csv"
ENTRY_TIME_DISTRIBUTION = "entry_time_distribution.csv"
KM_BY_ENTRY_QUARTILE = "delayed_entry_survival_by_entry_quartile.csv"
ENTRY_EXCLUSION_SUMMARY = "entry_at_or_after_exit_summary.csv"
COMPLETE_CASE = "complete_case_assessment_subset.csv"
ENDPOINT_ROBUSTNESS = "supportive_endpoint_robustness.csv"
CALIBRATION_DRIFT = "calibration_drift.csv"
LOCAL_RECALIBRATION = "local_recalibration.csv"
SCALED_BRIER = "scaled_brier.csv"
FIGURES = "figures"

_BRIER_KEYS = ("brier_365", "brier_730", "ibs")


class SensitivityWorkflowError(RuntimeError):
    """Aggregate-safe failure of the sensitivity-analysis workflow."""


@dataclass(frozen=True)
class SensitivityArtifacts:
    directory: Path


@dataclass(frozen=True)
class SensitivityResult:
    sequencing_lag_support: pd.DataFrame
    sequencing_lag_km: pd.DataFrame
    sequencing_lag_km_curve: pd.DataFrame
    sequencing_lag_pooled_minimal_cox: pd.DataFrame
    truncation_kendall_tau: pd.DataFrame
    entry_time_distribution: pd.DataFrame
    km_by_entry_quartile: pd.DataFrame
    entry_exclusion_summary: pd.DataFrame
    complete_case: pd.DataFrame
    endpoint_robustness: pd.DataFrame
    calibration_drift: pd.DataFrame
    local_recalibration: pd.DataFrame
    scaled_brier: pd.DataFrame
    artifacts: SensitivityArtifacts | None = None


def _predict_cell(
    view: EndpointView,
    design: str,
    centre: str,
    model_name: str,
    *,
    iecv_by_centre: dict,
    pooled_splits: tuple,
    registry: dict,
    models_directory: Path,
    min_admissible_pooled_repeats: int,
) -> _CellData:
    if design == "iecv":
        return _predict_iecv(view, iecv_by_centre[centre], model_name, registry, models_directory)
    return _predict_pooled(
        view,
        centre,
        model_name,
        pooled_splits,
        registry,
        models_directory,
        min_admissible_pooled_repeats,
    )


def run_sensitivity_workflow(
    phase2_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    model_names: tuple[str, ...] = SENSITIVITY_MODEL_NAMES,
    pooled_repeats: int | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    tau_resamples: int = 2000,
    min_admissible_pooled_repeats: int = MIN_ADMISSIBLE_POOLED_REPEATS,
) -> SensitivityResult:
    """Run every sensitivity analysis against the frozen `minimal_cox`/`spline_cox` bundles.

    `bootstrap_resamples`/`tau_resamples`/`pooled_repeats` narrow the workload for fast synthetic
    exercises of this boundary; the notebook must call this with all defaults left untouched.
    """
    if any(name not in SENSITIVITY_MODEL_NAMES for name in model_names):
        raise SensitivityContractError("Unknown sensitivity model requested")
    phase2 = Path(phase2_directory).resolve()
    try:
        frame, _phase2_manifest, input_hashes = _load_verified_inputs(phase2)
        _verify_phase3(phase2 / "preprocessing", frame)
        phase4_dir = phase2 / "model_training"
        registry, _fit_summary, phase4_hashes = _verify_phase4(phase4_dir)
        input_hashes = {**input_hashes, **phase4_hashes}
        models_directory = phase4_dir / "models"

        complete_case_flags = complete_case_flag_frame(frame)

        views: dict[str, EndpointView] = {}
        lag_support_frames: list[pd.DataFrame] = []
        lag_km_frames: list[pd.DataFrame] = []
        lag_km_curve_frames: list[pd.DataFrame] = []
        lag_pooled_frames: list[pd.DataFrame] = []
        tau_frames: list[pd.DataFrame] = []
        entry_dist_frames: list[pd.DataFrame] = []
        km_quartile_frames: list[pd.DataFrame] = []
        complete_case_rows: list[dict] = []
        calibration_rows: list[dict] = []
        local_adaptation_frames: list[dict] = []
        scaled_brier_all: list[dict] = []

        truncation_conclusions: dict[str, str] = {}
        for endpoint in endpoints:
            view = build_endpoint_view(frame, endpoint)
            views[endpoint] = view

            lag_support_frames.append(pd.DataFrame(lag_support_rows(view, endpoint)))
            lag_km_frames.append(pd.DataFrame(lag_km_rows(view, endpoint)))
            lag_km_curve_frames.append(pd.DataFrame(lag_km_curve_rows(view, endpoint)))
            tau_rows, conclusion = truncation_kendall_tau_rows(
                view, endpoint, resamples=tau_resamples
            )
            truncation_conclusions[endpoint] = conclusion
            for row in tau_rows:
                row["conclusion"] = conclusion
            tau_frames.append(pd.DataFrame(tau_rows))
            entry_dist_frames.append(pd.DataFrame(entry_time_distribution_rows(view, endpoint)))
            km_quartile_frames.append(pd.DataFrame(km_by_entry_quartile_rows(view, endpoint)))

            iecv_by_centre = {
                split.held_out_institution: split for split in build_iecv_splits(view)
            }
            pooled_splits = _pooled_splits(view, pooled_repeats)

            for centre in CENTRES:
                for model_name in model_names:
                    for design in ("iecv", "pooled"):
                        cell = _predict_cell(
                            view,
                            design,
                            centre,
                            model_name,
                            iecv_by_centre=iecv_by_centre,
                            pooled_splits=pooled_splits,
                            registry=registry,
                            models_directory=models_directory,
                            min_admissible_pooled_repeats=min_admissible_pooled_repeats,
                        )
                        if cell.status != "ok":
                            continue

                        index_matrix = bootstrap_index_matrix(
                            endpoint, centre, len(cell.patient_key), resamples=bootstrap_resamples
                        )

                        cc_mask = complete_case_mask_for(complete_case_flags, cell.patient_key)
                        cell_cc_rows = complete_case_comparison_rows(
                            endpoint=endpoint,
                            design=design,
                            centre=centre,
                            model_name=model_name,
                            entry=cell.entry,
                            exit_=cell.exit,
                            event=cell.event,
                            risk_unconditional=cell.risk_grid_unconditional,
                            linear_predictor=cell.linear_predictor,
                            complete_case_mask=cc_mask,
                            index_matrix=index_matrix,
                        )
                        complete_case_rows.extend(cell_cc_rows)

                        if model_name == "minimal_cox" and design == "pooled":
                            lag_pooled_frames.append(
                                pd.DataFrame(
                                    lag_pooled_minimal_cox_rows(
                                        endpoint=endpoint,
                                        centre=centre,
                                        entry=cell.entry,
                                        exit_=cell.exit,
                                        event=cell.event,
                                        risk_unconditional=cell.risk_grid_unconditional,
                                        linear_predictor=cell.linear_predictor,
                                        index_matrix=index_matrix,
                                    )
                                )
                            )

                        no_info = no_information_brier_rows(
                            endpoint, centre, cell.entry, cell.exit, cell.event
                        )
                        model_brier_by_key = {
                            row["metric"]: row["full_estimate"]
                            for row in cell_cc_rows
                            if row["metric"] in _BRIER_KEYS
                        }
                        scaled_brier_all.extend(
                            scaled_brier_rows(
                                endpoint=endpoint,
                                design=design,
                                centre=centre,
                                model_name=model_name,
                                model_brier_by_key=model_brier_by_key,
                                no_information_brier_by_key=no_info,
                            )
                        )

                        if design == "iecv":
                            steps = calibration_drift_steps(
                                cell.entry, cell.exit, cell.event, cell.risk_grid_unconditional
                            )
                            for step in steps:
                                calibration_rows.append(
                                    {
                                        "endpoint": endpoint,
                                        "centre": centre,
                                        "model": model_name,
                                        "step": step.step,
                                        "horizon": step.horizon,
                                        "slope_before": step.slope_before,
                                        "citl_before": step.citl_before,
                                        "slope_after": step.slope_after,
                                        "citl_after": step.citl_after,
                                        "n_eligible": step.n_eligible,
                                        "n_events": step.n_events,
                                        "reason": step.reason,
                                        "analysis_status": step.analysis_status,
                                    }
                                )
                            local_adaptation_frames.extend(
                                local_adaptation_rows(
                                    endpoint=endpoint,
                                    centre=centre,
                                    model_name=model_name,
                                    entry=cell.entry,
                                    exit_=cell.exit,
                                    event=cell.event,
                                    risk_unconditional=cell.risk_grid_unconditional,
                                )
                            )

        entry_exclusion_rows = []
        if "OS" in views:
            entry_exclusion_rows.append(
                {"endpoint": "OS", **entry_at_or_after_exit_summary(frame, "os_time")}
            )

        complete_case = pd.DataFrame(complete_case_rows)
        endpoint_robustness = pd.DataFrame(
            _endpoint_robustness_rows(complete_case_rows, tau_frames, endpoints)
        )

        result = SensitivityResult(
            sequencing_lag_support=_concat(lag_support_frames),
            sequencing_lag_km=_concat(lag_km_frames),
            sequencing_lag_km_curve=_concat(lag_km_curve_frames),
            sequencing_lag_pooled_minimal_cox=_concat(lag_pooled_frames),
            truncation_kendall_tau=_concat(tau_frames),
            entry_time_distribution=_concat(entry_dist_frames),
            km_by_entry_quartile=_concat(km_quartile_frames),
            entry_exclusion_summary=pd.DataFrame(entry_exclusion_rows),
            complete_case=complete_case,
            endpoint_robustness=endpoint_robustness,
            calibration_drift=pd.DataFrame(calibration_rows),
            local_recalibration=pd.DataFrame(local_adaptation_frames),
            scaled_brier=pd.DataFrame(scaled_brier_all),
        )

        output = Path(output_directory).resolve() if output_directory else phase2 / "sensitivity"
        artifacts = _publish(output, result, input_hashes, truncation_conclusions)
        _assert_unchanged(input_hashes)
        return SensitivityResult(**{**result.__dict__, "artifacts": artifacts})
    except (SensitivityContractError, SensitivityWorkflowError):
        raise
    except EvaluationWorkflowError as exc:
        # Normalize the reused primary-evaluation workflow `the model-training verifier` failure
        # type to sensitivity-analysis workflow's own, so every
        # failure path in this module surfaces one exception type, matching every other workflow.
        raise SensitivityWorkflowError("Phase 7 sensitivity workflow failed") from exc
    except Exception as exc:
        raise SensitivityWorkflowError("Phase 7 sensitivity workflow failed") from exc


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _endpoint_robustness_rows(
    complete_case_rows: list[dict], tau_frames: list[pd.DataFrame], endpoints: tuple[str, ...]
) -> list[dict]:
    """Native-population supportive-endpoint-robustness comparison for the complete-case and
    truncation sensitivities. See `implement.md`, "Endpoint robustness scope" for the disclosed
    scope decision (sequencing-lag and calibration-drift are published for both endpoints but not
    yet cross-endpoint-classified here)."""
    if "OS" not in endpoints or "TTNT-D" not in endpoints:
        return []
    rows: list[dict] = []

    os_cc = {
        (row["design"], row["centre"], row["model"], row["metric"]): row
        for row in complete_case_rows
        if row["endpoint"] == "OS"
    }
    ttnt_cc = {
        (row["design"], row["centre"], row["model"], row["metric"]): row
        for row in complete_case_rows
        if row["endpoint"] == "TTNT-D"
    }
    for key, os_row in os_cc.items():
        ttnt_row = ttnt_cc.get(key)
        if ttnt_row is None:
            continue
        os_effect = _gain(os_row)
        ttnt_effect = _gain(ttnt_row)
        classification, reason = classify_concordance(
            os_effect=os_effect,
            os_status=os_row["complete_case_reporting_status"],
            ttnt_effect=ttnt_effect,
            ttnt_status=ttnt_row["complete_case_reporting_status"],
        )
        design, centre, model, metric = key
        rows.append(
            {
                "sensitivity": "complete_case",
                "population_view": "native",
                "design": design,
                "centre": centre,
                "model": model,
                "metric": metric,
                "os_effect": os_effect,
                "os_status": os_row["complete_case_reporting_status"],
                "ttnt_effect": ttnt_effect,
                "ttnt_status": ttnt_row["complete_case_reporting_status"],
                "concordance_classification": classification,
                "reason": reason,
            }
        )

    tau_by_endpoint: dict[str, dict[str, dict]] = {}
    for endpoint, table in zip(endpoints, tau_frames, strict=False):
        if endpoint not in ("OS", "TTNT-D") or table.empty:
            continue
        tau_by_endpoint[endpoint] = {row["stratum"]: row for row in table.to_dict("records")}
    os_tau = tau_by_endpoint.get("OS", {})
    ttnt_tau = tau_by_endpoint.get("TTNT-D", {})
    for stratum in sorted(set(os_tau) & set(ttnt_tau)):
        os_row, ttnt_row = os_tau[stratum], ttnt_tau[stratum]
        classification, reason = classify_concordance(
            os_effect=os_row["tau"],
            os_status=os_row["reporting_status"],
            ttnt_effect=ttnt_row["tau"],
            ttnt_status=ttnt_row["reporting_status"],
            materiality_margin=0.1,
        )
        rows.append(
            {
                "sensitivity": "truncation_kendall_tau",
                "population_view": "native",
                "design": "",
                "centre": stratum,
                "model": "",
                "metric": "tau",
                "os_effect": os_row["tau"],
                "os_status": os_row["reporting_status"],
                "ttnt_effect": ttnt_row["tau"],
                "ttnt_status": ttnt_row["reporting_status"],
                "concordance_classification": classification,
                "reason": reason,
            }
        )
    return rows


def _gain(row: dict) -> float | None:
    if row["full_estimate"] is None or row["complete_case_estimate"] is None:
        return None
    return float(row["complete_case_estimate"]) - float(row["full_estimate"])


def _render_figures(result: SensitivityResult, figures_dir: Path) -> None:
    """The seven prespecified sensitivity-analysis figures (`plots.py`), each reading only the
    already-computed
    aggregate frames above -- no figure recomputes a metric. `render_complete_case_forest` writes
    two files (primary + supplementary), so this produces eight PNGs from seven functions."""
    render_sequencing_lag_headline(
        result.sequencing_lag_pooled_minimal_cox, figures_dir / "sequencing_lag_headline.png"
    )
    render_truncation_diagnostic(
        result.entry_time_distribution,
        result.km_by_entry_quartile,
        result.sequencing_lag_km_curve,
        result.truncation_kendall_tau,
        figures_dir / "truncation_diagnostic.png",
    )
    render_complete_case_forest(
        result.complete_case,
        figures_dir / "complete_case_forest_primary.png",
        figures_dir / "complete_case_forest_supplementary.png",
    )
    render_endpoint_robustness(
        result.endpoint_robustness, figures_dir / "endpoint_robustness_comparison.png"
    )
    render_calibration_drift(result.calibration_drift, figures_dir / "calibration_drift_curves.png")
    render_scaled_brier(result.scaled_brier, figures_dir / "scaled_brier_reference.png")
    render_local_adaptation_curve(
        result.local_recalibration,
        result.calibration_drift,
        figures_dir / "local_adaptation_learning_curve.png",
    )


def _render_report(result: SensitivityResult, truncation_conclusions: dict[str, str]) -> str:
    conclusion_lines = "\n".join(
        f"- **{endpoint}:** {conclusion}" for endpoint, conclusion in truncation_conclusions.items()
    )
    return (
        "# Phase 7 sensitivity-analyses report\n\n"
        "Every artifact here is reported **alongside** the frozen Phase 4/5 primary result, "
        "never in place of it. No primary model was refit.\n\n"
        f"## Truncation diagnostic\n\n{len(result.truncation_kendall_tau)} rows. Required "
        f"descriptive-only conclusion, per endpoint:\n\n{conclusion_lines}\n\n"
        f"## Complete-case assessment-subset sensitivity\n\n{len(result.complete_case)} rows.\n\n"
        f"## Supportive endpoint robustness\n\n{len(result.endpoint_robustness)} rows.\n\n"
        f"## Calibration drift and local recalibration\n\n"
        f"{len(result.calibration_drift)} drift rows; {len(result.local_recalibration)} "
        "local-adaptation rows.\n\n"
        f"## No-information reference and scaled Brier score\n\n{len(result.scaled_brier)} rows, "
        "descriptive only.\n\n"
        "## Phase 8 handoff\n\nNo manuscript prose, cohort diagram, or reporting checklist was "
        "produced here; that is Phase 8's work.\n"
    )


_ARTIFACT_NAMES = (
    REPORT,
    SEQUENCING_LAG_SUPPORT,
    SEQUENCING_LAG_KM,
    SEQUENCING_LAG_KM_CURVE,
    SEQUENCING_LAG_POOLED,
    TRUNCATION_TAU,
    ENTRY_TIME_DISTRIBUTION,
    KM_BY_ENTRY_QUARTILE,
    ENTRY_EXCLUSION_SUMMARY,
    COMPLETE_CASE,
    ENDPOINT_ROBUSTNESS,
    CALIBRATION_DRIFT,
    LOCAL_RECALIBRATION,
    SCALED_BRIER,
)


def _publish(
    output: Path,
    result: SensitivityResult,
    input_hashes: dict[str, str],
    truncation_conclusions: dict[str, str],
) -> SensitivityArtifacts:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase7-sensitivity-", dir=output.parent))
    backup = output.parent / f".{output.name}-previous"
    try:
        result.sequencing_lag_support.to_csv(temporary / SEQUENCING_LAG_SUPPORT, index=False)
        result.sequencing_lag_km.to_csv(temporary / SEQUENCING_LAG_KM, index=False)
        result.sequencing_lag_km_curve.to_csv(temporary / SEQUENCING_LAG_KM_CURVE, index=False)
        result.sequencing_lag_pooled_minimal_cox.to_csv(
            temporary / SEQUENCING_LAG_POOLED, index=False
        )
        result.truncation_kendall_tau.to_csv(temporary / TRUNCATION_TAU, index=False)
        result.entry_time_distribution.to_csv(temporary / ENTRY_TIME_DISTRIBUTION, index=False)
        result.km_by_entry_quartile.to_csv(temporary / KM_BY_ENTRY_QUARTILE, index=False)
        result.entry_exclusion_summary.to_csv(temporary / ENTRY_EXCLUSION_SUMMARY, index=False)
        result.complete_case.to_csv(temporary / COMPLETE_CASE, index=False)
        result.endpoint_robustness.to_csv(temporary / ENDPOINT_ROBUSTNESS, index=False)
        result.calibration_drift.to_csv(temporary / CALIBRATION_DRIFT, index=False)
        result.local_recalibration.to_csv(temporary / LOCAL_RECALIBRATION, index=False)
        result.scaled_brier.to_csv(temporary / SCALED_BRIER, index=False)

        figures_dir = temporary / FIGURES
        figures_dir.mkdir()
        _render_figures(result, figures_dir)

        report = _render_report(result, truncation_conclusions)
        (temporary / REPORT).write_text(report, encoding="utf-8")

        hashes = {name: _sha256(temporary / name) for name in _ARTIFACT_NAMES}
        row_counts = {
            SEQUENCING_LAG_SUPPORT: len(result.sequencing_lag_support),
            SEQUENCING_LAG_KM: len(result.sequencing_lag_km),
            SEQUENCING_LAG_KM_CURVE: len(result.sequencing_lag_km_curve),
            SEQUENCING_LAG_POOLED: len(result.sequencing_lag_pooled_minimal_cox),
            TRUNCATION_TAU: len(result.truncation_kendall_tau),
            ENTRY_TIME_DISTRIBUTION: len(result.entry_time_distribution),
            KM_BY_ENTRY_QUARTILE: len(result.km_by_entry_quartile),
            ENTRY_EXCLUSION_SUMMARY: len(result.entry_exclusion_summary),
            COMPLETE_CASE: len(result.complete_case),
            ENDPOINT_ROBUSTNESS: len(result.endpoint_robustness),
            CALIBRATION_DRIFT: len(result.calibration_drift),
            LOCAL_RECALIBRATION: len(result.local_recalibration),
            SCALED_BRIER: len(result.scaled_brier),
        }
        payload: dict[str, object] = {
            "phase": "phase-7-sensitivity-analyses",
            "input_hashes": {Path(path).name: digest for path, digest in input_hashes.items()},
            "software_versions": {
                "lifelines": version("lifelines"),
                "statsmodels": version("statsmodels"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
            },
            "artifact_hashes": hashes,
            "row_counts": row_counts,
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
        if isinstance(exc, SensitivityWorkflowError):
            raise
        raise SensitivityWorkflowError("Phase 7 atomic publication failed") from exc

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
        raise SensitivityWorkflowError("Phase 7 post-publication verification failed") from exc

    if backup.exists():
        shutil.rmtree(backup)
    return SensitivityArtifacts(output)


def _verify_published(output: Path) -> None:
    required = {MANIFEST, FIGURES, *_ARTIFACT_NAMES}
    if {path.name for path in output.iterdir()} != required:
        raise SensitivityWorkflowError("Published Phase 7 bundle has unexpected contents")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_unchanged(hashes: dict[str, str]) -> None:
    if any(_sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise SensitivityWorkflowError("Phase 7 mutated a canonical input")


__all__ = [
    "SensitivityArtifacts",
    "SensitivityResult",
    "SensitivityWorkflowError",
    "run_sensitivity_workflow",
]

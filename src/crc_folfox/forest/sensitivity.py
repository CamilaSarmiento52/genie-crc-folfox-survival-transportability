"""§10: the two sensitivity analyses that apply to the admitted `ltrc_cif`.

`forest-retry-spec.md` §10 records the decision for all six sensitivity analyses. Two are run here:

* **§10.6 — no-information reference and scaled Brier.** The forest-retry report quotes raw
  integrated
  Brier scores with no reference point, so a reader cannot tell whether the forest predicts well or
  barely better than knowing nothing about the patient. `scaled_brier = 1 − Brier_model / Brier_KM`
  supplies that reference. It is a **descriptive aid** carrying ``descriptive_only = True``; raw
  Brier and IBS remain the comparison quantities.
* **§10.5 — calibration drift and local recalibration.** The 2026-08-16 correction established that
  every paired calibration gain interval contains zero and that *both* models under-predict risk.
  That makes the recalibration question open rather than rhetorical: an intercept-only repair, a
  slope repair, and a residual shape mismatch are three different diagnoses, and none of them can be
  read off a slope whose interval spans 1.

Everything statistical is **imported** from sensitivity-analysis and primary-evaluation workflows —
`no_information_brier_rows`,
`scaled_brier_rows`, `calibration_drift_steps`, `local_adaptation_rows` — never reimplemented and
never re-parameterised. Support gates, seeds, horizons, and the feasibility rule come with them.

Two properties are enforced rather than assumed:

1. **The model's Brier values are read from the published `performance_summary.csv`**, so the scaled
   version is a rescaling of exactly the number that appears in the report, not a re-derivation that
   might differ in the last digits.
2. **The no-information reference is validated against sensitivity-analysis workflow's own published
   reference** for the
   same centre and endpoint. The reference depends only on the assessment set's
   `(entry, exit, event)`, which forest-retry and sensitivity-analysis workflows share, so any
   disagreement means the two workflows
   are describing different patients — a failure worth surfacing loudly, in the same spirit as the
   `minimal_cox` re-scoring check that validated the forest-model-retry workflow evaluation path.

No sensitivity-analysis artifact is read for anything but that validation, and none is rewritten.
Every row published here lives in the forest-retry artifact directory and carries
`post_hoc_exploratory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crc_folfox.evaluate.contracts import CENTRES, EvaluationContractError
from crc_folfox.forest.workflow import ANALYSIS_STATUS, CANDIDATE, _scope_id
from crc_folfox.sensitivity.calibration_drift import calibration_drift_steps, local_adaptation_rows
from crc_folfox.sensitivity.complete_case import (
    complete_case_comparison_rows,
    complete_case_flag_frame,
    complete_case_mask_for,
)
from crc_folfox.sensitivity.no_information import no_information_brier_rows, scaled_brier_rows
from crc_folfox.sensitivity.sequencing_lag import lag_pooled_minimal_cox_rows

SCALED_BRIER_ARTIFACT = "scaled_brier.csv"
CALIBRATION_DRIFT_ARTIFACT = "calibration_drift.csv"
LOCAL_RECALIBRATION_ARTIFACT = "local_recalibration.csv"
COMPLETE_CASE_ARTIFACT = "complete_case_assessment_subset.csv"
SEQUENCING_LAG_ARTIFACT = "sequencing_lag_pooled.csv"

#: The three metric keys `no_information_brier_rows` emits, matching sensitivity-analysis workflow's
#: own vocabulary.
_BRIER_KEYS = ("brier_365", "brier_730", "ibs")

#: sensitivity-analysis workflow publishes scaled Brier for both designs; calibration drift and the
#: local-adaptation
#: learning curve are IECV only. forest-model-retry workflow matches that scope exactly rather than
#: inventing its own.
_SCALED_BRIER_DESIGNS = ("iecv", "pooled")

#: The no-information reference must agree with sensitivity-analysis workflow's to within
#: bootstrap-free numerical noise:
#: both fit the same delayed-entry Kaplan-Meier to the same patients. This is a hard equality check
#: in practice, given a generous margin only for float round-tripping through CSV.
_REFERENCE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ForestSensitivityResult:
    scaled_brier: pd.DataFrame
    calibration_drift: pd.DataFrame
    local_recalibration: pd.DataFrame
    complete_case: pd.DataFrame
    sequencing_lag: pd.DataFrame
    reference_max_deviation: float | None
    output: Path


def _published_model_brier(performance: pd.DataFrame, endpoint: str) -> dict[tuple, float | None]:
    """`(design, centre, metric_key) -> Brier` for the forest, from the published forest-model-retry
    workflow summary.

    `ibs` carries no horizon; `brier` carries 365 or 730. Only `estimable`/`estimable_imprecise`
    rows contribute a value — a `not_computable` cell has no Brier to scale, and inventing one would
    manufacture a scaled score out of a missing measurement.
    """
    frame = performance[
        (performance["endpoint"] == endpoint)
        & (performance["model"] == CANDIDATE)
        & (performance["risk_scale"] == "unconditional")
        & (performance["metric"].isin(("brier", "ibs")))
    ]
    lookup: dict[tuple, float | None] = {}
    for row in frame.itertuples(index=False):
        key = "ibs" if row.metric == "ibs" else f"brier_{float(row.horizon):g}"
        reported = str(row.reporting_status) in ("estimable", "estimable_imprecise")
        value = float(row.estimate) if reported and not pd.isna(row.estimate) else None
        lookup[(str(row.design), str(row.centre), key)] = value
    return lookup


def _validate_reference(
    computed: dict[str, dict], published: pd.DataFrame, endpoint: str, centre: str
) -> float | None:
    """Largest absolute disagreement with sensitivity-analysis workflow's published no-information
    reference, or `None`
    when sensitivity-analysis workflow published nothing comparable (so the check is skipped
    rather than faked)."""
    subset = published[(published["endpoint"] == endpoint) & (published["centre"] == centre)]
    if subset.empty:
        return None
    deviations: list[float] = []
    for key, entry in computed.items():
        value = entry.get("value")
        if value is None:
            continue
        rows = subset[subset["metric"] == key]
        if rows.empty:
            continue
        reference = rows.iloc[0]["no_information_brier"]
        if pd.isna(reference):
            continue
        deviations.append(abs(float(reference) - float(value)))
    return max(deviations) if deviations else None


def run_forest_sensitivity_workflow(
    input_dir: Path,
    output_dir: Path,
    *,
    endpoint: str = "OS",
    centres: tuple[str, ...] = CENTRES,
    progress: bool = False,
) -> ForestSensitivityResult:
    """Scaled Brier plus calibration drift and local recalibration for the fitted `ltrc_cif`.

    Nothing is refitted. The forest's held-out predictions are rebuilt through
    `forest.evaluation._forest_iecv_cell`, the same code path the published performance numbers came
    from, so the drift steps describe exactly the predictions the report describes.
    """
    from crc_folfox.evaluate.bootstrap import bootstrap_index_matrix
    from crc_folfox.evaluate.contracts import BOOTSTRAP_RESAMPLES, MIN_ADMISSIBLE_POOLED_REPEATS
    from crc_folfox.evaluate.workflow import _verify_phase4
    from crc_folfox.forest.evaluation import (
        _forest_iecv_cell,
        _forest_pooled_cell,
        _load_forest_registry,
    )
    from crc_folfox.forest.r_adapter import RConditionalForestAdapter
    from crc_folfox.pipeline import load_clean_analysis_dataset
    from crc_folfox.preprocessing.contracts import build_endpoint_view
    from crc_folfox.preprocessing.splits import build_iecv_splits, build_pooled_splits

    # Resolved, not merely coerced: the R adapter runs with `cwd` set to its own script directory,
    # so a relative bundle path reaches `readRDS` as a path relative to `r/the forest-retry artifact
    # directory` and fails.
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = pd.read_csv(output_dir / "performance_summary.csv")
    model_brier = _published_model_brier(performance, endpoint)

    phase7 = input_dir / "sensitivity" / "scaled_brier.csv"
    published_reference = pd.read_csv(phase7) if phase7.is_file() else pd.DataFrame()

    frame = load_clean_analysis_dataset(input_dir / "analysis_dataset.csv")
    _verify_phase4(input_dir / "model_training")
    forest_registry = _load_forest_registry(output_dir)
    view = build_endpoint_view(frame, endpoint)
    adapter = RConditionalForestAdapter()

    # `complete_case_flag_frame` reads the RAW cleaned analysis frame: "complete case" is recomputed
    # from
    # the unimputed `crc_type`/`ca_grade` columns, never inferred from a transformed matrix where
    # imputation has already erased the distinction.
    complete_case_flags = complete_case_flag_frame(frame)
    pooled_splits = build_pooled_splits(view)

    scaled_rows: list[dict] = []
    drift_rows: list[dict] = []
    recalibration_rows: list[dict] = []
    complete_case_rows: list[dict] = []
    lag_rows: list[dict] = []
    deviations: list[float] = []

    for split in build_iecv_splits(view):
        centre = str(split.held_out_institution)
        if centre not in centres:
            continue
        record = forest_registry.get(_scope_id(split))
        if record is None or record.get("fit_status") != "fit_succeeded":
            raise EvaluationContractError(f"'{centre}' has no fit_succeeded forest to analyse")
        if progress:
            print(f"[sensitivity] {centre}: predicting", flush=True)

        cell = _forest_iecv_cell(view, split, forest_registry, output_dir / "models", adapter)
        if cell.status != "ok" or cell.risk_grid_unconditional is None:
            raise EvaluationContractError(f"'{centre}' forest cell is not computable")
        entry, exit_, event = cell.entry, cell.exit, cell.event

        # --- §10.6 scaled Brier -------------------------------------------------------------
        # The reference depends only on the assessment set's (entry, exit, event), which is the
        # held-out centre's patients under both designs -- exactly why sensitivity-analysis
        # workflow's own published
        # reference is identical across `iecv` and `pooled`. Computed once, used for both.
        reference = no_information_brier_rows(endpoint, centre, entry, exit_, event)
        if not published_reference.empty:
            deviation = _validate_reference(reference, published_reference, endpoint, centre)
            if deviation is not None:
                deviations.append(deviation)
                if deviation > _REFERENCE_TOLERANCE:
                    raise EvaluationContractError(
                        f"Phase 9's no-information reference for {centre} disagrees with Phase 7's "
                        f"by {deviation:.3e} -- the two phases are not describing the same patients"
                    )

        for design in _SCALED_BRIER_DESIGNS:
            by_key = {key: model_brier.get((design, centre, key)) for key in _BRIER_KEYS}
            for row in scaled_brier_rows(
                endpoint=endpoint,
                design=design,
                centre=centre,
                model_name=CANDIDATE,
                model_brier_by_key=by_key,
                no_information_brier_by_key=reference,
            ):
                scaled_rows.append({**row, "analysis_status": ANALYSIS_STATUS})

        # --- §10.5 calibration drift, steps 1-3 ---------------------------------------------
        for step in calibration_drift_steps(entry, exit_, event, cell.risk_grid_unconditional):
            drift_rows.append(
                {
                    "endpoint": endpoint,
                    "centre": centre,
                    "model": CANDIDATE,
                    "step": step.step,
                    "horizon": step.horizon,
                    "slope_before": step.slope_before,
                    "citl_before": step.citl_before,
                    "slope_after": step.slope_after,
                    "citl_after": step.citl_after,
                    "n_eligible": step.n_eligible,
                    "n_events": step.n_events,
                    "reason": step.reason,
                    "analysis_status": ANALYSIS_STATUS,
                }
            )

        # --- §10.5 step 4, the local-adaptation learning curve -------------------------------
        # An infeasible fraction is returned as infeasible with a reason; it is never manufactured.
        # This matters most at VICC, whose support gates are the ones that fail.
        for row in local_adaptation_rows(
            endpoint=endpoint,
            centre=centre,
            model_name=CANDIDATE,
            entry=entry,
            exit_=exit_,
            event=event,
            risk_unconditional=cell.risk_grid_unconditional,
        ):
            recalibration_rows.append({**row, "analysis_status": ANALYSIS_STATUS})

        # --- §10.3 complete-case assessment subset, and §10.1 sequencing lag -----------------
        # Both need the pooled cell as well as the IECV one: sensitivity-analysis workflow reports
        # complete-case for both
        # designs and the lag analysis on the pooled prediction. The pooled cell costs 50 fold
        # predictions, which is why it is built once here and shared.
        if progress:
            print(f"[sensitivity] {centre}: pooled prediction", flush=True)
        pooled = _forest_pooled_cell(
            view,
            centre,
            pooled_splits,
            forest_registry,
            output_dir / "models",
            adapter,
            MIN_ADMISSIBLE_POOLED_REPEATS,
        )

        for design, design_cell in (("iecv", cell), ("pooled", pooled)):
            if design_cell.status != "ok" or design_cell.risk_grid_unconditional is None:
                continue
            # Deterministic in (endpoint, centre) alone -- the same matrix primary-evaluation and
            # forest-retry workflows
            # evaluation used, so the complete-case-versus-full difference is exactly paired.
            index_matrix = bootstrap_index_matrix(
                endpoint,
                centre,
                len(design_cell.patient_key),
                resamples=BOOTSTRAP_RESAMPLES,
            )
            mask = complete_case_mask_for(complete_case_flags, design_cell.patient_key)
            for row in complete_case_comparison_rows(
                endpoint=endpoint,
                design=design,
                centre=centre,
                model_name=CANDIDATE,
                entry=design_cell.entry,
                exit_=design_cell.exit,
                event=design_cell.event,
                risk_unconditional=design_cell.risk_grid_unconditional,
                linear_predictor=design_cell.linear_predictor,
                complete_case_mask=mask,
                index_matrix=index_matrix,
            ):
                complete_case_rows.append({**row, "analysis_status": ANALYSIS_STATUS})

            if design == "pooled":
                for row in lag_pooled_minimal_cox_rows(
                    endpoint=endpoint,
                    centre=centre,
                    entry=design_cell.entry,
                    exit_=design_cell.exit,
                    event=design_cell.event,
                    risk_unconditional=design_cell.risk_grid_unconditional,
                    linear_predictor=design_cell.linear_predictor,
                    index_matrix=index_matrix,
                    model_name=CANDIDATE,
                ):
                    lag_rows.append({**row, "analysis_status": ANALYSIS_STATUS})

    scaled = pd.DataFrame(scaled_rows)
    drift = pd.DataFrame(drift_rows)
    recalibration = pd.DataFrame(recalibration_rows)
    complete_case = pd.DataFrame(complete_case_rows)
    lag = pd.DataFrame(lag_rows)
    _publish(output_dir, scaled, drift, recalibration, complete_case, lag)
    return ForestSensitivityResult(
        scaled_brier=scaled,
        calibration_drift=drift,
        local_recalibration=recalibration,
        complete_case=complete_case,
        sequencing_lag=lag,
        reference_max_deviation=max(deviations) if deviations else None,
        output=output_dir,
    )


def _publish(
    output_dir: Path,
    scaled: pd.DataFrame,
    drift: pd.DataFrame,
    recalibration: pd.DataFrame,
    complete_case: pd.DataFrame,
    lag: pd.DataFrame,
) -> None:
    """Stage then replace, so a partially written CSV is never left where a reader could mistake it
    for a complete result."""
    for name, frame in (
        (SCALED_BRIER_ARTIFACT, scaled),
        (CALIBRATION_DRIFT_ARTIFACT, drift),
        (LOCAL_RECALIBRATION_ARTIFACT, recalibration),
        (COMPLETE_CASE_ARTIFACT, complete_case),
        (SEQUENCING_LAG_ARTIFACT, lag),
    ):
        destination = output_dir / name
        staging = destination.with_suffix(".csv.tmp")
        frame.to_csv(staging, index=False)
        staging.replace(destination)


__all__ = [
    "CALIBRATION_DRIFT_ARTIFACT",
    "COMPLETE_CASE_ARTIFACT",
    "LOCAL_RECALIBRATION_ARTIFACT",
    "SCALED_BRIER_ARTIFACT",
    "SEQUENCING_LAG_ARTIFACT",
    "ForestSensitivityResult",
    "run_forest_sensitivity_workflow",
]

"""E1: coefficient explanations, with the uncertainty-provenance rule enforced structurally.

`explainability-spec.md`, "E1 -- coefficient explanations": model-based standard errors and
p-values are **prohibited** for every penalized fit; only the truly unpenalized `minimal_cox`
(`penalizer == 0`, `l1_ratio == 0`) may carry them, and even then only via a one-shot refit --
`CoxBundleState` deliberately discards its fitter after coefficient/baseline extraction
(`model_training/bundles.py`), so there is no stored fitter to ask for `standard_errors_`. Every
other explained model's interval comes only from the labelled bootstrap refit distribution
(`explain.refits`), attached by `attach_bootstrap_intervals`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from crc_folfox.explain.contracts import ExplainContractError
from crc_folfox.explain.feature_groups import group_for_column
from crc_folfox.explain.loading import LoadedModel, RiskPrediction
from crc_folfox.preprocessing.transform import FittedPreprocessor

PROVENANCE_MODEL_BASED = "model_based"
PROVENANCE_BOOTSTRAP_REFIT = "bootstrap_refit"

# The only truly unpenalized candidate. Every other explained model -- including `spline_cox`,
# which uses a fixed nonzero penalty -- is penalized and must never carry a model-based interval.
_MODEL_BASED_SE_MODELS = ("minimal_cox",)


@dataclass(frozen=True)
class CoefficientRow:
    model_name: str
    design_column: str
    feature_group: str
    coefficient: float
    hazard_ratio: float
    interval_provenance: str
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


def point_coefficients(loaded: LoadedModel, prediction: RiskPrediction) -> list[CoefficientRow]:
    """Point-estimate coefficients and hazard ratios, tagged with their required interval
    provenance. `bootstrap_refit` rows have null SE/CI here; `attach_bootstrap_intervals` fills
    them from the refit distribution."""
    if prediction.coefficients is None:
        raise ExplainContractError(f"'{loaded.model_name}' produced no coefficients to explain")
    provenance = (
        PROVENANCE_MODEL_BASED
        if loaded.model_name in _MODEL_BASED_SE_MODELS
        else PROVENANCE_BOOTSTRAP_REFIT
    )
    return [
        CoefficientRow(
            model_name=loaded.model_name,
            design_column=column,
            feature_group=group_for_column(column),
            coefficient=float(coefficient),
            hazard_ratio=float(np.exp(coefficient)),
            interval_provenance=provenance,
        )
        for column, coefficient in prediction.coefficients.items()
    ]


def compute_minimal_cox_model_based_uncertainty(
    loaded: LoadedModel, development_frame: pd.DataFrame, *, fit_scope: str
) -> dict[str, tuple[float, float, float]]:
    """Refit `minimal_cox` once on its own recorded development frame, purely to read lifelines'
    own standard errors and 95% CIs -- `CoxBundleState` never stores them. The refit's coefficients
    must match the loaded (never-refit) bundle to float precision before its SE is trusted; a
    mismatch means the development frame passed in is not the bundle's own, and this function
    refuses to report an uncertainty estimate for the wrong model.

    Returns `{design_column: (standard_error, ci_lower, ci_upper)}` on the log-hazard-ratio scale.
    """
    if loaded.model_name != "minimal_cox":
        raise ExplainContractError("Model-based uncertainty is defined only for minimal_cox")
    state = loaded.cox_state
    if state.penalizer != 0.0 or state.l1_ratio != 0.0:
        raise ExplainContractError("minimal_cox bundle is not truly unpenalized")

    preprocessor = FittedPreprocessor.fit(
        development_frame, endpoint=state.endpoint, feature_set="baseline", fit_scope=fit_scope
    )
    transformed = preprocessor.transform(development_frame)
    fitter = CoxPHFitter(penalizer=0.0, l1_ratio=0.0, baseline_estimation_method="breslow")
    frame = pd.concat(
        [
            transformed.metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
            transformed.matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    fitter.fit(frame, duration_col="exit", event_col="event", entry_col="entry")

    refit_coefficients = fitter.params_
    if tuple(refit_coefficients.index.astype(str)) != state.feature_columns:
        raise ExplainContractError("Refit design does not match the loaded minimal_cox bundle")
    if not np.allclose(
        refit_coefficients.to_numpy(float), np.asarray(state.coefficients), atol=1e-6
    ):
        raise ExplainContractError(
            "Refit coefficients do not match the frozen minimal_cox bundle -- refusing to report "
            "a model-based interval computed from a different fit"
        )

    standard_errors = fitter.standard_errors_
    ci = fitter.confidence_intervals_
    return {
        column: (
            float(standard_errors[column]),
            float(ci.loc[column].iloc[0]),
            float(ci.loc[column].iloc[1]),
        )
        for column in state.feature_columns
    }


def attach_model_based_uncertainty(
    rows: list[CoefficientRow], uncertainty: dict[str, tuple[float, float, float]]
) -> list[CoefficientRow]:
    updated = []
    for row in rows:
        if (
            row.interval_provenance != PROVENANCE_MODEL_BASED
            or row.design_column not in uncertainty
        ):
            updated.append(row)
            continue
        se, lower, upper = uncertainty[row.design_column]
        updated.append(
            CoefficientRow(
                model_name=row.model_name,
                design_column=row.design_column,
                feature_group=row.feature_group,
                coefficient=row.coefficient,
                hazard_ratio=row.hazard_ratio,
                interval_provenance=row.interval_provenance,
                standard_error=se,
                ci_lower=lower,
                ci_upper=upper,
            )
        )
    return updated


def attach_bootstrap_intervals(
    rows: list[CoefficientRow], refit_coefficients: dict[str, list[float]]
) -> list[CoefficientRow]:
    """Percentile 95% interval from the refit distribution -- `bootstrap_refit` rows only; a
    `model_based` row is left untouched even if a (spuriously supplied) refit distribution exists
    for it, so the two provenances can never collide on one row."""
    updated = []
    for row in rows:
        if row.interval_provenance != PROVENANCE_BOOTSTRAP_REFIT:
            updated.append(row)
            continue
        values = refit_coefficients.get(row.design_column, [])
        if not values:
            updated.append(row)
            continue
        lower, upper = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
        updated.append(
            CoefficientRow(
                model_name=row.model_name,
                design_column=row.design_column,
                feature_group=row.feature_group,
                coefficient=row.coefficient,
                hazard_ratio=row.hazard_ratio,
                interval_provenance=row.interval_provenance,
                standard_error=None,
                ci_lower=float(lower),
                ci_upper=float(upper),
            )
        )
    return updated


__all__ = [
    "PROVENANCE_BOOTSTRAP_REFIT",
    "PROVENANCE_MODEL_BASED",
    "CoefficientRow",
    "attach_bootstrap_intervals",
    "attach_model_based_uncertainty",
    "compute_minimal_cox_model_based_uncertainty",
    "point_coefficients",
]

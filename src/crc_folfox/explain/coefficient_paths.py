"""E3: coefficient paths across each model's own frozen tuning grid, at the development-scope fit.

`explainability-spec.md`'s 2026-08-11 amendment (`coefficient_paths.csv`): a different quantity
from the bootstrap-refit coefficient *sample* (`selection_stability.bootstrap_coefficient_values`)
at one already-selected configuration -- this module computes coefficients at *every* point of the
frozen tuning grid, once, with no bootstrap.

**Python `elastic_net_cox`.** Refit `fit_cox()` once per point of the frozen `ELASTIC_NET_GRID`
(12 `(penalizer, l1_ratio)` pairs), using the development-scope's already-fit preprocessing --
cheap, deterministic, no re-tuning or selection logic involved.

**R `glmnet_enet_core`.** optimization workflow's per-parent-scope reference lambda sequence
(`optimize.glmnet_selection.tune_glmnet_cox`'s `reference_lambdas`) is not persisted to the
permanent archive -- it is fit to a temporary bundle inside a directory that gets
`shutil.rmtree`'d after tuning. That reference fit is, however, fully deterministic: its seed is
`child_seed(..., purpose="glmnet_reference_path")`, a pure function of the split identifiers
explainability workflow already has for every registered scope. This module regenerates it on the
identical frozen
development-scope matrix (`loaded.r_preprocessor.transform(development_frame)` -- `r_preprocessor`
*is* the frozen preprocessor object from the original tuning run, persisted alongside the model
bundle) and **verifies** the regeneration by also reproducing the deterministic
`purpose="glmnet_final_refit"` single-lambda call at the frozen registry's recorded selected
lambda, comparing its coefficients against the frozen bundle's own actual loaded coefficients. Only
a passing verification is published as `path_provenance=verified_reconstruction`; every other
outcome is disclosed via `path_provenance`, never silently upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.contracts import IBS_GRID
from crc_folfox.explain.feature_groups import group_for_column
from crc_folfox.model_training.contracts import ELASTIC_NET_GRID, child_seed
from crc_folfox.model_training.cox import fit_cox
from crc_folfox.model_training.selection import EXPECTED_TUNING_EXCEPTIONS
from crc_folfox.optimize.contracts import OptimizationContractError
from crc_folfox.optimize.r_adapter import GlmnetAdapter
from crc_folfox.preprocessing.contracts import SplitDefinition

FIT_SUCCEEDED = "succeeded"
FIT_FAILED = "failed"

# Python: this artifact is itself the "development_scope_fit" -- there is no persistence question.
PROVENANCE_DEVELOPMENT_SCOPE_FIT = "development_scope_fit"
# R: the reference lambda sequence was regenerated deterministically and its selected-lambda
# coefficients verified bit-close against the frozen bundle's own coefficients.
PROVENANCE_VERIFIED_RECONSTRUCTION = "verified_reconstruction"
# R: regeneration ran, but the verification check did not pass -- published only as a dated,
# explicitly-labelled exploratory reconstruction, never presented as the original tuning path.
PROVENANCE_EXPLORATORY_RECONSTRUCTION = "exploratory_reconstruction"
# R: reconstruction could not be attempted or completed at all (e.g. the R adapter itself failed).
PROVENANCE_NOT_PERSISTED = "original_lambda_path_not_persisted"

_VERIFICATION_ATOL_R = 1e-6
_LAMBDA_MATCH_ATOL = 1e-9


@dataclass(frozen=True)
class GridPathRow:
    configuration_id: int
    configuration_label: str
    penalizer: float | None
    l1_ratio: float | None
    alpha: float | None
    lambda_value: float | None
    design_column: str | None
    feature_group: str | None
    coefficient: float | None
    is_selected_configuration: bool
    fit_status: str
    path_provenance: str


def _failed_configuration_row(
    configuration_id: int,
    label: str,
    *,
    penalizer: float | None,
    l1_ratio: float | None,
    alpha: float | None,
    lambda_value: float | None,
    is_selected: bool,
    provenance: str,
) -> GridPathRow:
    """A grid point that failed to fit publishes one status row -- never one null row per design
    column, which would falsely imply a per-column failure granularity glmnet/Cox fits do not
    have."""
    return GridPathRow(
        configuration_id=configuration_id,
        configuration_label=label,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        alpha=alpha,
        lambda_value=lambda_value,
        design_column=None,
        feature_group=None,
        coefficient=None,
        is_selected_configuration=is_selected,
        fit_status=FIT_FAILED,
        path_provenance=provenance,
    )


def compute_elastic_net_grid_path(
    development_matrix: pd.DataFrame,
    development_metadata: pd.DataFrame,
    *,
    selected_penalizer: float,
    selected_l1_ratio: float,
) -> list[GridPathRow]:
    """One row per `(configuration, design_column)` across the frozen `ELASTIC_NET_GRID`, at the
    already-transformed development-scope design -- no re-tuning, no bootstrap."""
    rows: list[GridPathRow] = []
    for configuration_id, (penalizer, l1_ratio) in enumerate(ELASTIC_NET_GRID):
        label = f"penalizer={penalizer},l1_ratio={l1_ratio}"
        is_selected = penalizer == selected_penalizer and l1_ratio == selected_l1_ratio
        try:
            fitter = fit_cox(
                development_matrix, development_metadata, penalizer=penalizer, l1_ratio=l1_ratio
            )
        except EXPECTED_TUNING_EXCEPTIONS:
            rows.append(
                _failed_configuration_row(
                    configuration_id,
                    label,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    alpha=None,
                    lambda_value=None,
                    is_selected=is_selected,
                    provenance=PROVENANCE_DEVELOPMENT_SCOPE_FIT,
                )
            )
            continue
        for design_column, coefficient in fitter.params_.items():
            design_column = str(design_column)
            rows.append(
                GridPathRow(
                    configuration_id=configuration_id,
                    configuration_label=label,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    alpha=None,
                    lambda_value=None,
                    design_column=design_column,
                    feature_group=group_for_column(design_column),
                    coefficient=float(coefficient),
                    is_selected_configuration=is_selected,
                    fit_status=FIT_SUCCEEDED,
                    path_provenance=PROVENANCE_DEVELOPMENT_SCOPE_FIT,
                )
            )
    return rows


def compute_glmnet_enet_grid_path(
    development_frame: pd.DataFrame,
    r_preprocessor,
    split: SplitDefinition,
    *,
    model_name: str,
    alpha: float,
    selected_lambda: float,
    frozen_coefficients: dict[str, float],
    tmp_dir: Path,
) -> list[GridPathRow]:
    """Regenerate `glmnet_enet_core`'s reference lambda sequence deterministically and verify it
    against the frozen bundle's own coefficients before publishing anything as
    `verified_reconstruction` (see module docstring)."""
    transformed = r_preprocessor.transform(development_frame)
    adapter = GlmnetAdapter()

    reference_seed = child_seed(
        endpoint=split.endpoint,
        design=split.design,
        held_out_institution=split.held_out_institution,
        repeat=split.repeat,
        fold=split.fold,
        model_name=model_name,
        candidate=str(alpha),
        purpose="glmnet_reference_path",
    )
    final_seed = child_seed(
        endpoint=split.endpoint,
        design=split.design,
        held_out_institution=split.held_out_institution,
        repeat=split.repeat,
        fold=split.fold,
        model_name=model_name,
        candidate=str(alpha),
        purpose="glmnet_final_refit",
    )

    try:
        reference = adapter.fit(
            transformed.matrix,
            transformed.metadata,
            transformed.matrix.iloc[:1],
            alpha=alpha,
            lambda_value=None,
            seed=reference_seed,
            horizons=IBS_GRID,
            destination=tmp_dir / "coefficient_path_reference.rds",
        )
        final_refit = adapter.fit(
            transformed.matrix,
            transformed.metadata,
            transformed.matrix.iloc[:1],
            alpha=alpha,
            lambda_value=float(selected_lambda),
            seed=final_seed,
            horizons=IBS_GRID,
            destination=tmp_dir / "coefficient_path_final.rds",
        )
    except (OptimizationContractError, *EXPECTED_TUNING_EXCEPTIONS):
        return [
            _failed_configuration_row(
                0,
                f"alpha={alpha},lambda_path=unavailable",
                penalizer=None,
                l1_ratio=None,
                alpha=alpha,
                lambda_value=None,
                is_selected=False,
                provenance=PROVENANCE_NOT_PERSISTED,
            )
        ]
    finally:
        for name in ("coefficient_path_reference.rds", "coefficient_path_final.rds"):
            (tmp_dir / name).unlink(missing_ok=True)

    verified = (
        final_refit.coefficients is not None
        and set(final_refit.coefficients) == set(frozen_coefficients)
        and np.allclose(
            [final_refit.coefficients[c] for c in frozen_coefficients],
            [frozen_coefficients[c] for c in frozen_coefficients],
            atol=_VERIFICATION_ATOL_R,
        )
    )
    provenance = (
        PROVENANCE_VERIFIED_RECONSTRUCTION if verified else PROVENANCE_EXPLORATORY_RECONSTRUCTION
    )

    if reference.coefficient_path is None or len(reference.lambda_values) == 0:
        return [
            _failed_configuration_row(
                0,
                f"alpha={alpha},lambda_path=unavailable",
                penalizer=None,
                l1_ratio=None,
                alpha=alpha,
                lambda_value=None,
                is_selected=False,
                provenance=provenance,
            )
        ]

    rows: list[GridPathRow] = []
    for configuration_id, (lambda_value, coefficients_at_lambda) in enumerate(
        zip(reference.lambda_values, reference.coefficient_path, strict=True)
    ):
        label = f"alpha={alpha},lambda={lambda_value:.6g}"
        is_selected = bool(np.isclose(lambda_value, selected_lambda, atol=_LAMBDA_MATCH_ATOL))
        for design_column, coefficient in coefficients_at_lambda.items():
            rows.append(
                GridPathRow(
                    configuration_id=configuration_id,
                    configuration_label=label,
                    penalizer=None,
                    l1_ratio=None,
                    alpha=float(alpha),
                    lambda_value=float(lambda_value),
                    design_column=design_column,
                    feature_group=group_for_column(design_column),
                    coefficient=float(coefficient),
                    is_selected_configuration=is_selected,
                    fit_status=FIT_SUCCEEDED,
                    path_provenance=provenance,
                )
            )
    return rows


__all__ = [
    "FIT_FAILED",
    "FIT_SUCCEEDED",
    "PROVENANCE_DEVELOPMENT_SCOPE_FIT",
    "PROVENANCE_EXPLORATORY_RECONSTRUCTION",
    "PROVENANCE_NOT_PERSISTED",
    "PROVENANCE_VERIFIED_RECONSTRUCTION",
    "GridPathRow",
    "compute_elastic_net_grid_path",
    "compute_glmnet_enet_grid_path",
]

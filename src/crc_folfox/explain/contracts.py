"""Explainability workflow public contracts: the model-availability hierarchy and its frozen
register.

`explainability-spec.md`, "Model availability hierarchy" -- every explained artifact row carries
`explanation_tier`, `source_registry`, and `analysis_status`, derived here from a single frozen
model register rather than supplied by a caller. This module is the one place the nine-model list
is enumerated; `explain.workflow` and every metric module import it rather than redeclaring it.
"""

from __future__ import annotations

from dataclasses import dataclass

from crc_folfox.evaluate.contracts import (
    MIN_BOOTSTRAP_FINITE_FRACTION,
    MIN_ELIGIBLE_EVENTS,
    MIN_EVENT_FREE,
    REPORTING_ESTIMABLE,
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
)
from crc_folfox.model_training.contracts import HORIZONS

EXPLANATION_HORIZONS = HORIZONS  # (365.0, 730.0), unchanged and locked.
PRIMARY_EXPLANATION_HORIZON = 730.0

TIER_PRIMARY_REFERENCE = "primary_reference"
TIER_COMPARISON = "comparison"
TIER_EXPLORATORY_SUPPLEMENTARY = "exploratory_supplementary"
TIER_NOT_EXPLAINED = "not_explained"

SOURCE_FROZEN_MODEL_TRAINING = "frozen_model_training"
SOURCE_CORRECTED_MODEL_TRAINING = "corrected_model_training"
SOURCE_EXPLORATORY_OPTIMIZATION = "exploratory_optimization"

ANALYSIS_STATUS_PRIMARY = "primary"
ANALYSIS_STATUS_TECHNICAL_CORRECTION = "technical_correction"
ANALYSIS_STATUS_POST_HOC_EXPLORATORY = "post_hoc_exploratory"

_ANALYSIS_STATUS_BY_SOURCE = {
    SOURCE_FROZEN_MODEL_TRAINING: ANALYSIS_STATUS_PRIMARY,
    SOURCE_CORRECTED_MODEL_TRAINING: ANALYSIS_STATUS_TECHNICAL_CORRECTION,
    SOURCE_EXPLORATORY_OPTIMIZATION: ANALYSIS_STATUS_POST_HOC_EXPLORATORY,
}

# R-backed candidates predict through the R adapter boundary and have no closed-form Cox linear
# predictor; a Python `CoxBundleState`-family candidate is fit and predicted entirely in Python.
RUNTIME_PYTHON = "python"
RUNTIME_R = "r"

REASON_FIT_FAILED_SINGULAR_DESIGN = "fit_failed_singular_design"
REASON_FIT_INSTABILITY_EXPLANATION_UNRELIABLE = "fit_instability_explanation_unreliable"


class ExplainContractError(ValueError):
    """Raised for a privacy-safe explainability workflow contract violation."""


@dataclass(frozen=True)
class ModelRegistration:
    """One frozen entry in the explainability model-availability hierarchy."""

    model_name: str
    tier: str
    source_registry: str
    feature_set: str  # "baseline" or "core"
    runtime: str
    sparsity_capable: bool  # True only for a model whose penalty can produce exact-zero columns
    supports_partial_effects: bool
    supports_admitted_interaction: bool
    not_explained_reason: str | None = None
    not_explained_fit_success_count: int | None = None
    not_explained_fit_total_count: int | None = None

    def __post_init__(self) -> None:
        if self.tier not in {
            TIER_PRIMARY_REFERENCE,
            TIER_COMPARISON,
            TIER_EXPLORATORY_SUPPLEMENTARY,
            TIER_NOT_EXPLAINED,
        }:
            raise ExplainContractError(f"Unknown explanation tier for '{self.model_name}'")
        if self.tier == TIER_NOT_EXPLAINED:
            if not self.not_explained_reason:
                raise ExplainContractError(f"Tier-4 model '{self.model_name}' must record a reason")
        elif self.not_explained_reason is not None:
            raise ExplainContractError(
                f"Explained model '{self.model_name}' must not carry a not-explained reason"
            )
        if self.feature_set not in {"baseline", "core"}:
            raise ExplainContractError(f"Unknown feature set for '{self.model_name}'")
        if self.runtime not in {RUNTIME_PYTHON, RUNTIME_R}:
            raise ExplainContractError(f"Unknown runtime for '{self.model_name}'")

    @property
    def analysis_status(self) -> str:
        """Derived from the source registry alone -- never overridable by a caller."""
        return _ANALYSIS_STATUS_BY_SOURCE[self.source_registry]

    @property
    def is_explained(self) -> bool:
        return self.tier != TIER_NOT_EXPLAINED


# Frozen 2026-08-10, `explainability-spec.md` "Model availability hierarchy". Real fit-success
# counts for the two Tier 4 models are the published model-training / optimization totals -- 3/106
# real
# development scopes for `core_cox` (103/106 `fit_failed`), 24/56 for `flexsurv_rp_spline`.
MODEL_REGISTER: tuple[ModelRegistration, ...] = (
    ModelRegistration(
        model_name="minimal_cox",
        tier=TIER_PRIMARY_REFERENCE,
        source_registry=SOURCE_FROZEN_MODEL_TRAINING,
        feature_set="baseline",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=False,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="spline_cox",
        tier=TIER_COMPARISON,
        source_registry=SOURCE_FROZEN_MODEL_TRAINING,
        feature_set="core",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=False,
        supports_partial_effects=True,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="elastic_net_cox",
        tier=TIER_COMPARISON,
        source_registry=SOURCE_CORRECTED_MODEL_TRAINING,
        feature_set="core",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=True,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="glmnet_ridge_core",
        tier=TIER_EXPLORATORY_SUPPLEMENTARY,
        source_registry=SOURCE_EXPLORATORY_OPTIMIZATION,
        feature_set="core",
        runtime=RUNTIME_R,
        sparsity_capable=False,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="glmnet_enet_core",
        tier=TIER_EXPLORATORY_SUPPLEMENTARY,
        source_registry=SOURCE_EXPLORATORY_OPTIMIZATION,
        feature_set="core",
        runtime=RUNTIME_R,
        sparsity_capable=True,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="penalized_nonlinear_cox",
        tier=TIER_EXPLORATORY_SUPPLEMENTARY,
        source_registry=SOURCE_EXPLORATORY_OPTIMIZATION,
        feature_set="core",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=False,
        supports_partial_effects=True,
        supports_admitted_interaction=True,
    ),
    ModelRegistration(
        model_name="reduced_core_penalized_cox",
        tier=TIER_EXPLORATORY_SUPPLEMENTARY,
        source_registry=SOURCE_EXPLORATORY_OPTIMIZATION,
        feature_set="core",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=False,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
    ),
    ModelRegistration(
        model_name="core_cox",
        tier=TIER_NOT_EXPLAINED,
        source_registry=SOURCE_FROZEN_MODEL_TRAINING,
        feature_set="core",
        runtime=RUNTIME_PYTHON,
        sparsity_capable=False,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
        not_explained_reason=REASON_FIT_FAILED_SINGULAR_DESIGN,
        not_explained_fit_success_count=3,
        not_explained_fit_total_count=106,
    ),
    ModelRegistration(
        model_name="flexsurv_rp_spline",
        tier=TIER_NOT_EXPLAINED,
        source_registry=SOURCE_EXPLORATORY_OPTIMIZATION,
        feature_set="baseline",
        runtime=RUNTIME_R,
        sparsity_capable=False,
        supports_partial_effects=False,
        supports_admitted_interaction=False,
        not_explained_reason=REASON_FIT_INSTABILITY_EXPLANATION_UNRELIABLE,
        not_explained_fit_success_count=24,
        not_explained_fit_total_count=56,
    ),
)

MODEL_REGISTER_BY_NAME: dict[str, ModelRegistration] = {
    registration.model_name: registration for registration in MODEL_REGISTER
}

EXPLAINED_MODEL_NAMES: tuple[str, ...] = tuple(
    registration.model_name for registration in MODEL_REGISTER if registration.is_explained
)
NOT_EXPLAINED_MODEL_NAMES: tuple[str, ...] = tuple(
    registration.model_name for registration in MODEL_REGISTER if not registration.is_explained
)
ALL_MODEL_NAMES: tuple[str, ...] = tuple(registration.model_name for registration in MODEL_REGISTER)

if len(ALL_MODEL_NAMES) != 9 or len(EXPLAINED_MODEL_NAMES) != 7:
    raise ExplainContractError("Phase 6 model register does not match its frozen inventory")


def registration_for(model_name: str) -> ModelRegistration:
    if model_name not in MODEL_REGISTER_BY_NAME:
        raise ExplainContractError(f"'{model_name}' is not a registered Phase 6 model")
    return MODEL_REGISTER_BY_NAME[model_name]


# `explainability-spec.md`, E1/E2/E5: sparsity-capable models are exactly the two whose penalty can
# zero a coefficient (`elastic_net_cox` at its selected `l1_ratio`, `glmnet_enet_core`). Pure-ridge
# candidates never produce a selection-frequency number.
SPARSITY_CAPABLE_MODEL_NAMES: tuple[str, ...] = tuple(
    registration.model_name for registration in MODEL_REGISTER if registration.sparsity_capable
)

# `explainability-spec.md`, E4: only the two spline-based Cox models support partial-effect curves.
PARTIAL_EFFECT_MODEL_NAMES: tuple[str, ...] = tuple(
    registration.model_name
    for registration in MODEL_REGISTER
    if registration.supports_partial_effects
)

R_PERM = 50
B_EXPLAIN = 200
ROOT_SEED = 42

# `explainability-spec.md`'s 2026-08-11 bootstrap-importance amendment: the inner permutation
# repeat count used *inside* each of the `B_EXPLAIN` refits (common random numbers across refits
# via the refit-independent `importance.permutation_seed`). Frozen 2026-08-12 from the
# prespecified nested canary against real data (`glmnet_enet_core`+`minimal_cox`, OS/DFCI): R_boot
# 10, 20, and 40 each failed the binding 730-day MCSE tolerance (driven by
# `metastatic_presentation`/`ca_d_site`); the 40->80 comparison passed cleanly on every metric for
# both scopes. See the spec amendment for the full canary record.
R_BOOT = 80

__all__ = [
    "ALL_MODEL_NAMES",
    "ANALYSIS_STATUS_POST_HOC_EXPLORATORY",
    "ANALYSIS_STATUS_PRIMARY",
    "ANALYSIS_STATUS_TECHNICAL_CORRECTION",
    "B_EXPLAIN",
    "EXPLAINED_MODEL_NAMES",
    "EXPLANATION_HORIZONS",
    "MIN_BOOTSTRAP_FINITE_FRACTION",
    "MIN_ELIGIBLE_EVENTS",
    "MIN_EVENT_FREE",
    "MODEL_REGISTER",
    "MODEL_REGISTER_BY_NAME",
    "NOT_EXPLAINED_MODEL_NAMES",
    "PARTIAL_EFFECT_MODEL_NAMES",
    "PRIMARY_EXPLANATION_HORIZON",
    "REASON_FIT_FAILED_SINGULAR_DESIGN",
    "REASON_FIT_INSTABILITY_EXPLANATION_UNRELIABLE",
    "REPORTING_ESTIMABLE",
    "REPORTING_IMPRECISE",
    "REPORTING_NOT_COMPUTABLE",
    "ROOT_SEED",
    "RUNTIME_PYTHON",
    "RUNTIME_R",
    "R_BOOT",
    "R_PERM",
    "SOURCE_FROZEN_MODEL_TRAINING",
    "SOURCE_CORRECTED_MODEL_TRAINING",
    "SOURCE_EXPLORATORY_OPTIMIZATION",
    "SPARSITY_CAPABLE_MODEL_NAMES",
    "TIER_COMPARISON",
    "TIER_EXPLORATORY_SUPPLEMENTARY",
    "TIER_NOT_EXPLAINED",
    "TIER_PRIMARY_REFERENCE",
    "ExplainContractError",
    "ModelRegistration",
    "registration_for",
]

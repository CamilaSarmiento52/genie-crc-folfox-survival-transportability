"""Sensitivity-analysis workflow public contracts: frozen constants, scope, and shared dataclasses.

Per `sensitivity-analyses-spec.md`, sensitivity-analysis workflow reuses the frozen model-training
and primary-evaluation delayed-entry machinery by
import and never refits a primary model. The model scope is restricted to `minimal_cox` and
`spline_cox` -- the same Tier 1/2 pair the frozen spec already names for calibration drift -- so
every model here is loaded from the single frozen model-training registry; the corrected
optimization
`elastic_net_cox` registry and the explainability model-availability hierarchy are not consumed as
data
inputs (see `implement.md`, "Model scope").
"""

from __future__ import annotations

from dataclasses import dataclass

from crc_folfox.evaluate.contracts import CENTRES, HORIZONS, IBS_GRID

SENSITIVITY_MODEL_NAMES = ("minimal_cox", "spline_cox")

# The four locked sequencing-lag thresholds, in order, from tech-stack.md's "Sequencing-lag
# sensitivity (headline figure)". `None` marks the "by FOLFOX initiation" stratum (entry == 0
# exactly); `float("inf")` marks "unrestricted". This exact tuple is asserted immutable by test --
# no caller may add, remove, or reorder a threshold.
LAG_BY_FOLFOX = "by_folfox"
LAG_UNRESTRICTED = "unrestricted"
SEQUENCING_LAG_THRESHOLDS: tuple[tuple[str, float], ...] = (
    (LAG_BY_FOLFOX, 0.0),
    ("90d", 90.0),
    ("180d", 180.0),
    ("365d", 365.0),
    (LAG_UNRESTRICTED, float("inf")),
)

# The required conclusion sentence for the conditional-Kendall's-tau truncation diagnostic --
# templated, non-optional report/notebook text (sensitivity-analyses-spec.md, item 2).
TRUNCATION_CONCLUSION_REJECTED = (
    "Evidence against quasi-independence was detected; however, the diagnostic cannot establish "
    "that sequencing-related cohort entry was non-informative."
)
TRUNCATION_CONCLUSION_NOT_REJECTED = (
    "Evidence against quasi-independence was not detected; however, the diagnostic cannot "
    "establish that sequencing-related cohort entry was non-informative."
)

# The required complete-case-subset limitation sentence (sensitivity-analyses-spec.md, item 3).
COMPLETE_CASE_LIMITATION = (
    "This tests whether reported performance changes among patients with complete predictor data. "
    "It does not test how model fitting would change without imputation. A genuine "
    "imputation-method sensitivity would require complete-case refitting or multiple imputation "
    "and would be a separate post_hoc_exploratory analysis, out of Phase 7's scope."
)

ANALYSIS_STATUS_LOCAL_ADAPTATION = "local_adaptation"

# Complete-case predictors: `crc_type` and `ca_grade` are the only two preprocessing workflow
# mode-imputed
# features (tech-stack.md, "Preprocessing rules"); `ca_grade` additionally carries a fold-fitted
# missingness indicator (`MISSING_INDICATORS = {"ca_grade"}` in `preprocessing/transform.py`).
# "Complete case" is recomputed directly from the raw data-cleaning workflow columns rather than
# from a persisted
# preprocessing artifact, because preprocessing writes no patient-level file (see `implement.md`).
COMPLETE_CASE_COLUMNS = ("crc_type", "ca_grade")

# concordance/auc/brier/ibs/slope/citl scored on the unconditional risk scale only, at the two
# locked horizons -- matching the primary evaluation's risk-scale emphasis and bounding the
# sensitivity analysis's scope
# (see implement.md, "Risk scale"). `None` horizon marks a horizon-independent metric.
SUBSET_METRIC_SPEC: tuple[tuple[str, float | None], ...] = (
    ("concordance", None),
    ("ibs", None),
    ("auc", 365.0),
    ("auc", 730.0),
    ("brier", 365.0),
    ("brier", 730.0),
    ("slope", 365.0),
    ("slope", 730.0),
    ("citl", 365.0),
    ("citl", 730.0),
)

CONCORDANCE_CLASSIFICATION_CONCORDANT = "concordant"
CONCORDANCE_CLASSIFICATION_DISCORDANT = "discordant"
CONCORDANCE_CLASSIFICATION_INDETERMINATE = "indeterminate"

POPULATION_VIEW_NATIVE = "native"
POPULATION_VIEW_DUAL_ELIGIBLE = "dual_eligible"


class SensitivityContractError(ValueError):
    """Raised for a privacy-safe sensitivity-analysis workflow contract violation."""


@dataclass(frozen=True)
class SensitivityScope:
    """One `(endpoint, design, centre, model)` sensitivity cell -- always a restriction of an
    already-computed primary-evaluation-shaped prediction, never a new fit."""

    endpoint: str
    design: str
    centre: str
    model_name: str

    def __post_init__(self) -> None:
        if self.design not in ("iecv", "pooled"):
            raise SensitivityContractError("Unknown sensitivity design")
        if self.centre not in CENTRES:
            raise SensitivityContractError("Unknown centre")
        if self.model_name not in SENSITIVITY_MODEL_NAMES:
            raise SensitivityContractError("Unknown sensitivity model")


__all__ = [
    "ANALYSIS_STATUS_LOCAL_ADAPTATION",
    "COMPLETE_CASE_COLUMNS",
    "COMPLETE_CASE_LIMITATION",
    "CONCORDANCE_CLASSIFICATION_CONCORDANT",
    "CONCORDANCE_CLASSIFICATION_DISCORDANT",
    "CONCORDANCE_CLASSIFICATION_INDETERMINATE",
    "HORIZONS",
    "IBS_GRID",
    "LAG_BY_FOLFOX",
    "LAG_UNRESTRICTED",
    "POPULATION_VIEW_DUAL_ELIGIBLE",
    "POPULATION_VIEW_NATIVE",
    "SENSITIVITY_MODEL_NAMES",
    "SEQUENCING_LAG_THRESHOLDS",
    "SUBSET_METRIC_SPEC",
    "TRUNCATION_CONCLUSION_NOT_REJECTED",
    "TRUNCATION_CONCLUSION_REJECTED",
    "SensitivityContractError",
    "SensitivityScope",
]

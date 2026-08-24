"""Primary-evaluation workflow public contracts: evaluation scopes, the metric registry, and
frozen constants."""

from __future__ import annotations

from dataclasses import dataclass, field

from crc_folfox.model_training.contracts import HORIZONS, REQUIRED_MODEL_NAMES
from crc_folfox.preprocessing.contracts import INSTITUTIONS

EVAL_MODEL_NAMES = REQUIRED_MODEL_NAMES
DESIGNS = ("iecv", "pooled")
CENTRES = INSTITUTIONS
RISK_SCALES = ("unconditional", "entry_conditional")

# Fourteen 30-day grid points spanning the locked [365, 730]-day integrated-Brier window: thirteen
# steps of 30 days from 365 to 725, plus the closing point at 730 (see evaluation-spec.md).
IBS_WINDOW = (365.0, 730.0)
IBS_GRID = tuple(365.0 + 30.0 * step for step in range(13)) + (730.0,)
IBS_WINDOW_WIDTH = IBS_WINDOW[1] - IBS_WINDOW[0]

BOOTSTRAP_RESAMPLES = 2000
MIN_ADMISSIBLE_POOLED_REPEATS = 5
POOLED_REPEATS_TOTAL = 10

MIN_ELIGIBLE_EVENTS = 20
MIN_EVENT_FREE = 20
MIN_BOOTSTRAP_FINITE_FRACTION = 0.90

N_QUANTILE_GROUPS = 5
MIN_CALIBRATION_GROUP_SUPPORT = 10
MIN_CALIBRATION_CELL_SUPPORT = 25

RISK_CLIP = 1e-6

REPORTING_ESTIMABLE = "estimable"
REPORTING_IMPRECISE = "estimable_imprecise"
REPORTING_NOT_COMPUTABLE = "not_computable"

DCA_OMISSION_REASON = "dca_gate_not_passed"


class EvaluationContractError(ValueError):
    """Raised for a privacy-safe primary-evaluation workflow contract violation."""


@dataclass(frozen=True)
class EvaluationScope:
    """One `(endpoint, design, centre, model)` evaluation cell -- 48 exist in total."""

    endpoint: str
    design: str
    centre: str
    model_name: str

    def __post_init__(self) -> None:
        if self.design not in DESIGNS:
            raise EvaluationContractError("Unknown evaluation design")
        if self.centre not in CENTRES:
            raise EvaluationContractError("Unknown centre")
        if self.model_name not in EVAL_MODEL_NAMES:
            raise EvaluationContractError("Unknown evaluation model")

    @property
    def scope_id(self) -> str:
        return (
            f"endpoint={self.endpoint}|design={self.design}"
            f"|centre={self.centre}|model={self.model_name}"
        )


def all_evaluation_scopes(
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
) -> tuple[EvaluationScope, ...]:
    return tuple(
        EvaluationScope(endpoint=endpoint, design=design, centre=centre, model_name=model_name)
        for endpoint in endpoints
        for design in DESIGNS
        for centre in CENTRES
        for model_name in EVAL_MODEL_NAMES
    )


# `higher_is_better=None` marks a metric with no comparator-direction meaning (e.g. calibration
# quantities near zero/one) -- see `evaluation-spec.md`, "Frozen metric set" and "Locked precision
# targets". Precision targets are the maximum acceptable 95% interval width for "estimable".
@dataclass(frozen=True)
class MetricDefinition:
    name: str
    horizon_specific: bool
    risk_scale_specific: bool
    higher_is_better: bool | None
    precision_target: float | None
    delayed_entry_evidence: str


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "concordance": MetricDefinition(
        "concordance",
        False,
        False,
        True,
        0.25,
        "Truncation-aware pair comparability requires entry_i, entry_j <= min(exit_i, exit_j); "
        "entry-only perturbation changes comparable pairs (test_concordance.py).",
    ),
    "auc": MetricDefinition(
        "auc",
        True,
        False,
        True,
        0.25,
        "Eligible set is entry <= horizon; IPCW weights from delayed-entry censoring KM "
        "(test_auc.py).",
    ),
    "brier": MetricDefinition(
        "brier",
        True,
        False,
        False,
        0.10,
        "Reuses crc_folfox.model_training.tuning.delayed_entry_ipcw_brier, Phase 4 Gate 3 "
        "validated (test_prediction_error.py).",
    ),
    "ibs": MetricDefinition(
        "ibs",
        False,
        False,
        False,
        0.10,
        "Trapezoidal integral of the reused per-grid-point delayed-entry Brier core over "
        "[365, 730] (test_prediction_error.py).",
    ),
    "slope": MetricDefinition(
        "slope",
        True,
        True,
        None,
        1.00,
        "Weighted logistic regression of horizon event indicator on cloglog(risk), IPCW-weighted "
        "by the delayed-entry censoring KM (test_calibration.py).",
    ),
    "citl": MetricDefinition(
        "citl",
        True,
        True,
        None,
        0.50,
        "Same weighted logistic fit with cloglog(risk) as a fixed offset (slope constrained to 1) "
        "(test_calibration.py).",
    ),
    "prognostic_index_slope": MetricDefinition(
        "prognostic_index_slope",
        False,
        False,
        None,
        1.00,
        "Delayed-entry Cox fit (entry_col) with the linear predictor as sole covariate "
        "(test_calibration.py).",
    ),
}

INELIGIBLE_IMPLEMENTATIONS = (
    "lifelines.utils.concordance_index",
    "sksurv.metrics.concordance_index_censored",
    "sksurv.metrics.concordance_index_ipcw",
    "sksurv.metrics.cumulative_dynamic_auc",
    "sksurv.metrics.brier_score",
    "sksurv.metrics.integrated_brier_score",
)


@dataclass(frozen=True)
class MetricResult:
    """One reported metric value: an estimate, its bootstrap interval, and reporting status."""

    metric: str
    value: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_eligible: int
    n_events: int
    n_event_free: int
    bootstrap_finite_fraction: float
    reporting_status: str
    reason: str = ""
    extra: dict[str, object] = field(default_factory=dict)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CENTRES",
    "DCA_OMISSION_REASON",
    "DESIGNS",
    "EVAL_MODEL_NAMES",
    "HORIZONS",
    "IBS_GRID",
    "IBS_WINDOW",
    "IBS_WINDOW_WIDTH",
    "INELIGIBLE_IMPLEMENTATIONS",
    "METRIC_REGISTRY",
    "MIN_ADMISSIBLE_POOLED_REPEATS",
    "MIN_BOOTSTRAP_FINITE_FRACTION",
    "MIN_CALIBRATION_CELL_SUPPORT",
    "MIN_CALIBRATION_GROUP_SUPPORT",
    "MIN_ELIGIBLE_EVENTS",
    "MIN_EVENT_FREE",
    "N_QUANTILE_GROUPS",
    "POOLED_REPEATS_TOTAL",
    "REPORTING_ESTIMABLE",
    "REPORTING_IMPRECISE",
    "REPORTING_NOT_COMPUTABLE",
    "RISK_CLIP",
    "RISK_SCALES",
    "EvaluationContractError",
    "EvaluationScope",
    "MetricDefinition",
    "MetricResult",
    "all_evaluation_scopes",
]

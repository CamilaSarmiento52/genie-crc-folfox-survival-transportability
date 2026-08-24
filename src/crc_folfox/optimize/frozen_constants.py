"""Re-export the model-training and primary-evaluation constants optimization workflow must reuse,
never redefine.

`correction-spec.md` and `optimization-spec.md` both require every grid value, split rule,
endpoint definition, horizon, metric registry entry, bootstrap seed tuple, reporting-status gate,
precision target, and materiality margin to come from the frozen model-training and
primary-evaluation modules by import. This
module is the single place that import happens, so a static or identity test can assert optimization
workflow
never shadows one of these names with a locally redefined value.
"""

from __future__ import annotations

from crc_folfox.evaluate.comparisons import MATERIALITY_MARGINS, classify_complexity_cell
from crc_folfox.evaluate.contracts import (
    BOOTSTRAP_RESAMPLES,
    DESIGNS,
    HORIZONS,
    IBS_GRID,
    IBS_WINDOW,
    IBS_WINDOW_WIDTH,
    METRIC_REGISTRY,
    MIN_ADMISSIBLE_POOLED_REPEATS,
    MIN_BOOTSTRAP_FINITE_FRACTION,
    MIN_ELIGIBLE_EVENTS,
    MIN_EVENT_FREE,
    POOLED_REPEATS_TOTAL,
    REPORTING_ESTIMABLE,
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
    RISK_SCALES,
)
from crc_folfox.evaluate.reporting import classify_reporting_status
from crc_folfox.model_training.contracts import (
    ELASTIC_NET_GRID,
    REQUIRED_MODEL_NAMES,
)
from crc_folfox.preprocessing.contracts import INSTITUTIONS
from crc_folfox.preprocessing.splits import ROOT_SEED

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "DESIGNS",
    "ELASTIC_NET_GRID",
    "HORIZONS",
    "IBS_GRID",
    "IBS_WINDOW",
    "IBS_WINDOW_WIDTH",
    "INSTITUTIONS",
    "MATERIALITY_MARGINS",
    "METRIC_REGISTRY",
    "MIN_ADMISSIBLE_POOLED_REPEATS",
    "MIN_BOOTSTRAP_FINITE_FRACTION",
    "MIN_ELIGIBLE_EVENTS",
    "MIN_EVENT_FREE",
    "POOLED_REPEATS_TOTAL",
    "REPORTING_ESTIMABLE",
    "REPORTING_IMPRECISE",
    "REPORTING_NOT_COMPUTABLE",
    "REQUIRED_MODEL_NAMES",
    "RISK_SCALES",
    "ROOT_SEED",
    "classify_complexity_cell",
    "classify_reporting_status",
]

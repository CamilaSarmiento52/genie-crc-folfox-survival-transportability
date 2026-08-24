"""Transportability gap and complexity-value gains, with the locked mechanical classification.

Both compute a paired point difference and a paired bootstrap interval from *per-resample*
differences (not from the two marginal intervals), using the shared `(endpoint, centre)` index
matrix from `crc_folfox.evaluate.bootstrap` so pairing is exact. See `evaluation-spec.md`,
"Transportability gap" and "Complexity value".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

MATERIALITY_MARGINS: dict[str, float] = {
    "auc": 0.02,
    "ibs": 0.005,
    "citl": 0.05,
    "slope": 0.05,
}


def _paired_interval(diffs: np.ndarray) -> tuple[float | None, float | None]:
    finite = diffs[np.isfinite(diffs)]
    if not len(finite):
        return None, None
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


@dataclass(frozen=True)
class TransportabilityGap:
    value: float | None
    ci_lower: float | None
    ci_upper: float | None
    higher_is_better: bool
    pooled_computable: bool
    iecv_computable: bool
    reason: str = ""


def compute_transportability_gap(
    *,
    pooled_point: float | None,
    pooled_resamples: np.ndarray,
    iecv_point: float | None,
    iecv_resamples: np.ndarray,
    higher_is_better: bool,
) -> TransportabilityGap:
    """`gap = pooled - iecv`, matching the locked `higher_is_better` sign convention per metric."""
    if pooled_point is None or iecv_point is None:
        return TransportabilityGap(
            None,
            None,
            None,
            higher_is_better,
            pooled_computable=pooled_point is not None,
            iecv_computable=iecv_point is not None,
            reason="pooled_not_computable" if pooled_point is None else "iecv_not_computable",
        )
    value = float(pooled_point) - float(iecv_point)
    diffs = np.asarray(pooled_resamples, dtype=float) - np.asarray(iecv_resamples, dtype=float)
    lower, upper = _paired_interval(diffs)
    reason = "" if lower is not None else "no_finite_paired_resamples"
    return TransportabilityGap(value, lower, upper, higher_is_better, True, True, reason)


def _auc_gain(fuller: np.ndarray, minimal: np.ndarray) -> np.ndarray:
    return fuller - minimal


def _ibs_gain(fuller: np.ndarray, minimal: np.ndarray) -> np.ndarray:
    return minimal - fuller


def _citl_gain(fuller: np.ndarray, minimal: np.ndarray) -> np.ndarray:
    return np.abs(minimal) - np.abs(fuller)


def _slope_gain(fuller: np.ndarray, minimal: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (fuller > 0) & (minimal > 0),
            np.abs(np.log(minimal)) - np.abs(np.log(fuller)),
            np.nan,
        )


_GAIN_FUNCS = {"auc": _auc_gain, "ibs": _ibs_gain, "citl": _citl_gain, "slope": _slope_gain}


@dataclass(frozen=True)
class ComplexityGain:
    kind: str
    value: float | None
    ci_lower: float | None
    ci_upper: float | None
    fuller_value: float | None
    minimal_value: float | None
    reason: str = ""


def compute_complexity_gain(
    kind: str,
    *,
    fuller_point: float | None,
    fuller_resamples: np.ndarray,
    minimal_point: float | None,
    minimal_resamples: np.ndarray,
) -> ComplexityGain:
    func = _GAIN_FUNCS[kind]
    if fuller_point is None or minimal_point is None:
        return ComplexityGain(
            kind, None, None, None, fuller_point, minimal_point, "underlying_not_computable"
        )
    point = float(func(np.asarray(float(fuller_point)), np.asarray(float(minimal_point))))
    if not np.isfinite(point):
        return ComplexityGain(kind, None, None, None, fuller_point, minimal_point, "undefined_gain")
    diffs = func(
        np.asarray(fuller_resamples, dtype=float), np.asarray(minimal_resamples, dtype=float)
    )
    lower, upper = _paired_interval(diffs)
    reason = "" if lower is not None else "no_finite_paired_resamples"
    return ComplexityGain(kind, point, lower, upper, fuller_point, minimal_point, reason)


def classify_gain(gain: ComplexityGain) -> str:
    """'favourable' / 'unfavourable' / 'neutral' / 'not_computable' for one gain."""
    if gain.value is None:
        return "not_computable"
    margin = MATERIALITY_MARGINS[gain.kind]
    if gain.ci_lower is not None and gain.ci_lower > 0 and gain.value >= margin:
        return "favourable"
    if gain.ci_upper is not None and gain.ci_upper < 0 and abs(gain.value) >= margin:
        return "unfavourable"
    return "neutral"


def classify_complexity_cell(gains: Mapping[str, ComplexityGain]) -> str:
    """'clear_added_value' / 'trade_off_uncertain' / 'no_demonstrated_advantage', from the seven
    computable gains (`net_benefit_gain` is always `not_applicable`, excluded from this count)."""
    computable = {name: gain for name, gain in gains.items() if gain.value is not None}
    if len(computable) < 4:
        return "trade_off_uncertain"
    labels = {name: classify_gain(gain) for name, gain in computable.items()}
    favourable = any(label == "favourable" for label in labels.values())
    unfavourable = any(label == "unfavourable" for label in labels.values())
    if favourable and unfavourable:
        return "trade_off_uncertain"
    if favourable:
        return "clear_added_value"
    return "no_demonstrated_advantage"


__all__ = [
    "MATERIALITY_MARGINS",
    "ComplexityGain",
    "TransportabilityGap",
    "classify_complexity_cell",
    "classify_gain",
    "compute_complexity_gain",
    "compute_transportability_gap",
]

"""Supportive endpoint robustness: does an OS sensitivity finding hold for supportive TTNT-D?

Named "supportive endpoint robustness," not "endpoint sensitivity" or "endpoint validation," so the
label itself never implies TTNT-D is an interchangeable endpoint (`mission.md` non-negotiable 1).
Every OS sensitivity above is repeated identically for TTNT-D on its own frozen eligible population
by the same shared components (`sequencing_lag`, `truncation`, `complete_case`) called a second time
with the TTNT-D endpoint view; this module owns only the two things specific to the *comparison*
itself: the concordant/discordant/indeterminate classifier, and the dual-eligible secondary
population view.
"""

from __future__ import annotations

import pandas as pd

from crc_folfox.sensitivity.contracts import (
    CONCORDANCE_CLASSIFICATION_CONCORDANT,
    CONCORDANCE_CLASSIFICATION_DISCORDANT,
    CONCORDANCE_CLASSIFICATION_INDETERMINATE,
)

# Analogous in spirit to `evaluate.contracts.MIN_ELIGIBLE_EVENTS`, applied to the dual-eligible
# secondary population view before it is computed at all.
DUAL_ELIGIBLE_MIN_EVENTS = 20


def classify_concordance(
    *,
    os_effect: float | None,
    os_status: str,
    ttnt_effect: float | None,
    ttnt_status: str,
    materiality_margin: float = 0.0,
) -> tuple[str, str]:
    """Classify one OS-versus-TTNT-D comparison.

    Never decides from a p-value alone: a cell that is `not_computable` or `estimable_imprecise` on
    *either* endpoint is always `indeterminate`, regardless of how large or "significant" its point
    estimate looks -- only when both sides are `estimable` does direction/materiality decide
    concordant versus discordant.
    """
    if os_status != "estimable" or ttnt_status != "estimable":
        return CONCORDANCE_CLASSIFICATION_INDETERMINATE, "insufficient_reporting_status"
    if os_effect is None or ttnt_effect is None:
        return CONCORDANCE_CLASSIFICATION_INDETERMINATE, "undefined_effect"

    os_material = abs(os_effect) > materiality_margin
    ttnt_material = abs(ttnt_effect) > materiality_margin
    if not os_material and not ttnt_material:
        return CONCORDANCE_CLASSIFICATION_CONCORDANT, "both_within_materiality_margin"
    if os_material != ttnt_material:
        return CONCORDANCE_CLASSIFICATION_DISCORDANT, "material_on_only_one_endpoint"
    same_sign = (os_effect > 0) == (ttnt_effect > 0)
    if same_sign:
        return CONCORDANCE_CLASSIFICATION_CONCORDANT, ""
    return CONCORDANCE_CLASSIFICATION_DISCORDANT, ""


def dual_eligible_frames(
    os_frame: pd.DataFrame, ttnt_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both endpoint frames to the patients eligible for both -- the secondary population
    view, used to test whether population composition (rather than the endpoint definition itself)
    explains an observed OS-versus-TTNT-D difference."""
    shared_keys = set(os_frame["patient_key"].astype(str)) & set(
        ttnt_frame["patient_key"].astype(str)
    )
    os_dual = (
        os_frame.loc[os_frame["patient_key"].astype(str).isin(shared_keys)]
        .sort_values("patient_key", kind="stable")
        .reset_index(drop=True)
    )
    ttnt_dual = (
        ttnt_frame.loc[ttnt_frame["patient_key"].astype(str).isin(shared_keys)]
        .sort_values("patient_key", kind="stable")
        .reset_index(drop=True)
    )
    return os_dual, ttnt_dual


def dual_eligible_support_ok(frame: pd.DataFrame) -> bool:
    """The dual-eligible view's own support gate -- computed *before* the view is used for
    anything, per the spec's "gated by its own support check"."""
    if frame.empty or "event" not in frame.columns:
        return False
    return int(frame["event"].sum()) >= DUAL_ELIGIBLE_MIN_EVENTS


__all__ = [
    "DUAL_ELIGIBLE_MIN_EVENTS",
    "classify_concordance",
    "dual_eligible_frames",
    "dual_eligible_support_ok",
]

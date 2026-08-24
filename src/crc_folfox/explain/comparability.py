"""Comparability rules R1-R6 (`explainability-spec.md`, "Comparability rules") and E5 attribution
stability.

The rules exist because the model-availability hierarchy has unequal scope coverage across
centres, models, and tiers, and the single most likely misreading of this workflow's output is
mistaking a coverage gap for a genuine attribution difference. R4 is binding: an apparent
difference that coincides with a fit-availability gap is reported as an availability artifact and
is never described as biological, clinical, or institutional heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

from crc_folfox.explain.contracts import MODEL_REGISTER_BY_NAME

SCOPE_BASIS_IDENTICAL = "identical_scopes"
SCOPE_BASIS_RESTRICTED = "restricted_scopes"

COMPARABLE = "comparable"
NOT_COMPARABLE = "not_comparable"

REASON_MODEL_UNAVAILABLE_IN_CENTRE = "model_unavailable_in_centre"


# --- R1: comparable scope sets --------------------------------------------------------------


def comparable_centres(availability: dict[str, bool]) -> tuple[str, ...]:
    """R1: the centres in which *every* model of interest is `explained` -- the only centres a
    cross-model comparison for this group of models may use."""
    return tuple(sorted(centre for centre, available in availability.items() if available))


# --- R2: cross-centre comparisons need the same model in both centres -------------------------


@dataclass(frozen=True)
class PairComparability:
    status: str  # COMPARABLE | NOT_COMPARABLE
    reason: str = ""


def cross_centre_pair_status(
    available_in_first: bool, available_in_second: bool
) -> PairComparability:
    """R2: a cross-centre pair is admissible only when the model is `explained` in both centres."""
    if available_in_first and available_in_second:
        return PairComparability(COMPARABLE)
    return PairComparability(NOT_COMPARABLE, REASON_MODEL_UNAVAILABLE_IN_CENTRE)


# --- R3/R4: scope basis and the interpretation-rule label --------------------------------------


@dataclass(frozen=True)
class ScopeBasisLabel:
    scope_basis: str
    availability_confounded: bool
    availability_artifact_possible: bool


def label_scope_basis(*, is_restricted: bool, is_cross_tier: bool) -> ScopeBasisLabel:
    """R3 + R4: `restricted_scopes` and every cross-tier comparison carry
    `availability_artifact_possible = True` -- the machine-readable form of the rule that an
    attribution difference coinciding with an availability gap is never reported as
    heterogeneity."""
    basis = SCOPE_BASIS_RESTRICTED if is_restricted else SCOPE_BASIS_IDENTICAL
    return ScopeBasisLabel(
        scope_basis=basis,
        availability_confounded=is_restricted,
        availability_artifact_possible=is_restricted or is_cross_tier,
    )


# --- R5: same definition, same scale ------------------------------------------------------------


class ComparabilityContractError(ValueError):
    """Raised when two attribution quantities cannot be compared under R5."""


def assert_comparable_quantities(
    *,
    metric_a: str,
    metric_b: str,
    horizon_a: float | None,
    horizon_b: float | None,
    grouping_a: str,
    grouping_b: str,
) -> None:
    """R5: refuse a comparison whose definition, scale, horizon, or grouping differs."""
    if (metric_a, horizon_a, grouping_a) != (metric_b, horizon_b, grouping_b):
        raise ComparabilityContractError(
            "Cannot compare attribution quantities with different definition, scale, horizon, "
            "or grouping"
        )


# --- E5: attribution stability -----------------------------------------------------------------


@dataclass(frozen=True)
class AttributionStabilityRow:
    comparison_type: str  # "cross_centre" | "cross_endpoint" | "cross_model"
    model_name: str
    key_a: str  # centre, endpoint, or "model_vs_minimal_cox" label
    key_b: str
    scope_basis: str
    availability_confounded: bool
    availability_artifact_possible: bool
    comparison_status: str
    reason: str = ""
    kendall_tau: float | None = None
    top_k: int = 0
    topk_overlap: int | None = None
    jaccard: float | None = None
    top_k_boundary_tied: bool = False
    sign_agreement: float | None = None
    n_groups: int = 0


def top_k_for_agreement(n_shared: int) -> int:
    """`k = min(5, max(1, floor(G/2)))`, `G` the shared-group count -- only meaningful once
    `G >= 2`; the caller checks the shared-groups floor first (`explainability-spec.md`'s
    2026-08-11 amendment, replacing a hardcoded top-5 that was degenerate for `minimal_cox`'s 4
    groups)."""
    return min(5, max(1, n_shared // 2))


@dataclass(frozen=True)
class RankAgreementResult:
    kendall_tau: float | None
    top_k: int
    topk_overlap: int
    jaccard: float | None
    top_k_boundary_tied: bool


def _deterministic_ranking(values: dict[str, float], shared: list[str]) -> list[str]:
    """Descending by value, ties broken alphabetically by group name -- the frozen secondary
    ordering used only for top-k selection/display, never for the rank statistic itself."""
    return sorted(shared, key=lambda g: (-values[g], g))


def rank_agreement(values_a: dict[str, float], values_b: dict[str, float]) -> RankAgreementResult:
    """Kendall's tau-b between two group->importance rankings, restricted to their shared groups,
    plus the adaptive top-k overlap count and Jaccard similarity.

    Below 2 shared groups, the whole comparison (tau, top-k, Jaccard) is `not_comparable` --
    checked *before* the adaptive-`k` formula, so a 0- or 1-shared-group case never falls through
    to a spuriously "valid" `k=1`.
    """
    shared = sorted(set(values_a) & set(values_b))
    if len(shared) < 2:
        return RankAgreementResult(None, 0, 0, None, False)
    a = np.array([values_a[g] for g in shared])
    b = np.array([values_b[g] for g in shared])
    tau, _p_value = kendalltau(a, b)
    k = top_k_for_agreement(len(shared))

    def _top_k(values: dict[str, float]) -> tuple[set[str], bool]:
        ordered = _deterministic_ranking(values, shared)
        selected = set(ordered[:k])
        boundary_tied = k < len(ordered) and values[ordered[k - 1]] == values[ordered[k]]
        return selected, boundary_tied

    top_a, tied_a = _top_k(values_a)
    top_b, tied_b = _top_k(values_b)
    overlap = len(top_a & top_b)
    union_size = len(top_a | top_b)
    jaccard = overlap / union_size if union_size else None
    return RankAgreementResult(
        kendall_tau=float(tau) if np.isfinite(tau) else None,
        top_k=k,
        topk_overlap=overlap,
        jaccard=jaccard,
        top_k_boundary_tied=tied_a or tied_b,
    )


def sign_agreement(
    coefficients_a: dict[str, float], coefficients_b: dict[str, float]
) -> float | None:
    """Proportion of shared groups whose dominant coefficient sign agrees (coefficient-bearing
    models only)."""
    shared = sorted(set(coefficients_a) & set(coefficients_b))
    if not shared:
        return None
    agreements = sum(1 for g in shared if np.sign(coefficients_a[g]) == np.sign(coefficients_b[g]))
    return agreements / len(shared)


def build_cross_centre_row(
    model_name: str,
    centre_a: str,
    centre_b: str,
    *,
    available_a: bool,
    available_b: bool,
    importance_a: dict[str, float] | None,
    importance_b: dict[str, float] | None,
    coefficients_a: dict[str, float] | None = None,
    coefficients_b: dict[str, float] | None = None,
) -> AttributionStabilityRow:
    pair = cross_centre_pair_status(available_a, available_b)
    if pair.status == NOT_COMPARABLE:
        return AttributionStabilityRow(
            comparison_type="cross_centre",
            model_name=model_name,
            key_a=centre_a,
            key_b=centre_b,
            scope_basis=SCOPE_BASIS_RESTRICTED,
            availability_confounded=True,
            availability_artifact_possible=True,
            comparison_status=NOT_COMPARABLE,
            reason=pair.reason,
        )
    agreement = rank_agreement(importance_a or {}, importance_b or {})
    signs = sign_agreement(coefficients_a or {}, coefficients_b or {}) if coefficients_a else None
    label = label_scope_basis(is_restricted=False, is_cross_tier=False)
    return AttributionStabilityRow(
        comparison_type="cross_centre",
        model_name=model_name,
        key_a=centre_a,
        key_b=centre_b,
        scope_basis=label.scope_basis,
        availability_confounded=label.availability_confounded,
        availability_artifact_possible=label.availability_artifact_possible,
        comparison_status=COMPARABLE,
        kendall_tau=agreement.kendall_tau,
        top_k=agreement.top_k,
        topk_overlap=agreement.topk_overlap,
        jaccard=agreement.jaccard,
        top_k_boundary_tied=agreement.top_k_boundary_tied,
        sign_agreement=signs,
        n_groups=len(set(importance_a or {}) & set(importance_b or {})),
    )


def build_cross_endpoint_row(
    model_name: str,
    centre: str,
    importance_os: dict[str, float],
    importance_ttnt: dict[str, float],
    coefficients_os: dict[str, float] | None = None,
    coefficients_ttnt: dict[str, float] | None = None,
) -> AttributionStabilityRow:
    agreement = rank_agreement(importance_os, importance_ttnt)
    signs = (
        sign_agreement(coefficients_os or {}, coefficients_ttnt or {}) if coefficients_os else None
    )
    label = label_scope_basis(is_restricted=False, is_cross_tier=False)
    return AttributionStabilityRow(
        comparison_type="cross_endpoint",
        model_name=model_name,
        key_a=f"OS@{centre}",
        key_b=f"TTNT-D@{centre}",
        scope_basis=label.scope_basis,
        availability_confounded=label.availability_confounded,
        availability_artifact_possible=label.availability_artifact_possible,
        comparison_status=COMPARABLE,
        kendall_tau=agreement.kendall_tau,
        top_k=agreement.top_k,
        topk_overlap=agreement.topk_overlap,
        jaccard=agreement.jaccard,
        top_k_boundary_tied=agreement.top_k_boundary_tied,
        sign_agreement=signs,
        n_groups=len(set(importance_os) & set(importance_ttnt)),
    )


def build_cross_model_row(
    endpoint: str,
    centre: str,
    model_name: str,
    importance_model: dict[str, float],
    importance_minimal: dict[str, float],
) -> AttributionStabilityRow:
    """Every non-`minimal_cox` model versus `minimal_cox`, restricted to the four shared
    minimal-set groups when the feature sets differ -- a `restricted_scopes` comparison by
    construction (R3), and a cross-tier one whenever `model_name` is Tier 2/3 (R4)."""
    shared = set(importance_model) & set(importance_minimal)
    is_restricted = shared != set(importance_model) or shared != set(importance_minimal)
    is_cross_tier = (
        MODEL_REGISTER_BY_NAME[model_name].tier != MODEL_REGISTER_BY_NAME["minimal_cox"].tier
    )
    agreement = rank_agreement(
        {g: importance_model[g] for g in shared}, {g: importance_minimal[g] for g in shared}
    )
    label = label_scope_basis(is_restricted=is_restricted, is_cross_tier=is_cross_tier)
    return AttributionStabilityRow(
        comparison_type="cross_model",
        model_name=model_name,
        key_a=f"{endpoint}@{centre}",
        key_b="minimal_cox",
        scope_basis=label.scope_basis,
        availability_confounded=label.availability_confounded,
        availability_artifact_possible=label.availability_artifact_possible,
        comparison_status=COMPARABLE,
        kendall_tau=agreement.kendall_tau,
        top_k=agreement.top_k,
        topk_overlap=agreement.topk_overlap,
        jaccard=agreement.jaccard,
        top_k_boundary_tied=agreement.top_k_boundary_tied,
        n_groups=len(shared),
    )


__all__ = [
    "COMPARABLE",
    "NOT_COMPARABLE",
    "REASON_MODEL_UNAVAILABLE_IN_CENTRE",
    "SCOPE_BASIS_IDENTICAL",
    "SCOPE_BASIS_RESTRICTED",
    "AttributionStabilityRow",
    "ComparabilityContractError",
    "PairComparability",
    "RankAgreementResult",
    "ScopeBasisLabel",
    "assert_comparable_quantities",
    "build_cross_centre_row",
    "build_cross_endpoint_row",
    "build_cross_model_row",
    "comparable_centres",
    "cross_centre_pair_status",
    "label_scope_basis",
    "rank_agreement",
    "sign_agreement",
    "top_k_for_agreement",
]

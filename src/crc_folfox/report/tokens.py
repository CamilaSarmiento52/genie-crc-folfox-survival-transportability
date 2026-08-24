"""Manuscript token grammar and resolver: every manuscript number is extracted, never typed.

`manuscript-spec.md` section 4.1: no numeric claim anywhere in the manuscript is written by hand.
Source prose carries a token; the build resolves it from a published aggregate artifact cell and
fails loudly on an unknown token, an ambiguous match, an empty match, or a `not_computable` cell
whose prose does not also carry that status.

**Grammar.** `{{source.key1.key2....value}}` -- a registered source name, an ordered *prefix* of
that source's key columns, and a final value column. Trailing keys may be omitted only when what
remains matches exactly one row; otherwise the build fails as ambiguous rather than silently picking
one. That is what makes the omission safe: `os.iecv.MSK.minimal_cox.concordance.estimate` resolves
because concordance has a single row, while the same shape for `slope` raises and forces the risk
scale to be named. Ambiguity is detected, never resolved by position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from crc_folfox.report.contracts import (
    IMPRECISE_PROSE_MARKERS,
    KIND_COUNT,
    KIND_METRIC,
    KIND_TEXT,
    NOT_COMPUTABLE_PROSE_MARKERS,
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
    TOKEN_SEGMENT_SEPARATOR,
    TokenResolutionError,
    rounding_for,
)
from crc_folfox.report.inputs import CanonicalInputs


@dataclass(frozen=True)
class TokenSource:
    """One artifact registered as a token namespace, with its ordered key columns."""

    name: str
    bundle: str
    artifact: str
    key_columns: tuple[str, ...]
    value_columns: frozenset[str]
    #: applied before any user-supplied key; how one artifact serves two endpoint namespaces
    fixed_filters: dict[str, Any] = field(default_factory=dict)
    #: column naming the metric, used to select the frozen rounding rule
    metric_column: str | None = None
    #: column carrying the three-way reporting status, when the artifact has one
    status_column: str | None = None
    #: value columns that are counts rather than metric-scale quantities
    count_columns: frozenset[str] = frozenset()
    #: value columns that are free text rather than numbers
    text_columns: frozenset[str] = frozenset()
    #: fixed metric name for artifacts whose rows are all one metric class
    metric_override: str | None = None


_PERFORMANCE_VALUES = frozenset(
    {
        "estimate",
        "ci_lower",
        "ci_upper",
        "n_eligible",
        "n_events",
        "n_event_free",
        "reporting_status",
        "reason",
    }
)
_PERFORMANCE_KEYS = ("design", "centre", "model", "metric", "horizon", "risk_scale")
_PERFORMANCE_COUNTS = frozenset({"n_eligible", "n_events", "n_event_free"})
_PERFORMANCE_TEXT = frozenset({"reporting_status", "reason"})


def _performance_source(name: str, endpoint: str) -> TokenSource:
    return TokenSource(
        name=name,
        bundle="phase5_evaluation",
        artifact="performance_summary.csv",
        key_columns=_PERFORMANCE_KEYS,
        value_columns=_PERFORMANCE_VALUES,
        fixed_filters={"endpoint": endpoint},
        metric_column="metric",
        status_column="reporting_status",
        count_columns=_PERFORMANCE_COUNTS,
        text_columns=_PERFORMANCE_TEXT,
    )


def _transportability_source(name: str, endpoint: str) -> TokenSource:
    return TokenSource(
        name=name,
        bundle="phase5_evaluation",
        artifact="transportability_summary.csv",
        key_columns=("centre", "model", "metric", "horizon"),
        value_columns=frozenset(
            {
                "gap",
                "gap_ci_lower",
                "gap_ci_upper",
                "gap_reason",
                "pooled_estimate",
                "iecv_estimate",
                "pooled_status",
                "iecv_status",
                "pooled_n_events",
                "iecv_n_events",
            }
        ),
        fixed_filters={"endpoint": endpoint},
        metric_column="metric",
        status_column="iecv_status",
        count_columns=frozenset({"pooled_n_events", "iecv_n_events"}),
        text_columns=frozenset({"gap_reason", "pooled_status", "iecv_status"}),
    )


def _complexity_source(name: str, endpoint: str) -> TokenSource:
    return TokenSource(
        name=name,
        bundle="phase5_evaluation",
        artifact="complexity_value_summary.csv",
        key_columns=("design", "centre", "fuller_model", "gain", "horizon"),
        value_columns=frozenset(
            {
                "value",
                "ci_lower",
                "ci_upper",
                "classification",
                "fuller_estimate",
                "minimal_estimate",
                "reason",
                "net_benefit_gain",
            }
        ),
        fixed_filters={"endpoint": endpoint},
        metric_column="gain",
        status_column=None,
        text_columns=frozenset({"classification", "reason", "net_benefit_gain"}),
    )


#: The registered token namespaces. Adding one is a reporting decision and belongs in `tokens.md`.
TOKEN_SOURCES: dict[str, TokenSource] = {
    source.name: source
    for source in (
        # Performance -- the spec's worked example resolves against these two namespaces verbatim.
        _performance_source("os", "OS"),
        _performance_source("ttntd", "TTNT-D"),
        _transportability_source("os_gap", "OS"),
        _transportability_source("ttntd_gap", "TTNT-D"),
        _complexity_source("os_value", "OS"),
        _complexity_source("ttntd_value", "TTNT-D"),
        TokenSource(
            name="entry_support",
            bundle="phase5_evaluation",
            artifact="entry_support_summary.csv",
            key_columns=("endpoint", "group", "day"),
            value_columns=frozenset(
                {"n_eligible", "n_events", "n_event_free", "n_censored_before"}
            ),
            count_columns=frozenset(
                {"n_eligible", "n_events", "n_event_free", "n_censored_before"}
            ),
        ),
        TokenSource(
            name="coef",
            bundle="phase6_explainability",
            artifact="coefficient_summary.csv",
            key_columns=("endpoint", "centre", "model", "feature_group"),
            value_columns=frozenset(
                {"coefficient", "hazard_ratio", "ci_lower", "ci_upper", "standard_error"}
            ),
            metric_override="slope",
        ),
        TokenSource(
            name="attribution",
            bundle="phase6_explainability",
            artifact="attribution_stability.csv",
            key_columns=("comparison_type", "model_name", "endpoint", "key_a", "key_b"),
            value_columns=frozenset(
                {
                    "kendall_tau",
                    "jaccard",
                    "sign_agreement",
                    "topk_overlap",
                    "comparison_status",
                    "n_groups",
                }
            ),
            metric_override="tau",
            text_columns=frozenset({"comparison_status"}),
            count_columns=frozenset({"n_groups"}),
        ),
        TokenSource(
            name="lag",
            bundle="phase7_sensitivity",
            artifact="sequencing_lag_support.csv",
            key_columns=("endpoint", "threshold", "group"),
            value_columns=frozenset(
                {"n_patients", "n_events", "entry_median", "entry_mean", "entry_max"}
            ),
            count_columns=frozenset({"n_patients", "n_events"}),
            metric_override="tau",
        ),
        TokenSource(
            name="fit",
            bundle="phase4_training",
            artifact="fit_summary.csv",
            key_columns=("endpoint", "design", "held_out_institution", "model_name", "fit_status"),
            value_columns=frozenset({"development_patients", "development_events"}),
            count_columns=frozenset({"development_patients", "development_events"}),
        ),
        TokenSource(
            name="fact_count",
            bundle="phase8_generated",
            artifact="manuscript_facts.csv",
            key_columns=("fact",),
            value_columns=frozenset({"value"}),
            fixed_filters={"kind": "count"},
            count_columns=frozenset({"value"}),
        ),
        TokenSource(
            name="fact_metric",
            bundle="phase8_generated",
            artifact="manuscript_facts.csv",
            key_columns=("fact",),
            value_columns=frozenset({"value"}),
            fixed_filters={"kind": "metric"},
            metric_column="metric",
        ),
        TokenSource(
            name="sens_tau",
            bundle="phase7_sensitivity",
            artifact="truncation_kendall_tau.csv",
            key_columns=("endpoint", "stratum"),
            value_columns=frozenset(
                {
                    "tau",
                    "ci_lower",
                    "ci_upper",
                    "p_value",
                    "n_comparable_pairs",
                    "n_patients",
                    "n_events",
                    "reporting_status",
                    "conclusion",
                }
            ),
            metric_override="tau",
            status_column="reporting_status",
            count_columns=frozenset({"n_comparable_pairs", "n_patients", "n_events"}),
            text_columns=frozenset({"reporting_status", "conclusion"}),
        ),
        TokenSource(
            name="phase9_os",
            bundle="phase9_forest",
            artifact="performance_summary.csv",
            key_columns=_PERFORMANCE_KEYS,
            value_columns=_PERFORMANCE_VALUES,
            fixed_filters={"endpoint": "OS"},
            metric_column="metric",
            status_column="reporting_status",
            count_columns=_PERFORMANCE_COUNTS,
            text_columns=_PERFORMANCE_TEXT,
        ),
        TokenSource(
            name="phase9_value",
            bundle="phase9_forest",
            artifact="complexity_value_summary.csv",
            key_columns=("design", "centre", "fuller_model", "gain", "horizon"),
            value_columns=frozenset(
                {
                    "value",
                    "ci_lower",
                    "ci_upper",
                    "classification",
                    "fuller_estimate",
                    "minimal_estimate",
                    "reason",
                    "net_benefit_gain",
                }
            ),
            fixed_filters={"endpoint": "OS"},
            metric_column="gain",
            text_columns=frozenset({"classification", "reason", "net_benefit_gain"}),
        ),
        TokenSource(
            name="sex_perf",
            bundle="fairness_subgroups",
            artifact="sex_stratified_performance.csv",
            key_columns=("centre", "group"),
            value_columns=frozenset(
                {
                    "estimate",
                    "ci_lower",
                    "ci_upper",
                    "n_patients",
                    "n_events",
                    "reporting_status",
                    "reason",
                }
            ),
            fixed_filters={"endpoint": "OS", "model": "minimal_cox", "metric": "concordance"},
            metric_column="metric",
            status_column="reporting_status",
            count_columns=frozenset({"n_patients", "n_events"}),
            text_columns=frozenset({"reporting_status", "reason"}),
        ),
        TokenSource(
            name="subgroup",
            bundle="fairness_subgroups",
            artifact="sociodemographic_support.csv",
            key_columns=("variable", "level_key", "centre"),
            value_columns=frozenset({"n_patients", "n_events"}),
            count_columns=frozenset({"n_patients", "n_events"}),
        ),
        TokenSource(
            name="phase9_importance",
            bundle="phase9_forest",
            artifact="importance_summary.csv",
            key_columns=("endpoint", "centre", "model", "metric", "feature_group"),
            value_columns=frozenset({"importance", "reporting_status"}),
            metric_override="brier",
            status_column="reporting_status",
            text_columns=frozenset({"reporting_status"}),
        ),
    )
}


@dataclass(frozen=True)
class ResolvedToken:
    """One resolved token: its rendered string plus the exact cell it came from."""

    token: str
    source: str
    artifact: str
    value_column: str
    raw_value: Any
    rendered: str
    status: str | None
    source_cell: str

    @property
    def is_not_computable(self) -> bool:
        return self.status == REPORTING_NOT_COMPUTABLE

    @property
    def is_imprecise(self) -> bool:
        return self.status == REPORTING_IMPRECISE


def _coerce(series: pd.Series, wanted: str) -> pd.Series:
    """Match a token segment against a column without depending on its dtype spelling."""

    as_text = series.astype(str).str.strip()
    if as_text.eq(wanted).any():
        return as_text.eq(wanted)
    # numeric columns are written as `730` in prose but may be `730.0` in the frame
    try:
        target = float(wanted)
    except ValueError:
        return as_text.eq(wanted)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.eq(target)


def resolve_token(token: str, inputs: CanonicalInputs) -> ResolvedToken:
    """Resolve one token to exactly one full-precision artifact cell, or raise."""

    segments = [s for s in token.split(TOKEN_SEGMENT_SEPARATOR) if s != ""]
    if len(segments) < 2:
        raise TokenResolutionError(
            f"token {token!r} has too few segments to name a source and value"
        )

    source_name, *rest = segments
    source = TOKEN_SOURCES.get(source_name)
    if source is None:
        known = ", ".join(sorted(TOKEN_SOURCES))
        raise TokenResolutionError(
            f"token {token!r} names unknown source {source_name!r}; known: {known}"
        )

    value_column = rest[-1]
    supplied_keys = rest[:-1]
    if value_column not in source.value_columns:
        allowed = ", ".join(sorted(source.value_columns))
        raise TokenResolutionError(
            f"token {token!r} requests value {value_column!r}, which {source_name!r} does not "
            f"publish; allowed: {allowed}"
        )
    if len(supplied_keys) > len(source.key_columns):
        raise TokenResolutionError(
            f"token {token!r} supplies {len(supplied_keys)} keys but {source_name!r} has only "
            f"{len(source.key_columns)}: {', '.join(source.key_columns)}"
        )

    frame = inputs.frame(source.bundle, source.artifact)
    for column, wanted in source.fixed_filters.items():
        frame = frame[frame[column].astype(str).str.strip() == str(wanted)]

    applied: list[str] = []
    for column, wanted in zip(source.key_columns, supplied_keys, strict=False):
        frame = frame[_coerce(frame[column], wanted)]
        applied.append(f"{column}={wanted}")
        if frame.empty:
            raise TokenResolutionError(
                f"token {token!r} matched no row after applying {', '.join(applied)}; "
                f"an empty match is a build failure, not an empty string"
            )

    if len(frame) > 1:
        unsupplied = source.key_columns[len(supplied_keys) :]
        distinguishing = [
            column
            for column in unsupplied
            if column in frame.columns and frame[column].astype(str).nunique() > 1
        ]
        matched_on = ", ".join(applied) if applied else "the source alone"
        remedy = (
            f"name the remaining key(s) {', '.join(distinguishing)}"
            if distinguishing
            else "supply more keys"
        )
        raise TokenResolutionError(
            f"token {token!r} is ambiguous: {len(frame)} rows match {matched_on}; {remedy} -- "
            "the build never picks a row for you."
        )

    row = frame.iloc[0]
    raw = row[value_column]
    status = str(row[source.status_column]) if source.status_column else None

    if value_column in source.text_columns:
        rendered = "" if pd.isna(raw) else str(raw)
        kind = KIND_TEXT
    elif pd.isna(raw):
        raise TokenResolutionError(
            f"token {token!r} resolved to an empty cell "
            f"({', '.join(applied)}); write the status explicitly rather than emitting a blank"
        )
    elif value_column in source.count_columns:
        rendered = rounding_for(None, KIND_COUNT).render(float(raw))
        kind = KIND_COUNT
    else:
        metric = source.metric_override or (
            str(row[source.metric_column]) if source.metric_column else None
        )
        rendered = rounding_for(metric, KIND_METRIC).render(float(raw))
        kind = KIND_METRIC

    parts = [*applied, f"value={value_column}"] if applied else [f"value={value_column}"]
    if "source_ref" in row.index and not pd.isna(row["source_ref"]):
        parts.append(f"source_ref={row['source_ref']}")
    cell = "; ".join(parts)
    return ResolvedToken(
        token=token,
        source=source_name,
        artifact=f"{source.bundle}/{source.artifact}",
        value_column=value_column,
        raw_value=raw if kind == KIND_TEXT else float(raw),
        rendered=rendered,
        status=status,
        source_cell=cell,
    )


def status_rule_violations(paragraph: str, resolved: list[ResolvedToken]) -> list[str]:
    """Enforce the two prose rules the spec attaches to reporting status.

    A `not_computable` cell may not be quoted without its status appearing in the same paragraph,
    and an `estimable_imprecise` point estimate may not appear without its interval. Both rules
    exist because a status stripped from its number reads as a clean result.
    """

    lowered = paragraph.lower()
    violations: list[str] = []

    for token in resolved:
        if (
            token.is_not_computable
            and token.value_column not in {"reporting_status", "reason"}
            and not any(marker in lowered for marker in NOT_COMPUTABLE_PROSE_MARKERS)
        ):
            violations.append(
                f"{token.token} resolves to a not_computable cell but the paragraph does not "
                "carry that status"
            )

    imprecise_estimates = [
        t for t in resolved if t.is_imprecise and t.value_column in {"estimate", "gap", "value"}
    ]
    for token in imprecise_estimates:
        stem = token.token.rsplit(TOKEN_SEGMENT_SEPARATOR, 1)[0]
        has_interval = any(
            other.token.startswith(stem) and other.value_column in {"ci_lower", "gap_ci_lower"}
            for other in resolved
        )
        if not has_interval and not any(marker in lowered for marker in IMPRECISE_PROSE_MARKERS):
            violations.append(
                f"{token.token} is estimable_imprecise but the paragraph shows neither its "
                "interval nor its status"
            )
    return violations

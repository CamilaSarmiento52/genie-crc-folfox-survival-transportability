"""Manuscript-reporting public contracts: output paths, rounding rules, and shared errors.

Per `manuscript-spec.md`, manuscript reporting covers what already exists. It fits nothing, and it
produces no
new number beyond the two computed descriptive frames of spec section 4.3. Every constant here is a
frozen reporting decision, not a tunable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# manuscript-reporting workflow writes only under this root. Every earlier workflow's output root is
# read-only here.
MANUSCRIPT_OUTPUT_DIRNAME = "manuscript"

# The manuscript source files, in the journal's *submission* order. The *drafting* order is
# different and is frozen in manuscript-spec.md section 2; assembly order is not drafting order.
MANUSCRIPT_SECTION_FILES = (
    "00_title_page.md",
    "01_abstract.md",
    "02_keywords.md",
    "03_introduction.md",
    "04_methods.md",
    "05_results.md",
    "06_discussion.md",
    "07_conclusion.md",
    "08_figure_legends.md",
    "09_tables.md",
    "10_supplementary.md",
)

# Sections whose words count toward the journal's 3,000-6,000 body-word budget. The title page,
# abstract, keywords, legends, tables, and supplement are excluded by the counting rule in
# journal-requirements.md.
BODY_WORD_SECTIONS = (
    "03_introduction.md",
    "04_methods.md",
    "05_results.md",
    "06_discussion.md",
    "07_conclusion.md",
)

BODY_WORD_MINIMUM = 3000
BODY_WORD_CEILING = 6000
ABSTRACT_WORD_CEILING = 300
MAX_COMBINED_DISPLAYS = 8

# Frozen keyword set (manuscript-spec.md section 7; revised 2026-08-19 to complement rather than
# duplicate title terms, which indexers already cover).
KEYWORDS = (
    "colorectal neoplasms",
    "oxaliplatin",
    "overall survival",
    "clinical prediction model",
    "left truncation",
    "calibration",
)

# --- Rounding (manuscript-spec.md section 4.4) -------------------------------------------------
#
# Precision is fixed per metric class *before* drafting and applied once, at resolution, from the
# full-precision artifact cell -- so a quantity is never rounded two ways in two places, and a
# rounded value is never re-rounded downstream. An interval is always reported at the precision of
# its own point estimate, which is why precision is a property of the metric and not of the column.

DECIMALS_BY_METRIC: dict[str, int] = {
    # concordance / AUC / slope / Jaccard / tau -> 3
    "concordance": 3,
    "auc": 3,
    "slope": 3,
    "jaccard": 3,
    "tau": 3,
    "kendall_tau": 3,
    # `citl` is a calibration quantity on the same scale family as `slope` and is rounded with it.
    # The spec names the class "slope"; CITL is grouped here deliberately rather than defaulted,
    # because an unlisted metric raises instead of guessing.
    "citl": 3,
    # Brier / IBS -> 4
    "brier": 4,
    "ibs": 4,
    "scaled_brier": 4,
    # 2026-08-19: patient-level stability quantities. Bootstrap risk-interval widths and
    # cross-family prediction spreads are absolute-risk differences and follow the 3-decimal
    # discrimination/calibration class; proportions of patients are rendered as percentages
    # to the spec's one decimal.
    "risk_width": 3,
    "percentage": 1,
    # gains inherit the precision of the metric they are a difference of
    "auc_gain": 3,
    "slope_gain": 3,
    "citl_gain": 3,
    "ibs_gain": 4,
    "brier_gain": 4,
    "concordance_gain": 3,
}

PERCENTAGE_DECIMALS = 1
COUNT_DECIMALS = 0
DAY_DECIMALS = 0

# Value kinds a token may request that are not metric-scale quantities. These carry their own
# precision rule rather than a per-metric one.
KIND_PERCENTAGE = "percentage"
KIND_COUNT = "count"
KIND_DAYS = "days"
KIND_METRIC = "metric"
KIND_TEXT = "text"

# --- Reporting statuses (imported semantics, re-declared for display) --------------------------
#
# These three are displayed as three distinct categories and never collapsed into one bucket of bad
# results (the evaluation-diagnosis note, binding on this workflow).

REPORTING_ESTIMABLE = "estimable"
REPORTING_IMPRECISE = "estimable_imprecise"
REPORTING_NOT_COMPUTABLE = "not_computable"
REPORTING_STATUSES = (REPORTING_ESTIMABLE, REPORTING_IMPRECISE, REPORTING_NOT_COMPUTABLE)

# Human phrasings that satisfy the "prose must carry the status" rule for a `not_computable` cell.
# Matching is case-insensitive and is done on the paragraph containing the token.
NOT_COMPUTABLE_PROSE_MARKERS = (
    "not_computable",
    "not computable",
    "could not be computed",
    "not estimable",
)

IMPRECISE_PROSE_MARKERS = (
    "estimable_imprecise",
    "imprecise",
    "wide interval",
    "wide confidence interval",
)

# --- Token grammar ------------------------------------------------------------------------------
#
# A token is `{{source.key1.key2....value}}`: a registered source name, an ordered prefix of that
# source's key columns, and a final value column. Unsupplied trailing keys are permitted only when
# they leave exactly one matching row -- otherwise the build fails as ambiguous rather than picking
# a row. See `tokens.py`.

# `@` is admitted because explainability workflow publishes composite scope keys of the form
# `OS@VICC`; the token
# grammar matches key values literally, so the charset must cover the vocabularies actually
# published rather than force a rename of a frozen artifact's column values.
TOKEN_PATTERN = re.compile(r"\{\{([a-zA-Z0-9_.\-@]+)\}\}")
TOKEN_SEGMENT_SEPARATOR = "."

# A placeholder the build must report but may not invent a value for (title-page items such as the
# corresponding author's email). Distinct from a token: a placeholder has no artifact source.
PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class ReportContractError(ValueError):
    """Raised for a privacy-safe manuscript-reporting workflow contract violation."""


class TokenResolutionError(ReportContractError):
    """Raised when a token cannot be resolved to exactly one full-precision artifact cell."""


class BundleVerificationError(ReportContractError):
    """Raised when a canonical input bundle fails hash verification before any value is read."""


@dataclass(frozen=True)
class RoundingRule:
    """How one resolved value is rendered, decided before drafting and applied exactly once."""

    kind: str
    decimals: int

    def render(self, value: float) -> str:
        if self.decimals == 0:
            return f"{round(float(value)):d}"
        return f"{float(value):.{self.decimals}f}"


def rounding_for(metric: str | None, kind: str = KIND_METRIC) -> RoundingRule:
    """Return the frozen rounding rule for a metric, refusing to guess an unlisted one."""

    if kind == KIND_PERCENTAGE:
        return RoundingRule(KIND_PERCENTAGE, PERCENTAGE_DECIMALS)
    if kind == KIND_COUNT:
        return RoundingRule(KIND_COUNT, COUNT_DECIMALS)
    if kind == KIND_DAYS:
        return RoundingRule(KIND_DAYS, DAY_DECIMALS)
    if metric is None:
        raise ReportContractError(
            "a metric-scale value requires a metric name to select its frozen precision"
        )
    key = str(metric).strip().lower()
    if key not in DECIMALS_BY_METRIC:
        raise ReportContractError(
            f"no frozen rounding rule for metric {metric!r}; add it to DECIMALS_BY_METRIC as a "
            "dated decision rather than defaulting to an arbitrary precision"
        )
    return RoundingRule(KIND_METRIC, DECIMALS_BY_METRIC[key])

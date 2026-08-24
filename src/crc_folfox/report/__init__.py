"""Manuscript-reporting package: the manuscript build, its token resolver, and its computed frames.

Manuscript reporting covers what already exists. Nothing in this package fits, refits, recalibrates,
or tunes a
model, and nothing here produces a performance, transportability, explanation, or sensitivity number
that is not already published in a frozen data-cleaning-through-forest-retry artifacts artifact --
with exactly the two computed
descriptive exceptions defined in `manuscript-spec.md` section 4.3 (cohort flow and Table 1).
"""

from crc_folfox.report.contracts import (
    ReportContractError,
    TokenResolutionError,
    rounding_for,
)
from crc_folfox.report.inputs import (
    CanonicalInputs,
    verify_canonical_inputs,
)
from crc_folfox.report.tokens import resolve_token

__all__ = [
    "CanonicalInputs",
    "ReportContractError",
    "TokenResolutionError",
    "resolve_token",
    "rounding_for",
    "verify_canonical_inputs",
]

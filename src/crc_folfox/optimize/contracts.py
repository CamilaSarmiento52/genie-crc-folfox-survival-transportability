"""Optimization workflow public contracts: shared errors and small structural helpers."""

from __future__ import annotations

# optimization-spec.md, "Epistemic status": every optimization artifact row carries this label, in
# both workstreams -- the frozen model-training and primary-evaluation primary analysis carries
# none.
ANALYSIS_STATUS = "post_hoc_exploratory"


class OptimizationContractError(ValueError):
    """Raised for a privacy-safe correction or exploratory-optimization contract violation."""


# Backward-compatible alias for modules outside the explainability workflow that still use the
# historical specification-era name. New code should import `OptimizationContractError`.
Phase5AContractError = OptimizationContractError

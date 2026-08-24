"""Forest-model-retry workflow delayed-entry conditional-inference forest (`LTRCforests::ltrccif`).

Everything in this package is **post hoc exploratory**: the primary held-out results were
inspected on 2026-08-08, so nothing here can become a co-equal candidate or change the frozen
primary-evaluation workflow conclusion (`mission.md` non-negotiable 9). It publishes only under
`analysis/out/current/the forest-retry artifact directory`.
"""

from crc_folfox.forest.admission import (
    CANDIDATE,
    GateOutcome,
    decide_ltrccif_admission,
    run_admission_gates,
    synthetic_ltrc_fixture,
)
from crc_folfox.forest.r_adapter import (
    RISK_SCALES,
    RConditionalForestAdapter,
    RConditionalForestResult,
)

__all__ = [
    "CANDIDATE",
    "RISK_SCALES",
    "GateOutcome",
    "RConditionalForestAdapter",
    "RConditionalForestResult",
    "decide_ltrccif_admission",
    "run_admission_gates",
    "synthetic_ltrc_fixture",
]

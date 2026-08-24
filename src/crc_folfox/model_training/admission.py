"""Forest and boosting pre-result compatibility admission gates (Block C, Step 7).

Both decisions are nonblocking: a failed gate produces exactly one aggregate
`technically_ineligible` record and never blocks the four required Cox candidates, and never
authorizes an ordinary right-censored substitute. Neither function inspects restricted data or
outer-assessment predictions -- the forest gate runs entirely on a small synthetic fixture in
repeated clean R subprocesses.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.model_training.contracts import ModelTrainingContractError
from crc_folfox.model_training.r_adapter import RForestAdapter

FOREST_ADMISSION_REPEATS = 3


@dataclass(frozen=True)
class AdmissionRecord:
    """One aggregate gate-decision record for a conditional candidate model."""

    candidate: str
    status: str  # "admitted" or "technically_ineligible"
    reason: str
    evidence: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


def _synthetic_forest_fixture(
    seed: int = 20260807, n: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small, purely synthetic delayed-entry fixture -- no restricted data involved."""
    rng = np.random.default_rng(seed)
    matrix = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    entry = rng.uniform(0.0, 40.0, n)
    exit_ = entry + rng.uniform(10.0, 400.0, n)
    event = rng.integers(0, 2, n)
    metadata = pd.DataFrame({"entry": entry, "exit": exit_, "event": event})
    return matrix, metadata


def decide_forest_admission(
    *,
    script: str | Path | None = None,
    rscript: str = "Rscript",
    repeats: int = FOREST_ADMISSION_REPEATS,
    mtry: int = 2,
    nodesize: int = 10,
    seed: int = 1,
) -> AdmissionRecord:
    """Run the forest pre-result compatibility gate in repeated clean R processes.

    Each `RForestAdapter.fit()` call is its own `Rscript` subprocess, i.e. already a fresh clean
    process; `repeats` reruns it that many times **at the same seed** to prove the
    fit/predict/reload cycle is reproducible rather than a one-off success or a nondeterministic
    crash.
    """
    adapter = RForestAdapter(script=script, rscript=rscript)
    try:
        runtime = adapter.preflight()
    except ModelTrainingContractError as exc:
        return AdmissionRecord(
            candidate="ltrc_rrf",
            status="technically_ineligible",
            reason=f"preflight_failed: {exc}",
        )

    matrix, metadata = _synthetic_forest_fixture(seed=seed)
    survivals: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="crc-phase4-forest-admission-") as directory:
        for attempt in range(repeats):
            destination = Path(directory) / f"attempt-{attempt}.rds"
            try:
                # The seed is held FIXED across repeats. Each attempt is its own clean `Rscript`
                # process, so identical predictions prove cross-process reproducibility at a fixed
                # seed -- which is what the check below asserts and what the record claims.
                # Corrected 2026-08-17: this previously passed `seed + attempt`, varying the seed
                # and then demanding identical predictions, so any correctly-behaving stochastic
                # forest was rejected as `nondeterministic`. Measured on an `ltrccif` control:
                # same seed max|diff| 0.0, seeds 1 vs 2 max|diff| 0.0129. No recorded result
                # changes -- `ltrc_rrf` fails at the first fit, before this check is reached.
                result = adapter.fit(
                    matrix,
                    metadata,
                    mtry=mtry,
                    nodesize=nodesize,
                    seed=seed,
                    destination=destination,
                )
            except ModelTrainingContractError as exc:
                return AdmissionRecord(
                    candidate="ltrc_rrf",
                    status="technically_ineligible",
                    reason=(
                        f"fit_failed_on_attempt_{attempt + 1}_of_{repeats}: {exc}. Installation "
                        "succeeded (preflight passed) but the fit/predict cycle did not complete "
                        "cleanly in a fresh R process; see implement.md for the captured evidence."
                    ),
                    evidence=dict(runtime),
                )
            survivals.append(result.survival)

    for survival in survivals[1:]:
        if not np.allclose(survival, survivals[0]):
            return AdmissionRecord(
                candidate="ltrc_rrf",
                status="technically_ineligible",
                reason=(
                    "nondeterministic_synthetic_prediction: repeated clean-process fits with the "
                    "same seed produced different synthetic survival predictions"
                ),
                evidence=dict(runtime),
            )

    return AdmissionRecord(
        candidate="ltrc_rrf",
        status="admitted",
        reason=(
            f"passed preflight and {repeats} repeated clean-process fit/predict/reload cycles "
            "with reproducible synthetic predictions"
        ),
        evidence=dict(runtime),
    )


def decide_boosting_admission() -> AdmissionRecord:
    """No maintained delayed-entry-aware boosting implementation is named in this repository.

    Per `model-training-spec.md` ("Conditional candidate 6") and `tech-stack.md`, admission
    requires a dated pre-result amendment naming a maintained estimator, an explicit entry-aware
    objective, an absolute-survival mapping, a bounded tuning grid, pinned versions, and a
    serialization format -- before restricted-data fitting or held-out results are inspected.
    Ordinary scikit-survival gradient boosting and standard XGBoost Cox/AFT survival objectives
    are explicitly ineligible because their documented outcome interfaces do not encode sample
    entry/risk-set truncation. No such amendment exists, so this is a deterministic
    `technically_ineligible` record, not a computation.
    """
    return AdmissionRecord(
        candidate="ltrc_boosting",
        status="technically_ineligible",
        reason="no_passing_pre_result_admission_gate: no dated amendment names a maintained "
        "delayed-entry-aware boosting implementation",
    )


__all__ = [
    "FOREST_ADMISSION_REPEATS",
    "AdmissionRecord",
    "decide_boosting_admission",
    "decide_forest_admission",
]

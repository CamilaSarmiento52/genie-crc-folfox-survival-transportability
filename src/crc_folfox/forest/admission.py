"""Forest-model-retry workflow admission gate for the delayed-entry conditional-inference forest
(`ltrccif`).

The seven gates are imported verbatim from `optimization-spec.md` (optimization workflow,
"Conditional B7")
and may not be weakened, reordered, or waived. A partial pass is a failure.

Like model-training workflow's gate, **this never raises**: every failure mode returns an
`AdmissionRecord` with
`technically_ineligible` and the failing gate, so a gate failure is a return value rather than an
exception and cannot block anything downstream.

Nothing here touches restricted data. Every gate runs on a small synthetic delayed-entry fixture
in clean `Rscript` subprocesses.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from crc_folfox.forest.r_adapter import RISK_SCALES, RConditionalForestAdapter
from crc_folfox.model_training.admission import AdmissionRecord
from crc_folfox.model_training.contracts import ModelTrainingContractError

CANDIDATE = "ltrc_cif"

#: Repeated clean-process fits proving gate 4 and gate 7.
CIF_ADMISSION_REPEATS = 3

#: The frozen primary-evaluation workflow IBS grid resolution, used for the monotonicity/boundedness
#: gate.
GATE_TIME_GRID: tuple[float, ...] = tuple(float(t) for t in range(365, 731, 30))

#: Tolerance for "identical" predictions across reload and across processes.
EXACT_TOLERANCE = 1e-12

#: Tolerance for the hand-calculated Kaplan-Meier fixture. The comparison is cross-language
#: (R partykit against Python lifelines), so this is a numerical-agreement bound, not a fudge.
FIXTURE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class GateOutcome:
    """One gate's result, recorded whether it passed or failed."""

    gate: int
    name: str
    passed: bool
    detail: str = ""
    measurements: dict[str, float] = field(default_factory=dict)


def synthetic_ltrc_fixture(seed: int = 20260815, n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small, purely synthetic delayed-entry fixture -- no restricted data involved."""
    rng = np.random.default_rng(seed)
    matrix = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    entry = rng.uniform(0.0, 50.0, n)
    exit_ = entry + rng.exponential(250.0, n) + 5.0
    event = rng.integers(0, 2, n)
    metadata = pd.DataFrame({"entry": entry, "exit": exit_, "event": event})
    return matrix, metadata


def _delayed_entry_km(metadata: pd.DataFrame, times: tuple[float, ...]) -> np.ndarray:
    """Independent Python delayed-entry Kaplan-Meier, for the gate-3 hand-calculated fixture."""
    fitter = KaplanMeierFitter().fit(
        metadata["exit"].to_numpy(float),
        event_observed=metadata["event"].to_numpy(int),
        entry=metadata["entry"].to_numpy(float),
    )
    return np.asarray(
        [float(fitter.predict(t)) for t in times],
        dtype=float,
    )


def run_admission_gates(
    *,
    script: str | Path | None = None,
    rscript: str = "Rscript",
    repeats: int = CIF_ADMISSION_REPEATS,
    mtry: int = 2,
    minbucket: int = 7,
    seed: int = 1,
) -> tuple[AdmissionRecord, list[GateOutcome]]:
    """Run all seven admission gates. Returns the record and every gate's outcome.

    Every gate outcome is recorded, including those after a failure is already certain, so the
    evidence trail shows what was checked rather than only where it stopped.
    """
    outcomes: list[GateOutcome] = []
    adapter = RConditionalForestAdapter(script=script, rscript=rscript)

    def record(status: str, reason: str, runtime: dict[str, str] | None = None) -> AdmissionRecord:
        evidence = dict(runtime or {})
        evidence["gates_passed"] = f"{sum(o.passed for o in outcomes)}/{len(outcomes)}"
        failed = [str(o.gate) for o in outcomes if not o.passed]
        if failed:
            evidence["failed_gates"] = ",".join(failed)
        return AdmissionRecord(candidate=CANDIDATE, status=status, reason=reason, evidence=evidence)

    try:
        runtime = adapter.preflight()
    except ModelTrainingContractError as exc:
        outcomes.append(GateOutcome(0, "preflight", False, str(exc)))
        return record("technically_ineligible", f"preflight_failed: {exc}"), outcomes

    matrix, metadata = synthetic_ltrc_fixture()

    with tempfile.TemporaryDirectory(prefix="crc-phase9-admission-") as directory:
        root = Path(directory)

        # -- Gates 4 and 7: repeated clean-process fits, and cross-process reproducibility -----
        fits = []
        for attempt in range(repeats):
            try:
                fits.append(
                    adapter.fit(
                        matrix,
                        metadata,
                        mtry=mtry,
                        minbucket=minbucket,
                        seed=seed,
                        destination=root / f"attempt-{attempt}.rds",
                    )
                )
            except ModelTrainingContractError as exc:
                outcomes.append(GateOutcome(4, "repeated_fits_without_crash", False, str(exc)))
                return (
                    record(
                        "technically_ineligible",
                        f"fit_failed_on_attempt_{attempt + 1}_of_{repeats}: {exc}",
                        runtime,
                    ),
                    outcomes,
                )
        outcomes.append(
            GateOutcome(4, "repeated_fits_without_crash", True, f"{repeats}/{repeats} fits")
        )

        # Same seed, separate Rscript processes -> identical predictions on both scales.
        cross_process = (
            max(
                float(np.max(np.abs(fits[i].survival(scale) - fits[0].survival(scale))))
                for i in range(1, len(fits))
                for scale in RISK_SCALES
            )
            if len(fits) > 1
            else 0.0
        )
        outcomes.append(
            GateOutcome(
                7,
                "cross_process_reproducibility",
                cross_process <= EXACT_TOLERANCE,
                f"max abs difference {cross_process:.3e} across {repeats} clean processes",
                {"max_abs_difference": cross_process},
            )
        )

        bundle = fits[0].bundle

        # -- Gate 1/5: finite, bounded, monotone survival across the frozen IBS grid ----------
        # (`_as_survival_array` already enforces this and would have raised; re-measure so the
        # evidence trail carries a number rather than an assumption.)
        try:
            predictions = adapter.predict(bundle, matrix, metadata, times=GATE_TIME_GRID)
        except ModelTrainingContractError as exc:
            outcomes.append(GateOutcome(5, "finite_bounded_monotone", False, str(exc)))
            return (
                record("technically_ineligible", f"prediction_failed: {exc}", runtime),
                outcomes,
            )

        worst_increase = max(
            float(np.max(np.diff(predictions[scale], axis=1))) for scale in RISK_SCALES
        )
        bounded = all(
            bool(np.isfinite(predictions[s]).all())
            and float(predictions[s].min()) >= 0.0
            and float(predictions[s].max()) <= 1.0
            for s in RISK_SCALES
        )
        outcomes.append(
            GateOutcome(
                5,
                "finite_bounded_monotone",
                bounded and worst_increase <= 1e-9,
                f"bounded={bounded}, largest survival increase {worst_increase:.3e}",
                {"largest_increase": worst_increase},
            )
        )
        outcomes.append(
            GateOutcome(
                1,
                "accepts_and_uses_entry_exit_event",
                True,
                "counting-process formula; gate 2 establishes use rather than mere acceptance",
            )
        )

        # -- Gate 2: entry-only perturbation must change predictions --------------------------
        shifted = metadata.copy()
        shifted["entry"] = np.minimum(shifted["entry"] + 40.0, shifted["exit"] - 1.0)
        try:
            perturbed = adapter.predict(bundle, matrix, shifted, times=GATE_TIME_GRID)
        except ModelTrainingContractError as exc:
            outcomes.append(
                GateOutcome(2, "entry_perturbation_changes_predictions", False, str(exc))
            )
            return (
                record("technically_ineligible", f"entry_perturbation_failed: {exc}", runtime),
                outcomes,
            )
        entry_effect = float(
            np.max(np.abs(perturbed["entry_conditional"] - predictions["entry_conditional"]))
        )
        outcomes.append(
            GateOutcome(
                2,
                "entry_perturbation_changes_predictions",
                entry_effect > 1e-6,
                f"max abs survival change {entry_effect:.6f} when only entry moves",
                {"entry_effect": entry_effect},
            )
        )

        # The unconditional scale predicts at entry = 0 by construction, so it must be
        # *invariant* to the same perturbation. This is what proves the two scales are genuinely
        # different quantities rather than a relabelling.
        unconditional_drift = float(
            np.max(np.abs(perturbed["unconditional"] - predictions["unconditional"]))
        )
        outcomes.append(
            GateOutcome(
                2,
                "unconditional_scale_invariant_to_entry",
                unconditional_drift <= EXACT_TOLERANCE,
                f"unconditional scale moved {unconditional_drift:.3e} (must be 0)",
                {"unconditional_drift": unconditional_drift},
            )
        )

        # -- Gate 6: serialize/reload identical -----------------------------------------------
        reloaded = adapter.predict(bundle, matrix, metadata, times=GATE_TIME_GRID)
        reload_delta = max(float(np.max(np.abs(reloaded[s] - predictions[s]))) for s in RISK_SCALES)
        outcomes.append(
            GateOutcome(
                6,
                "serialize_reload_identical",
                reload_delta <= EXACT_TOLERANCE,
                f"max abs difference {reload_delta:.3e} from a reloaded bundle",
                {"reload_delta": reload_delta},
            )
        )

        # -- Gate 3: hand-calculated delayed-entry fixture -------------------------------------
        # A deliberately unsplittable forest over the full sample must reduce, on the
        # unconditional scale, to the delayed-entry Kaplan-Meier of its training rows --
        # computed here by Python lifelines, an implementation independent of R partykit.
        try:
            degenerate = adapter._fit_degenerate(
                matrix,
                metadata,
                mtry=mtry,
                seed=seed,
                destination=root / "degenerate.rds",
            )
            degenerate_curve = adapter.predict(
                degenerate.bundle, matrix.iloc[:1], metadata.iloc[:1], times=GATE_TIME_GRID
            )["unconditional"][0]
        except ModelTrainingContractError as exc:
            outcomes.append(GateOutcome(3, "hand_calculated_fixture", False, str(exc)))
            return (
                record("technically_ineligible", f"fixture_check_failed: {exc}", runtime),
                outcomes,
            )

        km = _delayed_entry_km(metadata, GATE_TIME_GRID)
        fixture_delta = float(np.max(np.abs(degenerate_curve - km)))
        outcomes.append(
            GateOutcome(
                3,
                "hand_calculated_fixture",
                fixture_delta <= FIXTURE_TOLERANCE,
                (
                    f"single-node forest vs independent lifelines delayed-entry KM: "
                    f"max abs difference {fixture_delta:.3e}"
                ),
                {"fixture_delta": fixture_delta},
            )
        )

    failures = [o for o in outcomes if not o.passed]
    if failures:
        first = failures[0]
        return (
            record(
                "technically_ineligible",
                f"gate_{first.gate}_failed ({first.name}): {first.detail}",
                runtime,
            ),
            outcomes,
        )
    return (
        record(
            "admitted",
            f"passed all seven admission gates on {repeats} repeated clean-process fits",
            runtime,
        ),
        outcomes,
    )


def decide_ltrccif_admission(**kwargs: object) -> AdmissionRecord:
    """The gate as a single record. Never raises."""
    try:
        record, _ = run_admission_gates(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 -- a gate failure must never propagate
        return AdmissionRecord(
            candidate=CANDIDATE,
            status="technically_ineligible",
            reason=f"unexpected_gate_error: {type(exc).__name__}: {exc}",
        )
    return record


__all__ = [
    "CANDIDATE",
    "CIF_ADMISSION_REPEATS",
    "EXACT_TOLERANCE",
    "FIXTURE_TOLERANCE",
    "GATE_TIME_GRID",
    "GateOutcome",
    "decide_ltrccif_admission",
    "run_admission_gates",
    "synthetic_ltrc_fixture",
]

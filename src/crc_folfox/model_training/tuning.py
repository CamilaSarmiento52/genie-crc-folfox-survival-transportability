"""Delayed-entry IPCW Brier tuning objective (training-only model-selection diagnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from crc_folfox.model_training.contracts import HORIZONS, ModelTrainingContractError


def delayed_entry_ipcw_brier(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    survival: np.ndarray,
    *,
    horizons: tuple[float, ...] = HORIZONS,
) -> float:
    """Score inner-validation predictions with the delayed-entry IPCW Brier objective.

    The censoring distribution is fitted on `training` rows only (with `entry` as the left-
    truncation bound). Each horizon's score targets survival among validation rows eligible by
    `entry <= horizon`; a row censored before the horizon contributes a zero-weighted term but
    still counts in that horizon's denominator (the standard Graf et al. IPCW convention). The two
    horizon scores are averaged with equal weight.
    """
    km = KaplanMeierFitter().fit(
        training["exit"], event_observed=1 - training["event"].astype(int), entry=training["entry"]
    )
    scores: list[float] = []
    for index, horizon in enumerate(horizons):
        eligible = validation["entry"].to_numpy(float) <= horizon
        if not eligible.any():
            raise ModelTrainingContractError("No validation rows are eligible at a tuning horizon")
        exit_ = validation["exit"].to_numpy(float)
        event = validation["event"].to_numpy(int)
        predicted = survival[:, index]
        weights = np.zeros(len(validation), dtype=float)
        occurred = (exit_ <= horizon) & (event == 1) & eligible
        survived = (exit_ > horizon) & eligible
        for mask, time in ((occurred, exit_), (survived, np.full(len(validation), horizon))):
            for row in np.flatnonzero(mask):
                # The left limit is used for observed events, as required by IPCW.
                point = np.nextafter(time[row], -np.inf) if mask is occurred else time[row]
                probability = float(km.predict(point))
                if not np.isfinite(probability) or probability <= 0:
                    raise ModelTrainingContractError(
                        "Censoring support is zero at a tuning horizon"
                    )
                weights[row] = 1.0 / probability
        target = survived.astype(float)
        score = float(np.sum(weights * (target - predicted) ** 2) / int(eligible.sum()))
        if not np.isfinite(score):
            raise ModelTrainingContractError("IPCW Brier score is nonfinite")
        scores.append(score)
    return float(np.mean(scores))

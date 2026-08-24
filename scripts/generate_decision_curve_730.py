"""Generate held-out 730-day OS decision curves for the manuscript.

The analysis reuses the frozen leave-one-centre-out minimal-Cox bundles.  The
clinical comparator is a delayed-entry Cox model containing metastatic
presentation alone, fitted on the same two development centres for each outer
iteration.  Net benefit is adjusted for delayed entry and right censoring with
the assessment-centre reverse Kaplan-Meier weights already used by the main
evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from crc_folfox.evaluate.censoring import fit_censoring_km, horizon_weights
from crc_folfox.evaluate.prediction import predict_for_frame
from crc_folfox.model_training.bundles import (
    RegistryRecord,
    verify_registry_load,
)
from crc_folfox.model_training.contracts import ModelScope
from crc_folfox.preprocessing.contracts import build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits


HORIZON = 730.0
THRESHOLDS = np.round(np.arange(0.20, 0.501, 0.01), 2)
CENTRES = ("DFCI", "MSK", "VICC")


def _presentation_design(frame: pd.DataFrame) -> pd.DataFrame:
    levels = set(frame["metastatic_presentation"].astype(str))
    expected = {"De novo Stage IV", "Recurrent/metachronous"}
    if not levels <= expected:
        raise ValueError(f"Unexpected metastatic-presentation level(s): {levels - expected}")
    design = frame[["entry", "exit", "event"]].copy()
    design["de_novo_stage_iv"] = (
        frame["metastatic_presentation"].astype(str) == "De novo Stage IV"
    ).astype(int)
    return design


def _net_benefit(
    risk: np.ndarray,
    threshold: float,
    *,
    weights,
    n_eligible: int,
) -> float:
    positive = np.asarray(risk, dtype=float) >= threshold
    true_positive = np.sum(weights.weight * weights.occurred * positive) / n_eligible
    false_positive = np.sum(weights.weight * weights.survived * positive) / n_eligible
    return float(true_positive - false_positive * threshold / (1.0 - threshold))


def generate(workspace: Path) -> tuple[Path, Path]:
    output_dir = workspace / "analysis" / "out" / "current" / "manuscript"
    analysis_root = workspace / "analysis" / "out" / "current"
    data = pd.read_csv(analysis_root / "analysis_dataset.csv")
    view = build_endpoint_view(data, "OS")
    registry = json.loads(
        (analysis_root / "model_training" / "model_registry.json").read_text()
    )
    models_dir = analysis_root / "model_training" / "models"

    records: list[dict[str, object]] = []
    for split in build_iecv_splits(view):
        development, assessment = split.frames(view)
        model_id = ModelScope(split).model_id("minimal_cox")
        state = verify_registry_load(
            models_dir,
            RegistryRecord(**registry[model_id]),
        )
        prediction = predict_for_frame(state, assessment, (HORIZON,))
        model_risk = prediction.unconditional[:, 0]

        comparator = CoxPHFitter(penalizer=0.0)
        comparator.fit(
            _presentation_design(development),
            duration_col="exit",
            event_col="event",
            entry_col="entry",
            show_progress=False,
        )
        comparator_risk = 1.0 - comparator.predict_survival_function(
            _presentation_design(assessment)[["de_novo_stage_iv"]],
            times=[HORIZON],
        ).to_numpy(dtype=float).ravel()

        censoring = fit_censoring_km(
            prediction.entry,
            prediction.exit,
            prediction.event,
        )
        weights = horizon_weights(
            censoring,
            prediction.entry,
            prediction.exit,
            prediction.event,
            HORIZON,
        )
        n_eligible = int(weights.eligible.sum())
        n_events = int(weights.occurred.sum())
        strategies = {
            "Four-variable Cox": model_risk,
            "Metastatic presentation only": comparator_risk,
            "Treat all": np.ones_like(model_risk),
            "Treat none": np.zeros_like(model_risk),
        }
        for threshold in THRESHOLDS:
            for strategy, risk in strategies.items():
                records.append(
                    {
                        "endpoint": "OS",
                        "horizon_days": int(HORIZON),
                        "held_out_centre": split.held_out_institution,
                        "threshold_probability": threshold,
                        "strategy": strategy,
                        "net_benefit": _net_benefit(
                            risk,
                            threshold,
                            weights=weights,
                            n_eligible=n_eligible,
                        ),
                        "held_out_patients": len(assessment),
                        "eligible_at_horizon": n_eligible,
                        "observed_deaths_by_horizon": n_events,
                    }
                )

    result = pd.DataFrame.from_records(records)
    csv_path = output_dir / "decision_curve_os_730.csv"
    result.to_csv(csv_path, index=False)

    colours = {
        "Four-variable Cox": "#1f4e79",
        "Metastatic presentation only": "#d97706",
        "Treat all": "#777777",
        "Treat none": "#111111",
    }
    styles = {
        "Four-variable Cox": "-",
        "Metastatic presentation only": "-",
        "Treat all": "--",
        "Treat none": ":",
    }
    widths = {
        "Four-variable Cox": 2.5,
        "Metastatic presentation only": 2.5,
        "Treat all": 1.7,
        "Treat none": 1.7,
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharex=True, sharey=True)
    for axis, centre in zip(axes, CENTRES, strict=True):
        panel = result[result["held_out_centre"].eq(centre)]
        for strategy in (
            "Four-variable Cox",
            "Metastatic presentation only",
            "Treat all",
            "Treat none",
        ):
            curve = panel[panel["strategy"].eq(strategy)]
            axis.plot(
                100 * curve["threshold_probability"],
                100 * curve["net_benefit"],
                color=colours[strategy],
                linestyle=styles[strategy],
                linewidth=widths[strategy],
                label=strategy,
            )
        n_eligible = int(panel["eligible_at_horizon"].iloc[0])
        n_events = int(panel["observed_deaths_by_horizon"].iloc[0])
        axis.set_title(f"{centre}\n{n_eligible} eligible; {n_events} deaths", fontsize=12)
        axis.set_xlim(20, 50)
        axis.set_ylim(-46, 31)
        axis.set_xticks([20, 30, 40, 50])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Threshold probability (%)", fontsize=11)
        axis.tick_params(labelsize=10)
    axes[0].set_ylabel("Net benefit per 100 patients", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Held-out decision-curve analysis for 730-day mortality",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    figure_path = output_dir / "figures" / "figure5_decision_curve_os_730.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return csv_path, figure_path


if __name__ == "__main__":
    csv_output, figure_output = generate(Path(__file__).resolve().parents[1])
    print(csv_output)
    print(figure_output)

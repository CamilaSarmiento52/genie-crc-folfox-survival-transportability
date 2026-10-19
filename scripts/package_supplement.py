"""Assemble the supplementary tables S34 to S40 from existing aggregate artifacts.

Read-only with respect to every earlier output. Nothing is fitted or re-estimated; the only new
computation is the percentile summary of already-frozen held-out predictions (S40).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.prediction import transform_for_bundle, unconditional_risk
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.preprocessing.contracts import build_endpoint_view

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "analysis/out/current"
SUPP = CURRENT / "supplementary"
DATA = SUPP / "supplementary_data"
MARGINS = {"auc": 0.02, "ibs": 0.005, "citl": 0.05, "slope": 0.05}
CENTRES = ("DFCI", "MSK", "VICC")


def out(item: str) -> Path:
    path = DATA / item
    path.mkdir(parents=True, exist_ok=True)
    return path


def s34() -> None:
    target = out("S34")
    for path in sorted((SUPP / "ph_assessment").iterdir()):
        shutil.copy2(path, target / path.name)


def s36() -> pd.DataFrame:
    checks = pd.DataFrame(json.loads((SUPP / "ph_assessment/reproduction_checks.json").read_text()))
    dev = checks[checks["model"] == "minimal_cox"].copy()
    dev["coefficients"] = dev["design_columns"]
    dev["events_per_coefficient"] = (dev["development_events"] / dev["coefficients"]).round(1)
    dev = dev[
        [
            "endpoint",
            "held_out_institution",
            "development_patients",
            "development_events",
            "coefficients",
            "events_per_coefficient",
        ]
    ]
    dev.to_csv(out("S36") / "development_support.csv", index=False)
    perf = pd.read_csv(CURRENT / "evaluation/performance_summary.csv")
    cell = perf[(perf["design"] == "iecv") & (perf["model"] == "minimal_cox")]
    rows = []
    for (endpoint, centre), block in cell.groupby(["endpoint", "centre"]):
        overall = block[block["metric"] == "concordance"].iloc[0]
        row = {
            "endpoint": endpoint,
            "held_out_institution": centre,
            "patients": int(overall["n_eligible"]),
            "events": int(overall["n_events"]),
        }
        for horizon in (365.0, 730.0):
            auc = block[(block["metric"] == "auc") & (block["horizon"] == horizon)].iloc[0]
            row[f"eligible_{int(horizon)}"] = int(auc["n_eligible"])
            row[f"events_{int(horizon)}"] = int(auc["n_events"])
            row[f"event_free_{int(horizon)}"] = int(auc["n_event_free"])
        row["min_finite_bootstrap_fraction"] = float(block["bootstrap_finite_fraction"].min())
        row["cells_estimable"] = int((block["reporting_status"] == "estimable").sum())
        row["cells_imprecise"] = int((block["reporting_status"] == "estimable_imprecise").sum())
        row["cells_not_computable"] = int((block["reporting_status"] == "not_computable").sum())
        rows.append(row)
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(out("S36") / "evaluation_support.csv", index=False)
    return evaluation


def s37() -> None:
    target = out("S37")
    enet = pd.read_csv(CURRENT / "phase5a/correction/model_training/tuning_summary.csv")
    enet = enet[enet["model_id"].str.contains("design=iecv_outer")].copy()
    enet["endpoint"] = enet["model_id"].str.extract(r"endpoint=([^|]+)")
    enet["held_out_institution"] = enet["model_id"].str.extract(r"holdout=([^|]+)")
    enet[
        [
            "endpoint",
            "held_out_institution",
            "selected_penalizer",
            "selected_l1_ratio",
            "required_candidate_count",
            "eligible_candidate_count",
            "required_fold_count",
            "tie_break_used",
        ]
    ].to_csv(target / "elastic_net_selected.csv", index=False)
    forest = pd.read_csv(CURRENT / "phase9/fit_summary.csv")
    forest = forest[forest["design"] == "iecv_outer"]
    forest[
        [
            "endpoint",
            "held_out_institution",
            "development_patients",
            "development_events",
            "ntree",
            "selected_mtry",
            "selected_minbucket",
            "selected_score",
        ]
    ].to_csv(target / "forest_selected.csv", index=False)
    grid = pd.read_csv(CURRENT / "phase9/tuning_summary.csv")
    grid = grid[grid["design"] == "iecv_outer"]
    grid[["held_out_institution", "mtry", "minbucket", "score", "selected"]].to_csv(
        target / "forest_grid_scores.csv", index=False
    )
    extra = pd.read_csv(CURRENT / "phase5a/optimization/fit_summary.csv")
    extra = extra[extra["design"] == "iecv_outer"]
    extra[
        ["endpoint", "held_out_institution", "candidate", "feature_set", "fit_status", "config"]
    ].to_csv(target / "post_hoc_penalized_selected.csv", index=False)


def s38() -> pd.DataFrame:
    frames = []
    for label, path in [
        ("prespecified", "evaluation/complexity_value_summary.csv"),
        ("post hoc penalized", "phase5a/optimization/complexity_value_summary.csv"),
        ("post hoc forest", "phase9/complexity_value_summary.csv"),
    ]:
        frame = pd.read_csv(CURRENT / path)
        frame["comparison_set"] = label
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["materiality_margin"] = table["gain"].map(MARGINS)
    computable = table["ci_lower"].notna() & table["ci_upper"].notna()
    table["direction"] = np.where(
        ~computable,
        "not computable",
        np.where(table["value"] > 0, "favours fuller", "favours minimal"),
    )
    table["interval_excludes_zero"] = computable & (
        (table["ci_lower"] > 0) | (table["ci_upper"] < 0)
    )
    table["material_benefit_excluded"] = computable & (
        table["ci_upper"] < table["materiality_margin"]
    )
    keep = [
        "comparison_set",
        "endpoint",
        "design",
        "centre",
        "fuller_model",
        "gain",
        "horizon",
        "minimal_estimate",
        "fuller_estimate",
        "value",
        "ci_lower",
        "ci_upper",
        "materiality_margin",
        "direction",
        "interval_excludes_zero",
        "material_benefit_excluded",
        "reason",
        "classification",
    ]
    table = table[keep].rename(columns={"value": "paired_gain"})
    table.to_csv(out("S38") / "paired_complexity_comparisons.csv", index=False)
    return table


def s39() -> None:
    importance = pd.read_csv(CURRENT / "explainability/importance_summary.csv")
    block = importance[importance["model"] == "minimal_cox"]
    block[
        ["endpoint", "centre", "feature_group", "metric", "importance", "reporting_status"]
    ].to_csv(out("S39") / "minimal_cox_importance_values.csv", index=False)


def s40() -> None:
    registry = json.loads((CURRENT / "model_training/model_registry.json").read_text())
    data = pd.read_csv(CURRENT / "analysis_dataset.csv")
    view = build_endpoint_view(data, "OS").data
    rows = []
    for centre in CENTRES:
        key = f"endpoint=OS|design=iecv_outer|holdout={centre}|model=minimal_cox"
        state = verify_registry_load(
            CURRENT / "model_training/models", RegistryRecord(**registry[key])
        )
        held = view.loc[view["institution"].eq(centre)].copy()
        matrix, _ = transform_for_bundle(state, held)
        risk = unconditional_risk(state, matrix)
        for column, horizon in enumerate((365, 730)):
            quantiles = np.percentile(risk[:, column], [5, 25, 50, 75, 95])
            rows.append(
                {
                    "held_out_institution": centre,
                    "horizon_days": horizon,
                    "patients": len(held),
                    **{
                        f"p{p}": round(float(q), 3)
                        for p, q in zip((5, 25, 50, 75, 95), quantiles, strict=True)
                    },
                    **{
                        f"share_at_or_above_{int(100 * t)}pct": round(
                            float((risk[:, column] >= t).mean()), 3
                        )
                        for t in (0.2, 0.3, 0.5)
                    },
                }
            )
    pd.DataFrame(rows).to_csv(out("S40") / "predicted_risk_distribution.csv", index=False)


def main() -> None:
    s34()
    evaluation = s36()
    s37()
    table = s38()
    s39()
    s40()
    os_rows = table[(table["endpoint"] == "OS") & (table["comparison_set"] == "prespecified")]
    computable = os_rows[os_rows["direction"] != "not computable"]
    summary = {
        "os_prespecified_computable_gains": int(len(computable)),
        "os_prespecified_intervals_containing_zero": int(
            (~computable["interval_excludes_zero"]).sum()
        ),
        "os_prespecified_material_benefit_excluded": int(
            computable["material_benefit_excluded"].sum()
        ),
        "min_finite_bootstrap_fraction_minimal_iecv": float(
            evaluation["min_finite_bootstrap_fraction"].min()
        ),
    }
    (SUPP / "supplement_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

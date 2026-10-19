"""Run the proportional-hazards assessment: engine admission, then the frozen real models.

Read-only with respect to every earlier output. Publishes aggregates only, under
`analysis/out/current/supplementary/ph_assessment/` (Supplementary Table S34).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from crc_folfox.evaluate.prediction import transform_for_bundle
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.preprocessing.contracts import build_endpoint_view
from crc_folfox.revision import proportional_hazards as ph

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "analysis/out/current"
SUPP = CURRENT / "supplementary"
OUT = SUPP / "ph_assessment"
ADMISSION_SEEDS = 40
MODELS = (("OS", "minimal_cox"), ("TTNT-D", "minimal_cox"), ("OS", "spline_cox"))
TERM_LABELS = {
    "age_at_folfox": "Age at FOLFOX",
    "metastatic_presentation": "Metastatic presentation",
    "ca_d_site": "Primary site",
    "number_metastatic_sites": "Number of metastatic sites",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {
        str(p.relative_to(CURRENT)): sha(p)
        for p in sorted(CURRENT.rglob("*"))
        if p.is_file() and SUPP not in p.parents and p.name != ".DS_Store"
    }


def admission() -> dict[str, object]:
    def fit(matrix, metadata):
        frame = pd.concat([metadata, matrix], axis=1)
        return CoxPHFitter().fit(frame, duration_col="exit", event_col="event", entry_col="entry")

    matrix, metadata = ph.simulate_left_truncated(600, seed=101)
    fitter = fit(matrix, metadata)
    try:
        fitter.compute_residuals(pd.concat([metadata, matrix], axis=1), "schoenfeld")
        lifelines_status = "returned residuals"
    except NotImplementedError as error:
        lifelines_status = f"NotImplementedError: {error}"
    response = ph.run_engine(matrix, metadata, fitter.params_.to_dict(), return_residuals=True)
    x, beta = matrix.to_numpy(), fitter.params_.to_numpy()
    _, reference = ph.schoenfeld_reference(
        x, metadata["entry"], metadata["exit"], metadata["event"], beta
    )
    _, ignored = ph.schoenfeld_reference(
        x, np.zeros(len(metadata)), metadata["exit"], metadata["event"], beta
    )
    planted = proportional = 0
    for seed in range(ADMISSION_SEEDS):
        m, md = ph.simulate_left_truncated(
            900, seed=1000 + seed, beta_early=(1.2, -0.4), beta_late=(-0.6, -0.4)
        )
        tests = ph.flag_tests(ph.run_engine(m, md, fit(m, md).params_.to_dict()).tests)
        planted += bool(tests.loc[tests["term"] == "x_cont", "flag"].any())
        m, md = ph.simulate_left_truncated(900, seed=2000 + seed)
        tests = ph.flag_tests(ph.run_engine(m, md, fit(m, md).params_.to_dict()).tests)
        proportional += bool(tests["flag"].any())
    result = {
        "engine": "survival::cox.zph",
        "versions": ph.preflight(),
        "lifelines_delayed_entry_residuals": lifelines_status,
        "check1_max_abs_difference_from_arithmetic_reference": float(
            np.abs(response.schoenfeld - reference).max()
        ),
        "check2_max_abs_difference_from_entry_ignored": float(
            np.abs(response.schoenfeld - ignored).max()
        ),
        "check3_seeds": ADMISSION_SEEDS,
        "check3_planted_effect_flagged_fraction": planted / ADMISSION_SEEDS,
        "check3_proportional_fixture_flagged_fraction": proportional / ADMISSION_SEEDS,
    }
    # Penalised test (third amendment): zero-penalty agreement with cox.zph, with and without ties.
    agreement = []
    for ties in (False, True):
        m, md = ph.simulate_left_truncated(
            500, seed=13, beta_early=(1.0, -0.4), beta_late=(0.2, -0.4)
        )
        if ties:
            md["exit"] = np.ceil(md["exit"] / 15) * 15
            md["entry"] = np.floor(md["entry"] / 15) * 15
            keep = md["entry"] < md["exit"]
            m, md = m[keep].reset_index(drop=True), md[keep].reset_index(drop=True)
        beta = fit(m, md).params_.to_dict()
        reference = ph.run_engine(m, md, beta).tests
        ours, _ = ph.penalised_ph_test(m, md, beta, penalizer=0.0)
        merged = reference.merge(ours, on=["transform", "term"])
        agreement.append(float(np.abs(merged["chisq_x"] - merged["chisq_y"]).max()))
    planted = proportional = 0
    for seed in range(ADMISSION_SEEDS):
        for late, counter in (((-0.6, -0.4), "planted"), (None, "proportional")):
            m, md = ph.simulate_left_truncated(
                900,
                seed=(3000 if late else 4000) + seed,
                beta_early=(1.2, -0.4) if late else (0.7, -0.4),
                beta_late=late,
            )
            m["x_dup"] = m["x_bin"] * 1.0
            frame = pd.concat([md, m], axis=1)
            beta = (
                CoxPHFitter(penalizer=0.01)
                .fit(frame, duration_col="exit", event_col="event", entry_col="entry")
                .params_.to_dict()
            )
            tests, _ = ph.penalised_ph_test(m, md, beta, penalizer=0.01)
            flagged = ph.flag_tests(tests)
            if counter == "planted":
                planted += bool(flagged.loc[flagged["term"] == "x_cont", "flag"].any())
            else:
                proportional += bool(flagged["flag"].any())
    result["penalised_zero_penalty_max_chisq_difference_from_cox_zph"] = max(agreement)
    result["penalised_planted_effect_flagged_fraction"] = planted / ADMISSION_SEEDS
    result["penalised_proportional_fixture_flagged_fraction"] = proportional / ADMISSION_SEEDS
    result["penalised_passed"] = bool(
        max(agreement) < 1e-6
        and planted / ADMISSION_SEEDS >= 0.80
        and proportional / ADMISSION_SEEDS <= 0.15
    )
    result["passed"] = bool(
        result["check1_max_abs_difference_from_arithmetic_reference"] < 1e-6
        and result["check2_max_abs_difference_from_entry_ignored"] > 1e-2
        and result["check3_planted_effect_flagged_fraction"] >= 0.80
        and result["check3_proportional_fixture_flagged_fraction"] <= 0.15
    )
    return result


def figure(curves: pd.DataFrame, title: str, path: Path) -> None:
    terms = list(dict.fromkeys(curves["term"]))
    columns = min(len(terms), 4)
    rows = int(np.ceil(len(terms) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.0 * rows), squeeze=False)
    for ax, term in zip(axes.ravel(), terms, strict=False):
        block = curves[curves["term"] == term]
        ax.fill_between(block["time"], block["lower"], block["upper"], color="#9ecae1", alpha=0.5)
        ax.plot(block["time"], block["fit"], color="#08519c", lw=1.6)
        centre = float(np.average(block["fit"]))
        ax.axhline(centre, color="0.4", lw=0.8, ls=":")
        for horizon in (365, 730):
            ax.axvline(horizon, color="0.6", lw=0.8, ls="--")
        ax.set_title(TERM_LABELS.get(term, term.replace("_", " ")), fontsize=9)
        ax.set_xlabel("Days since FOLFOX initiation", fontsize=8)
        ax.set_ylabel("Smoothed scaled Schoenfeld residual", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(terms) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_path = SUPP / "frozen_inputs_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
    else:
        before = snapshot()
        before_path.write_text(json.dumps(before, indent=2, sort_keys=True))
    assert snapshot() == before, "Frozen outputs changed since baseline capture"

    admitted = admission()
    (OUT / "engine_admission.json").write_text(json.dumps(admitted, indent=2))
    if not (admitted["passed"] and admitted["penalised_passed"]):
        raise SystemExit("Engine admission failed; no real result produced")

    registry = json.loads((CURRENT / "model_training/model_registry.json").read_text())
    data = pd.read_csv(CURRENT / "analysis_dataset.csv")
    test_frames, curve_frames, checks, cross_checks = [], [], [], []
    for endpoint, model in MODELS:
        view = build_endpoint_view(data, endpoint).data
        for centre in ("DFCI", "MSK", "VICC"):
            key = f"endpoint={endpoint}|design=iecv_outer|holdout={centre}|model={model}"
            state = verify_registry_load(
                CURRENT / "model_training/models", RegistryRecord(**registry[key])
            )
            development = view.loc[view["institution"].ne(centre)].copy()
            matrix, metadata = transform_for_bundle(state, development)
            base = {
                "endpoint": endpoint,
                "model": model,
                "tier": ph.tier_for(endpoint, model),
                "held_out_institution": centre,
            }
            reason = ""
            if tuple(matrix.columns.astype(str)) != state.feature_columns:
                reason = "design columns do not match the bundle"
            elif (len(matrix), int(metadata["event"].sum())) != (
                state.development_patient_count,
                state.development_event_count,
            ):
                reason = "development counts do not match the bundle"
            response = None
            if not reason and model == "spline_cox":
                coefficients = dict(zip(state.feature_columns, state.coefficients, strict=True))
                tests, diagnostics = ph.penalised_ph_test(
                    matrix, metadata, coefficients, penalizer=state.penalizer
                )
                status = (
                    "computed" if diagnostics["stationarity_statistic"] < 1e-6 else "not_computable"
                )
                failure = "" if status == "computed" else "frozen coefficients fail stationarity"
                checks.append(
                    {
                        **base,
                        "design_columns": int(matrix.shape[1]),
                        "design_rank": int(np.linalg.matrix_rank(matrix.to_numpy(float))),
                        "development_patients": len(matrix),
                        "development_events": int(metadata["event"].sum()),
                        "penalised_stationarity_statistic": diagnostics["stationarity_statistic"],
                        "test": "ridge-penalised score test",
                        "status": status,
                        "reason": failure,
                    }
                )
                if status != "computed":
                    test_frames.append(
                        pd.DataFrame([{**base, "status": status, "reason": failure}])
                    )
                    continue
                test_frames.append(
                    ph.flag_tests(tests).assign(
                        **base,
                        n=len(matrix),
                        events=int(metadata["event"].sum()),
                        status="computed",
                        reason="approximate: ridge-penalised information",
                    )
                )
                continue
            if not reason and model == "minimal_cox":
                ours, _ = ph.penalised_ph_test(
                    matrix,
                    metadata,
                    dict(zip(state.feature_columns, state.coefficients, strict=True)),
                    penalizer=0.0,
                )
            if not reason:
                try:
                    response = ph.run_engine(
                        matrix,
                        metadata,
                        dict(zip(state.feature_columns, state.coefficients, strict=True)),
                    )
                except ph.PHAssessmentError as error:
                    message = str(error)
                    if "singular" not in message and "may be infinite" not in message:
                        raise
                    reason = (
                        "unpenalised information matrix is singular or the unpenalised "
                        "likelihood is monotone for this design"
                    )
                if (
                    response is not None
                    and model == "minimal_cox"
                    and not response.frozen_score_statistic < 1e-6
                ):
                    reason = "frozen coefficients are not the optimum of the rebuilt rows"
            checks.append(
                {
                    **base,
                    "design_columns": int(matrix.shape[1]),
                    "design_rank": int(np.linalg.matrix_rank(matrix.to_numpy(float))),
                    "development_patients": len(matrix),
                    "development_events": int(metadata["event"].sum()),
                    "score_statistic_at_frozen_coefficients": (
                        response.frozen_score_statistic if response else None
                    ),
                    "max_difference_from_free_optimum": (
                        response.max_coefficient_difference if response else None
                    ),
                    "status": "not_computable" if reason else "computed",
                    "reason": reason,
                }
            )
            if reason:
                test_frames.append(
                    pd.DataFrame([{**base, "status": "not_computable", "reason": reason}])
                )
                continue
            merged = response.tests.merge(ours, on=["transform", "term"])
            cross_checks.append(float(np.abs(merged["chisq_x"] - merged["chisq_y"]).max()))
            tests = ph.flag_tests(response.tests).assign(
                **base, n=response.n, events=response.events, status="computed", reason=""
            )
            test_frames.append(tests)
            curve_frames.append(response.curves.assign(**base))
            figure(
                response.curves,
                f"{endpoint}, {model.replace('_', ' ')}, {centre} held out "
                f"(development: {response.n} patients, {response.events} events)",
                OUT / f"ph_residuals_{endpoint}_{model}_{centre}.png",
            )
    tests = pd.concat(test_frames, ignore_index=True)
    tests["analysis_label"] = ph.ANALYSIS_LABEL
    tests = tests[[c for c in ph._ALLOWED_TEST_COLUMNS if c in tests.columns]]
    ph.validate_published_tests(tests)
    tests.to_csv(OUT / "ph_tests.csv", index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(OUT / "ph_smoothed_curves.csv", index=False)
    (OUT / "reproduction_checks.json").write_text(json.dumps(checks, indent=2))
    admitted["penalised_real_minimal_cox_max_chisq_difference_from_cox_zph"] = max(cross_checks)
    assert max(cross_checks) < 1e-6, "Penalised implementation disagrees with cox.zph on real data"
    (OUT / "engine_admission.json").write_text(json.dumps(admitted, indent=2))
    trigger = ph.evaluate_trigger(tests)
    (OUT / "ph_trigger.json").write_text(json.dumps(trigger, indent=2))
    assert snapshot() == before, "Frozen outputs modified"
    print(
        json.dumps(
            {"admission": admitted, "trigger": trigger, "frozen_files": len(before)}, indent=1
        )
    )


if __name__ == "__main__":
    main()

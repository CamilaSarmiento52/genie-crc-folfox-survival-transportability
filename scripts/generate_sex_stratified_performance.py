"""Generate held-out sex-stratified OS discrimination and sociodemographic subgroup support.

TRIPOD+AI asks authors to address performance across sociodemographic groups. This script answers
that request with the machinery the study already froze: it reuses the frozen leave-one-centre-out
minimal-Cox bundles, reconstructs each centre's held-out linear predictor exactly as the primary
evaluation did, and computes truncation-aware concordance within each sex, with the pipeline's own
2000-resample percentile bootstrap and three-way reporting-status rule. No model is refitted and no
patient is excluded; a subgroup that misses a support or precision target is reported with its full
interval as ``estimable_imprecise``, never suppressed.

Race/ethnicity subgroups are deliberately *not* evaluated for performance: the OS-eligible
population contains too few patients and events in every non-White group for any estimate to be
informative. This script publishes those support counts (patients and events by centre) so the
manuscript can state the limitation from a published aggregate rather than a hand-typed number.

Each ``All`` row is a positive control: it recomputes the centre's overall held-out concordance and
this script asserts the result equals the frozen primary-evaluation ``performance_summary.csv``
estimate. A reconstruction that cannot reproduce the published number must fail here, not be
published.

Outputs are an aggregate bundle at ``analysis/out/current/fairness/`` with a manifest carrying
``artifact_hashes``, so the manuscript build can hash-verify it like the other canonical inputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.bootstrap import (
    run_bootstrap,
    summarize_bootstrap,
)
from crc_folfox.evaluate.contracts import BOOTSTRAP_RESAMPLES, METRIC_REGISTRY
from crc_folfox.evaluate.discrimination import truncation_aware_concordance
from crc_folfox.evaluate.prediction import predict_for_frame
from crc_folfox.evaluate.reporting import classify_reporting_status
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.model_training.contracts import ModelScope
from crc_folfox.preprocessing.contracts import build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits, derive_seed

ENDPOINT = "OS"
MODEL = "minimal_cox"
SEX_LEVELS = ("Female", "Male")
CENTRES = ("DFCI", "MSK", "VICC")
#: The reconstruction must reproduce the frozen published concordance to this tolerance; the
#: pipeline and this script share one estimator and one prediction path, so agreement is exact up
#: to floating-point noise and anything larger means the reconstruction is wrong.
POSITIVE_CONTROL_TOLERANCE = 1e-9


def _token_safe(level: str) -> str:
    """A key usable inside the manuscript token grammar ([a-zA-Z0-9_.\\-@], no spaces/slashes)."""

    key = re.sub(r"[^a-zA-Z0-9]+", "_", level).strip("_").lower()
    if not key:
        raise ValueError(f"level {level!r} produced an empty token key")
    return key


def _subgroup_bootstrap_seed(centre: str, group: str) -> int:
    """Deterministic child seed per (endpoint, centre, group), same mechanic as the pipeline."""

    return derive_seed(
        endpoint=ENDPOINT,
        design=f"sex_subgroup_{group}",
        held_out_institution=centre,
        repeat=None,
        fold=None,
        purpose="evaluation_bootstrap",
    )


def _concordance_summary(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    lp: np.ndarray,
    *,
    centre: str,
    group: str,
) -> dict[str, object]:
    point = truncation_aware_concordance(entry, exit_, event, lp)
    n = len(entry)
    rng = np.random.default_rng(_subgroup_bootstrap_seed(centre, group))
    index_matrix = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    pair_indices = np.triu_indices(n, k=1)

    def metric(indices: np.ndarray) -> float | None:
        return truncation_aware_concordance(
            entry[indices],
            exit_[indices],
            event[indices],
            lp[indices],
            pair_indices=pair_indices,
        ).value

    summary = summarize_bootstrap(point.value, run_bootstrap(index_matrix, metric))
    n_events = int(event.sum())
    n_event_free = int((event == 0).sum())
    status = classify_reporting_status(
        value=summary.value,
        ci_lower=summary.ci_lower,
        ci_upper=summary.ci_upper,
        n_events=n_events,
        n_event_free=n_event_free,
        bootstrap_finite_fraction=summary.finite_fraction,
        precision_target=METRIC_REGISTRY["concordance"].precision_target,
    )
    return {
        "endpoint": ENDPOINT,
        "design": "iecv",
        "centre": centre,
        "model": MODEL,
        "group": group,
        "metric": "concordance",
        "estimate": summary.value,
        "ci_lower": summary.ci_lower,
        "ci_upper": summary.ci_upper,
        "n_patients": n,
        "n_events": n_events,
        "n_event_free": n_event_free,
        "bootstrap_finite_fraction": summary.finite_fraction,
        "reporting_status": status.status,
        "reason": ";".join(status.reasons),
    }


def _support_rows(view_data: pd.DataFrame) -> list[dict[str, object]]:
    """OS-eligible patients and deaths by centre per sociodemographic level (aggregate only)."""

    rows: list[dict[str, object]] = []
    for variable in ("naaccr_sex_code", "race_ethnicity"):
        levels = sorted(view_data[variable].astype(str).unique())
        for level in levels:
            cell = view_data[view_data[variable].astype(str) == level]
            for centre in (*CENTRES, "All"):
                subset = (
                    cell
                    if centre == "All"
                    else cell[cell["institution"].astype(str) == centre]
                )
                rows.append(
                    {
                        "variable": variable,
                        "level_key": _token_safe(level),
                        "level_label": level,
                        "centre": centre,
                        "n_patients": int(len(subset)),
                        "n_events": int(subset["event"].sum()),
                    }
                )
    return rows


def generate(workspace: Path) -> Path:
    analysis_root = workspace / "analysis" / "out" / "current"
    output_dir = analysis_root / "fairness"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(analysis_root / "analysis_dataset.csv")
    view = build_endpoint_view(data, ENDPOINT)
    registry = json.loads(
        (analysis_root / "model_training" / "model_registry.json").read_text()
    )
    models_dir = analysis_root / "model_training" / "models"
    published = pd.read_csv(analysis_root / "evaluation" / "performance_summary.csv")

    performance_rows: list[dict[str, object]] = []
    for split in build_iecv_splits(view):
        _, assessment = split.frames(view)
        centre = split.held_out_institution
        model_id = ModelScope(split).model_id(MODEL)
        state = verify_registry_load(models_dir, RegistryRecord(**registry[model_id]))
        prediction = predict_for_frame(state, assessment, (730.0,))

        # Predictions come back in the transform's row order; align sex by patient key rather
        # than assuming the assessment frame order survived the transform.
        sex_by_key = dict(
            zip(
                assessment["patient_key"].astype(str),
                assessment["naaccr_sex_code"].astype(str),
                strict=True,
            )
        )
        sex = np.array([sex_by_key[key] for key in prediction.patient_key])
        lp = prediction.linear_predictor

        for group in ("All", *SEX_LEVELS):
            mask = np.ones(len(sex), dtype=bool) if group == "All" else sex == group
            performance_rows.append(
                _concordance_summary(
                    prediction.entry[mask],
                    prediction.exit[mask],
                    prediction.event[mask],
                    lp[mask],
                    centre=centre,
                    group=group,
                )
            )

        control = performance_rows[-3]  # the "All" row appended first for this centre
        frozen = published[
            (published["endpoint"] == ENDPOINT)
            & (published["design"] == "iecv")
            & (published["centre"] == centre)
            & (published["model"] == MODEL)
            & (published["metric"] == "concordance")
        ]
        if len(frozen) != 1:
            raise RuntimeError(f"expected one frozen concordance row for {centre}")
        difference = abs(float(control["estimate"]) - float(frozen.iloc[0]["estimate"]))
        if difference > POSITIVE_CONTROL_TOLERANCE:
            raise RuntimeError(
                f"positive control failed in {centre}: reconstructed concordance "
                f"{control['estimate']} differs from frozen {frozen.iloc[0]['estimate']} "
                f"by {difference}"
            )

    performance = pd.DataFrame(performance_rows)
    support = pd.DataFrame(_support_rows(view.data))

    performance_path = output_dir / "sex_stratified_performance.csv"
    support_path = output_dir / "sociodemographic_support.csv"
    performance.to_csv(performance_path, index=False)
    support.to_csv(support_path, index=False)

    from crc_folfox.report.inputs import sha256_of

    manifest = {
        "phase": "phase-8-manuscript-draft (revision: TRIPOD+AI fairness reporting)",
        "provenance": (
            "post hoc: computed from the frozen Phase 4 minimal-Cox IECV bundles and the frozen "
            "Phase 2 dataset after the primary results were known; no model was refitted"
        ),
        "endpoint": ENDPOINT,
        "model": MODEL,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed_mechanic": (
            "derive_seed(endpoint='OS', design='sex_subgroup_<group>', "
            "held_out_institution=<centre>, purpose='evaluation_bootstrap')"
        ),
        "positive_control": (
            "each centre's All-row concordance reproduced the frozen "
            "evaluation/performance_summary.csv estimate to within "
            f"{POSITIVE_CONTROL_TOLERANCE}"
        ),
        "artifact_hashes": {
            performance_path.name: sha256_of(performance_path),
            support_path.name: sha256_of(support_path),
        },
    }
    manifest_path = output_dir / "fairness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    print(generate(Path(__file__).resolve().parents[1]))

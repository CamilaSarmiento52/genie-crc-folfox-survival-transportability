"""Read-only export and independent reconstruction of the six frozen minimal-Cox models.

Writes the machine-readable model specifications (Supplementary Table S31) under
`analysis/out/current/supplementary/model_specification/` and verifies that predictions rebuilt
from the exported files, using only `predict.py`, match the saved bundles. No model is refitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from predict import baseline, design, predict

from crc_folfox.evaluate.prediction import entry_conditional_risk, unconditional_risk
from crc_folfox.model_training.bundles import RegistryRecord, verify_registry_load
from crc_folfox.preprocessing.contracts import build_endpoint_view

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "analysis/out/current"
OUT = CURRENT / "supplementary"
SPEC_KEYS = {
    "schema_version",
    "model_id",
    "endpoint",
    "held_out_institution",
    "training_centres",
    "source_bundle_sha256",
    "lifelines_version",
    "predictors",
    "feature_columns",
    "coefficients",
    "cox_norm_mean",
    "baseline",
    "preprocessing",
}
PRE_KEYS = {
    "imputation_values",
    "references",
    "level_maps",
    "encoded_levels",
    "scaling",
    "missing_indicators",
    "removed_zero_variance",
    "output_columns",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {
        str(p.relative_to(CURRENT)): sha(p)
        for p in CURRENT.rglob("*")
        if p.is_file() and OUT not in p.parents
    }


def validate(spec):
    if set(spec) != SPEC_KEYS or set(spec["preprocessing"]) != PRE_KEYS:
        raise ValueError("Unexpected model export field")
    n = len(spec["feature_columns"])
    if not n == len(spec["coefficients"]) == len(spec["cox_norm_mean"]):
        raise ValueError("Incomplete model dimensions")
    if spec["feature_columns"] != spec["preprocessing"]["output_columns"]:
        raise ValueError("Cross-scope feature mismatch")
    times, survival = spec["baseline"]["times"], spec["baseline"]["survival"]
    if not times or len(times) != len(survival) or times != sorted(set(times)):
        raise ValueError("Invalid baseline function")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in survival):
        raise ValueError("Invalid baseline values")
    if any(b > a for a, b in zip(survival, survival[1:], strict=False)):
        raise ValueError("Baseline survival increases")


def export(state, record):
    prep = state.preprocessor.aggregate_state()
    spec = dict(
        schema_version=1,
        model_id=state.model_id,
        endpoint=state.endpoint,
        held_out_institution=state.held_out_institution,
        training_centres=[c for c in ("DFCI", "MSK", "VICC") if c != state.held_out_institution],
        source_bundle_sha256=record.per_file_sha256,
        lifelines_version=state.lifelines_version,
        predictors=[asdict(d) for d in state.preprocessor.definitions],
        feature_columns=list(state.feature_columns),
        coefficients=list(state.coefficients),
        cox_norm_mean=list(state.norm_mean),
        baseline={
            "times": list(state.baseline_survival_times),
            "survival": list(state.baseline_survival_values),
            "time_units": "days since first post-metastatic FOLFOX",
            "lookup": "last stored time <= query; 1 before first stored time",
        },
        preprocessing={k: prep[k] for k in PRE_KEYS},
    )
    validate(spec)
    return spec


def synthetic_profiles(spec):
    base = {
        d["name"]: (
            spec["preprocessing"]["references"][d["name"]]
            if d["name"] in spec["preprocessing"]["references"]
            else 55.0
            if d["name"] == "age_at_folfox"
            else 2.0
        )
        for d in spec["predictors"]
    }
    rows = [base]
    for name, mapping in spec["preprocessing"]["level_maps"].items():
        for level in [*mapping, "__UNSEEN_TEST__"]:
            rows.append({**base, name: level})
        if name in spec["preprocessing"]["imputation_values"]:
            rows.append({**base, name: None})
    for age, count in [(30, 0), (80, 6)]:
        rows.append({**base, "age_at_folfox": age, "number_metastatic_sites": count})
    return rows


def verify(state, spec, frame):
    matrix = state.preprocessor.transform(frame).matrix
    raw = frame.loc[:, list(state.preprocessor.feature_names)].to_dict("records")
    ours = np.asarray([design(spec, row) for row in raw])
    np.testing.assert_allclose(ours, matrix.to_numpy(), rtol=0, atol=1e-12)
    entries = frame["entry"].to_numpy(float)
    u = unconditional_risk(state, matrix)
    c = entry_conditional_risk(state, matrix, entries)
    ou = np.asarray([[predict(spec, row, t) for t in (365, 730)] for row in raw])
    oc = np.asarray(
        [
            [predict(spec, row, t, e) if e <= t else np.nan for t in (365, 730)]
            for row, e in zip(raw, entries, strict=True)
        ],
        dtype=float,
    )
    np.testing.assert_allclose(ou, u, rtol=0, atol=1e-9)
    np.testing.assert_allclose(oc, c, rtol=0, atol=1e-9, equal_nan=True)
    return max(float(np.max(np.abs(ou - u))), float(np.nanmax(np.abs(oc - c))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before_path = OUT / "frozen_inputs_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        assert snapshot() == before, "Frozen outputs changed since baseline capture"
    else:
        before = snapshot()
        before_path.write_text(json.dumps(before, indent=2, sort_keys=True))
    output = OUT / "model_specification"
    output.mkdir(exist_ok=True)
    registry = json.loads((CURRENT / "model_training/model_registry.json").read_text())
    data = pd.read_csv(CURRENT / "analysis_dataset.csv")
    summaries = []
    examples = []
    coefficients = []
    baselines = []
    inventory = []
    for key, record_dict in registry.items():
        if "|model=minimal_cox" not in key or "design=iecv_outer" not in key:
            continue
        record = RegistryRecord(**record_dict)
        state = verify_registry_load(CURRENT / "model_training/models", record)
        spec = export(state, record)
        filename = f"{state.endpoint}_{state.held_out_institution}.json"
        path = output / filename
        path.write_text(json.dumps(spec, indent=2, sort_keys=True, allow_nan=False))
        spec = json.loads(path.read_text())
        validate(spec)
        view = build_endpoint_view(data, state.endpoint).data
        held = view.loc[view.institution.eq(state.held_out_institution)].copy()
        real_error = verify(state, spec, held)
        profiles = synthetic_profiles(spec)
        syn = pd.DataFrame(profiles)
        syn["patient_key"] = [f"SYNTHETIC_{i}" for i in range(len(syn))]
        syn["institution"] = state.held_out_institution
        syn["entry"] = np.resize([0, 215, 365, 730, 731], len(syn))
        syn["exit"] = 1000.0
        syn["event"] = 0
        syn_error = verify(state, spec, syn)
        for t in (365, 730):
            baselines.append(
                dict(
                    endpoint=state.endpoint,
                    held_out=state.held_out_institution,
                    horizon=t,
                    baseline_survival=baseline(spec, t),
                )
            )
        for name, beta, mean in zip(
            state.feature_columns, state.coefficients, state.norm_mean, strict=True
        ):
            coefficients.append(
                dict(
                    endpoint=state.endpoint,
                    held_out=state.held_out_institution,
                    feature=name,
                    coefficient=beta,
                    cox_norm_mean=mean,
                )
            )
        example = profiles[0]
        examples.append(
            dict(
                model_file=filename,
                synthetic_profile=example,
                entry_days=215,
                unconditional_730=predict(spec, example, 730),
                entry_conditional_730=predict(spec, example, 730, 215),
            )
        )
        summaries.append(
            dict(
                model_id=key,
                assessment_patients=len(held),
                synthetic_profiles=len(syn),
                max_absolute_error=max(real_error, syn_error),
                atol=1e-9,
                rtol=0,
                status="passed",
                source_bundle_sha256=record.per_file_sha256,
            )
        )
        inventory.append(dict(model_id=key, filename=filename, sha256=sha(path)))
    assert len(inventory) == 6
    for name, rows in [("coefficients.csv", coefficients), ("baseline_horizons.csv", baselines)]:
        with (output / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "synthetic_examples.json").write_text(json.dumps(examples, indent=2))
    (output / "verification.json").write_text(json.dumps(summaries, indent=2))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "models": inventory,
                "scope": (
                    "Six frozen IECV minimal-Cox models; no final deployable all-centre model "
                    "selected"
                ),
                "fitting_performed": False,
            },
            indent=2,
        )
    )
    shutil.copy2(Path(__file__).with_name("predict.py"), output / "predict.py")
    assert snapshot() == before, "Frozen outputs modified"
    print(
        json.dumps(
            {
                "models_verified": len(summaries),
                "max_absolute_error": max(s["max_absolute_error"] for s in summaries),
                "assessment_rows": sum(s["assessment_patients"] for s in summaries),
                "frozen_files_unchanged": len(before),
            }
        )
    )


if __name__ == "__main__":
    main()

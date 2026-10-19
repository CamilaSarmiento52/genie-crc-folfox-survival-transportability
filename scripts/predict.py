"""Independent, standard-library predictor for the exported research Cox specifications."""

from __future__ import annotations

import bisect
import math


def design(spec: dict, raw: dict) -> list[float]:
    """Transform a raw predictor profile using only published, training-fitted parameters."""
    p = spec["preprocessing"]
    columns = {}
    for definition in spec["predictors"]:
        name = definition["name"]
        value = raw.get(name)
        missing = value is None or isinstance(value, float) and math.isnan(value)
        if definition["role"] == "categorical" or definition["encoding"] == "training_only_one_hot":
            if missing:
                if name not in p["imputation_values"]:
                    raise ValueError("Required categorical predictor is missing")
                value = p["imputation_values"][name]
            value = str(value)
            if not value or value != value.strip():
                raise ValueError("Categorical values must be normalized")
            levels = p["encoded_levels"][name]
            fallback = "__RARE__" if "__RARE__" in levels else p["references"][name]
            mapped = p["level_maps"][name].get(value, fallback)
            if name + "__missing" in p["missing_indicators"]:
                columns[name + "__missing"] = float(missing)
            for level in levels:
                if level != p["references"][name]:
                    columns[name + "__" + level] = float(mapped == level)
        else:
            if missing:
                raise ValueError("Required numeric predictor is missing")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Numeric predictor is nonfinite")
            if definition["role"] == "continuous":
                scale = p["scaling"][name]
                columns[name] = (
                    (value - scale["mean"]) / scale["population_std"]
                    if scale["population_std"]
                    else 0.0
                )
            else:
                if value not in (0.0, 1.0):
                    raise ValueError("Binary predictor must be zero or one")
                columns[name] = value
    return [columns[name] for name in spec["feature_columns"]]


def baseline(spec: dict, time: float) -> float:
    if not math.isfinite(time) or time < 0:
        raise ValueError("Time must be finite and nonnegative")
    index = bisect.bisect_right(spec["baseline"]["times"], time) - 1
    return 1.0 if index < 0 else spec["baseline"]["survival"][index]


def predict(spec: dict, raw: dict, time: float, entry: float | None = None) -> float | None:
    """Return mortality/event risk, or None when entry exceeds the prediction horizon.

    This reproduces saved research predictions; it is not a clinical decision tool.
    """
    z = design(spec, raw)
    beta, center = spec["coefficients"], spec["cox_norm_mean"]
    if not len(z) == len(beta) == len(center):
        raise ValueError("Model dimensions disagree")
    hazard = math.exp(sum((v - m) * b for v, m, b in zip(z, center, beta, strict=True)))
    s = baseline(spec, time)
    if entry is not None:
        se = baseline(spec, entry)
        if entry > time:
            return None
        if se <= 0:
            raise ValueError("Entry-conditional survival is undefined at zero baseline survival")
        s /= se
    risk = 1.0 - s**hazard
    if not math.isfinite(risk) or not -1e-9 <= risk <= 1 + 1e-9:
        raise ValueError("Computed risk is invalid")
    return risk

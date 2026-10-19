"""Proportional-hazards assessment of frozen Cox bundles under delayed entry.

Residuals and the Grambsch-Therneau test are evaluated by `survival::cox.zph` in the pinned R
library at the frozen coefficients; no coefficient is re-estimated. Holm adjustment, flags and the
trigger rule are applied here, mechanically, from the frozen rule.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from crc_folfox.explain.feature_groups import build_feature_groups

ANALYSIS_LABEL = "post_hoc_reviewer_requested"
PRIMARY_TRANSFORM = "km"
TRANSFORMS = (PRIMARY_TRANSFORM, "log")
ALPHA = 0.05
GLOBAL_TERM = "GLOBAL"
REFIT_TOLERANCE = 1e-6
TIER_PRIMARY, TIER_SUPPORTIVE, TIER_DESCRIPTIVE = "primary", "supportive", "descriptive"
SCOPES_REQUIRED_FOR_TRIGGER = 2
_SCRIPT = Path(__file__).resolve().parents[3] / "r/ph_adapter.R"
_ALLOWED_TEST_COLUMNS = (
    "endpoint",
    "model",
    "tier",
    "held_out_institution",
    "term",
    "transform",
    "chisq",
    "df",
    "p_value",
    "p_holm",
    "n",
    "events",
    "flag",
    "status",
    "reason",
    "analysis_label",
)


class PHAssessmentError(RuntimeError):
    """Raised when the assessment cannot be carried out as specified."""


def tier_for(endpoint: str, model_name: str) -> str:
    if model_name == "minimal_cox":
        return TIER_PRIMARY if endpoint == "OS" else TIER_SUPPORTIVE
    if model_name == "spline_cox" and endpoint == "OS":
        return TIER_DESCRIPTIVE
    raise PHAssessmentError(f"'{model_name}'/{endpoint} is not covered by the assessment")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise PHAssessmentError("Holm adjustment needs finite p-values in [0, 1]")
    order = np.argsort(p, kind="stable")
    scaled = (p.size - np.arange(p.size)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return [float(v) for v in out]


def schoenfeld_reference(
    matrix: np.ndarray, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled Schoenfeld residuals with delayed-entry risk sets, for events without ties.

    A subject is at risk at event time `t` only when `entry < t <= exit`. This is an independent
    arithmetic reference for the R engine and deliberately refuses tied event times.
    """
    x = np.asarray(matrix, dtype=float)
    entry, exit_ = np.asarray(entry, dtype=float), np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    times = exit_[event == 1]
    if len(np.unique(times)) != len(times):
        raise PHAssessmentError("The arithmetic reference is defined only without tied events")
    weights = np.exp(x @ np.asarray(beta, dtype=float))
    order = np.argsort(times)
    rows = np.flatnonzero(event == 1)[order]
    residuals = np.empty((len(rows), x.shape[1]))
    for k, i in enumerate(rows):
        at_risk = (entry < exit_[i]) & (exit_ >= exit_[i])
        w = weights[at_risk]
        residuals[k] = x[i] - (w[:, None] * x[at_risk]).sum(axis=0) / w.sum()
    return times[order], residuals


@dataclass(frozen=True)
class EngineResponse:
    n: int
    events: int
    frozen_score_statistic: float
    max_coefficient_difference: float
    tests: pd.DataFrame
    curves: pd.DataFrame
    schoenfeld_times: np.ndarray | None = None
    schoenfeld: np.ndarray | None = None


def preflight(rscript: str = "Rscript") -> dict[str, str]:
    if shutil.which(rscript) is None:
        raise PHAssessmentError("Rscript is not available")
    done = subprocess.run(
        [rscript, _SCRIPT.name, "--preflight"],
        cwd=_SCRIPT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise PHAssessmentError(f"R preflight failed: {done.stderr.strip()}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def run_engine(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    coefficients: Mapping[str, float],
    *,
    terms: Mapping[str, Sequence[str]] | None = None,
    return_residuals: bool = False,
    curve_points: int = 40,
    rscript: str = "Rscript",
) -> EngineResponse:
    """Evaluate `cox.zph` at the frozen coefficients on a counting-process response."""
    columns = tuple(str(c) for c in matrix.columns)
    if set(columns) != set(coefficients):
        raise PHAssessmentError("Coefficient names do not match the design columns")
    terms = dict(terms) if terms is not None else dict(build_feature_groups(columns))
    if sorted(c for members in terms.values() for c in members) != sorted(columns):
        raise PHAssessmentError("Terms must partition the design columns exactly")
    entry = metadata["entry"].to_numpy(float)
    exit_ = metadata["exit"].to_numpy(float)
    if not (entry < exit_).all() or (entry < 0).any():
        raise PHAssessmentError("Counting-process rows need 0 <= entry < exit")
    exchange = pd.concat(
        [
            metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
            matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    with tempfile.TemporaryDirectory(prefix="ph_exchange_") as tmp:
        tmp_path = Path(tmp)
        exchange.to_csv(tmp_path / "train.csv", index=False)
        (tmp_path / "spec.json").write_text(
            json.dumps(
                {
                    "coefficients": {k: float(v) for k, v in coefficients.items()},
                    "terms": {k: list(v) for k, v in terms.items()},
                    "transforms": list(TRANSFORMS),
                    "return_residuals": return_residuals,
                    "curve_points": curve_points,
                }
            )
        )
        done = subprocess.run(
            [
                rscript,
                _SCRIPT.name,
                "--assess",
                str(tmp_path / "train.csv"),
                str(tmp_path / "spec.json"),
                str(tmp_path / "response.json"),
            ],
            cwd=_SCRIPT.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise PHAssessmentError(f"R assessment failed: {done.stderr.strip()[-500:]}")
        response = json.loads((tmp_path / "response.json").read_text())
    tests = pd.DataFrame(response["tests"]).rename(columns={"p": "p_value"})
    curve_rows = [
        {"term": c["term"], "time": t, "fit": f, "lower": lo, "upper": hi}
        for c in response["curves"]
        for t, f, lo, hi in zip(c["time"], c["fit"], c["lower"], c["upper"], strict=True)
    ]
    difference = float(
        np.max(
            np.abs(
                np.asarray(response["free_coefficients"], float)
                - np.asarray(response["frozen_coefficients"], float)
            )
        )
    )
    residuals = response.get("schoenfeld")
    return EngineResponse(
        n=int(response["n"]),
        events=int(response["events"]),
        frozen_score_statistic=float(response["frozen_score_statistic"]),
        max_coefficient_difference=difference,
        tests=tests,
        curves=pd.DataFrame(curve_rows),
        schoenfeld_times=(
            np.asarray(response["schoenfeld_times"], float) if residuals is not None else None
        ),
        schoenfeld=np.asarray(residuals, float) if residuals is not None else None,
    )


def flag_tests(tests: pd.DataFrame) -> pd.DataFrame:
    """Attach Holm-adjusted p-values (within model and transform, terms only) and term flags.

    A term is flagged when its Holm-adjusted p-value on the primary transform is below 0.05. The
    global row is flagged on its unadjusted p-value. Secondary-transform rows are never flagged.
    """
    out = tests.copy()
    out["p_holm"] = np.nan
    out["flag"] = False
    for transform, block in out.groupby("transform"):
        term_rows = block.index[block["term"] != GLOBAL_TERM]
        out.loc[term_rows, "p_holm"] = holm_adjust(out.loc[term_rows, "p_value"].tolist())
        if transform != PRIMARY_TRANSFORM:
            continue
        out.loc[term_rows, "flag"] = out.loc[term_rows, "p_holm"] < ALPHA
        global_rows = block.index[block["term"] == GLOBAL_TERM]
        out.loc[global_rows, "flag"] = out.loc[global_rows, "p_value"] < ALPHA
    return out


def evaluate_trigger(tests: pd.DataFrame) -> dict[str, object]:
    """Apply the frozen two-of-three-scope rule to Primary-tier rows on the primary transform."""
    primary = tests[
        (tests["tier"] == TIER_PRIMARY)
        & (tests["transform"] == PRIMARY_TRANSFORM)
        & (tests["status"] == "computed")
    ]
    flagged_scopes: dict[str, list[str]] = {}
    for term, block in primary.groupby("term"):
        scopes = sorted(block.loc[block["flag"].astype(bool), "held_out_institution"].unique())
        if scopes:
            flagged_scopes[str(term)] = scopes
    firing = {
        term: scopes
        for term, scopes in flagged_scopes.items()
        if len(scopes) >= SCOPES_REQUIRED_FOR_TRIGGER
    }
    other = tests[
        (tests["tier"] != TIER_PRIMARY)
        & (tests["transform"] == PRIMARY_TRANSFORM)
        & tests["flag"].astype(bool)
    ]
    return {
        "rule": (
            "Trigger fires when the same term, or the global test, is flagged in at least two of "
            "the three OS minimal-Cox development scopes on the Kaplan-Meier transform. A term is "
            "flagged at Holm-adjusted p < 0.05 within model; the global test at p < 0.05."
        ),
        "primary_flags_by_term": flagged_scopes,
        "trigger_fired": bool(firing),
        "firing_terms": firing,
        "non_triggering_flags": [
            {
                "endpoint": r.endpoint,
                "model": r.model,
                "held_out_institution": r.held_out_institution,
                "term": r.term,
            }
            for r in other.itertuples()
        ],
        "analysis_label": ANALYSIS_LABEL,
    }


def validate_published_tests(frame: pd.DataFrame) -> None:
    """Refuse any column outside the aggregate allowlist."""
    unexpected = [c for c in frame.columns if c not in _ALLOWED_TEST_COLUMNS]
    if unexpected:
        raise PHAssessmentError(f"Non-allowlisted column(s) in published tests: {unexpected}")


def simulate_left_truncated(
    n: int,
    *,
    seed: int,
    beta_early: Sequence[float] = (0.7, -0.4),
    beta_late: Sequence[float] | None = None,
    change_time: float = 300.0,
    base_rate: float = 1 / 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic delayed-entry survival data, proportional unless `beta_late` differs."""
    rng = np.random.default_rng(seed)
    early = np.asarray(beta_early, float)
    late = early if beta_late is None else np.asarray(beta_late, float)
    matrices, entries, exits, events = [], [], [], []
    kept = 0
    while kept < n:
        size = 4 * n
        x = np.column_stack([rng.normal(size=size), rng.binomial(1, 0.4, size=size)])
        h1, h2 = base_rate * np.exp(x @ early), base_rate * np.exp(x @ late)
        target = rng.exponential(size=size)
        time = np.where(
            target <= h1 * change_time, target / h1, change_time + (target - h1 * change_time) / h2
        )
        entry = np.where(rng.random(size) < 0.3, 0.0, rng.uniform(0, 500, size))
        censor = entry + rng.uniform(100, 1500, size)
        keep = time > entry
        exit_ = np.minimum(time, censor)
        matrices.append(x[keep])
        entries.append(entry[keep])
        exits.append(exit_[keep])
        events.append((time <= censor)[keep].astype(int))
        kept += int(keep.sum())
    matrix = pd.DataFrame(np.vstack(matrices)[:n], columns=["x_cont", "x_bin"])
    metadata = pd.DataFrame(
        {
            "entry": np.concatenate(entries)[:n],
            "exit": np.concatenate(exits)[:n],
            "event": np.concatenate(events)[:n],
        }
    )
    return matrix, metadata


__all__ = [
    "ALPHA",
    "ANALYSIS_LABEL",
    "EngineResponse",
    "PHAssessmentError",
    "REFIT_TOLERANCE",
    "evaluate_trigger",
    "flag_tests",
    "holm_adjust",
    "penalised_ph_test",
    "preflight",
    "run_engine",
    "schoenfeld_reference",
    "simulate_left_truncated",
    "tier_for",
    "validate_published_tests",
]


# --------------------------------------------------------------------------- penalised test


def _km_left_limit(entry: np.ndarray, exit_: np.ndarray, event: np.ndarray) -> np.ndarray:
    """`1 - S(t-)` of the delayed-entry Kaplan-Meier estimate at each distinct event time."""
    times = np.unique(exit_[event == 1])
    survival, values = 1.0, []
    for t in times:
        values.append(1.0 - survival)
        at_risk = int(((entry < t) & (exit_ >= t)).sum())
        deaths = int(((exit_ == t) & (event == 1)).sum())
        survival *= 1.0 - deaths / at_risk
    return np.asarray(values)


def penalised_ph_test(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    coefficients: Mapping[str, float],
    *,
    penalizer: float,
    terms: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Grambsch-Therneau score test using the ridge-penalised information `I + lambda * D`.

    `lambda * D = n * penalizer * diag(sigma_j ** 2)` is the raw-scale form of the lifelines
    penalty. With `penalizer = 0` this is the `cox.zph` score test. Returns the test table and
    diagnostics, including the penalised stationarity statistic of the supplied coefficients.
    """
    columns = [str(c) for c in matrix.columns]
    if set(columns) != set(coefficients):
        raise PHAssessmentError("Coefficient names do not match the design columns")
    terms = dict(terms) if terms is not None else dict(build_feature_groups(tuple(columns)))
    x = matrix.to_numpy(float)
    x = x - x.mean(axis=0)
    beta = np.asarray([coefficients[c] for c in columns], float)
    entry = metadata["entry"].to_numpy(float)
    exit_ = metadata["exit"].to_numpy(float)
    event = metadata["event"].to_numpy(int)
    n, p = x.shape
    weight = np.exp(x @ beta)
    event_times = np.unique(exit_[event == 1])
    g_by_transform = {"km": _km_left_limit(entry, exit_, event), "log": np.log(event_times)}

    score = np.zeros(p)
    info = np.zeros((p, p))
    per_time_u = np.zeros((len(event_times), p))
    per_time_v = np.zeros((len(event_times), p, p))
    deaths_at = np.zeros(len(event_times))
    for k, t in enumerate(event_times):
        at_risk = (entry < t) & (exit_ >= t)
        dead = at_risk & (exit_ == t) & (event == 1)
        d = int(dead.sum())
        w_r, x_r = weight[at_risk], x[at_risk]
        w_d, x_d = weight[dead], x[dead]
        s0, s1 = w_r.sum(), w_r @ x_r
        s2 = (x_r * w_r[:, None]).T @ x_r
        d0, d1 = w_d.sum(), w_d @ x_d
        d2 = (x_d * w_d[:, None]).T @ x_d
        u = x_d.sum(axis=0)
        v = np.zeros((p, p))
        for step in range(d):
            f = step / d
            a0, a1, a2 = s0 - f * d0, s1 - f * d1, s2 - f * d2
            mean = a1 / a0
            u = u - mean
            v += a2 / a0 - np.outer(mean, mean)
        per_time_u[k], per_time_v[k], deaths_at[k] = u, v, d
        score += u
        info += v

    sigma = matrix.to_numpy(float).std(axis=0, ddof=1)
    penalty = n * penalizer * sigma**2
    info_pen = info + np.diag(penalty)
    score_pen = score - penalty * beta
    solved = np.linalg.solve(info_pen, score_pen) if penalizer > 0 else None
    if penalizer == 0:
        stationarity = float(score @ np.linalg.lstsq(info, score, rcond=None)[0])
    else:
        stationarity = float(score_pen @ solved)

    index = {c: i for i, c in enumerate(columns)}
    rows = []
    for transform, g_raw in g_by_transform.items():
        g = g_raw - np.average(g_raw, weights=deaths_at)
        u_theta = (g[:, None] * per_time_u).sum(axis=0)
        i_bt = np.einsum("k,kij->ij", g, per_time_v)
        i_tt = np.einsum("k,kij->ij", g**2, per_time_v)
        for name, members in [*terms.items(), (GLOBAL_TERM, columns)]:
            j = [index[c] for c in members]
            cross = i_bt[:, j]
            reduced = i_tt[np.ix_(j, j)] - cross.T @ np.linalg.solve(info_pen, cross)
            reduced = (reduced + reduced.T) / 2
            eigenvalues = np.linalg.eigvalsh(reduced)
            rank = int((eigenvalues > 1e-9 * max(eigenvalues.max(), 1e-300)).sum())
            chisq = float(u_theta[j] @ np.linalg.pinv(reduced, hermitian=True) @ u_theta[j])
            rows.append({"transform": transform, "term": name, "chisq": chisq, "df": rank})
    tests = pd.DataFrame(rows)
    tests["p_value"] = chi2.sf(tests["chisq"], tests["df"])
    diagnostics = {
        "n": float(n),
        "events": float(event.sum()),
        "stationarity_statistic": stationarity,
    }
    return tests, diagnostics

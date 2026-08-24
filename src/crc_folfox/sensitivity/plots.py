"""Sensitivity-analysis figures: seven prespecified plots, one function each, every one reading only
the already-computed aggregate `SensitivityResult` frames (or the small per-endpoint curve tables
the workflow threads in alongside them) -- no plot recomputes a metric or refits anything. Every
function is a no-op (writes nothing) when its source frame is empty, so a smaller synthetic exercise
of the workflow never fails on a missing figure.

Follows this codebase's existing figure conventions (`evaluate/workflow.py` and the two figures
`sensitivity/workflow.py` already drew before this module existed): `Agg` backend, `dpi=110`,
`fig.tight_layout()` before saving, small (6-7pt) legend/label fonts, no seaborn or shared rcParams.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crc_folfox.evaluate.contracts import CENTRES, RISK_CLIP  # noqa: E402
from crc_folfox.sensitivity.calibration_drift import (  # noqa: E402
    LOCAL_ADAPTATION_FRACTIONS,
    STEP_INTERCEPT_ONLY,
    STEP_INTERCEPT_SLOPE,
    STEP_UNTOUCHED,
)
from crc_folfox.sensitivity.contracts import (  # noqa: E402
    CONCORDANCE_CLASSIFICATION_CONCORDANT,
    CONCORDANCE_CLASSIFICATION_DISCORDANT,
    SEQUENCING_LAG_THRESHOLDS,
)

_THRESHOLD_ORDER = tuple(label for label, _ in SEQUENCING_LAG_THRESHOLDS)
_CENTRE_COLOR = dict(zip(CENTRES, ("tab:blue", "tab:orange", "tab:green"), strict=False))
_MODEL_MARKER = {"minimal_cox": "o", "spline_cox": "s"}
_CLASSIFICATION_COLOR = {
    CONCORDANCE_CLASSIFICATION_CONCORDANT: "tab:green",
    CONCORDANCE_CLASSIFICATION_DISCORDANT: "tab:red",
}


def _num(value: object) -> float | None:
    """`None`/`""`/NaN all mean "not reported" across these frames -- one predicate for all three,
    normalizing the wide-then-CSV-round-tripped-then-in-memory value types this module reads."""
    if value is None or value == "":
        return None
    value = float(value)
    return None if np.isnan(value) else value


def _reported_mask(status: object) -> bool:
    return status in ("estimable", "estimable_imprecise")


# ---------------------------------------------------------------------------
# 1. Sequencing-lag sensitivity -- headline figure
# ---------------------------------------------------------------------------


def render_sequencing_lag_headline(pooled_minimal_cox: pd.DataFrame, path: Path) -> None:
    """Pooled `minimal_cox` concordance (point + 95% CI) at each locked threshold, one panel per
    endpoint, one line per centre where the primary-evaluation reporting-status gate is met;
    retained N/events
    annotated beneath each point. Restricted to `concordance` (the headline discrimination metric)
    and to `minimal_cox`, per `sensitivity-analyses-spec.md` item 1 -- the by-FOLFOX stratum is a
    support diagnostic, not a performance comparison, so a centre missing at that threshold is
    simply absent from the line rather than interpolated."""
    if pooled_minimal_cox.empty:
        return
    concordance = pooled_minimal_cox[pooled_minimal_cox["metric"] == "concordance"]
    endpoints = [e for e in ("OS", "TTNT-D") if e in set(concordance["endpoint"])]
    if not endpoints:
        return

    fig, axes = plt.subplots(1, len(endpoints), figsize=(6.5 * len(endpoints), 4.5), squeeze=False)
    for ax, endpoint in zip(axes[0], endpoints, strict=False):
        sub = concordance[concordance["endpoint"] == endpoint]
        for centre in CENTRES:
            rows = sub[sub["centre"] == centre].set_index("threshold")
            xs, ys, lo, hi, notes = [], [], [], [], []
            for x, label in enumerate(_THRESHOLD_ORDER):
                if label not in rows.index:
                    continue
                row = rows.loc[label]
                estimate = _num(row["estimate"])
                if not _reported_mask(row["reporting_status"]) or estimate is None:
                    continue
                ci_lower, ci_upper = _num(row["ci_lower"]), _num(row["ci_upper"])
                xs.append(x)
                ys.append(estimate)
                lo.append(ci_lower if ci_lower is not None else estimate)
                hi.append(ci_upper if ci_upper is not None else estimate)
                notes.append(f"n={int(row['n_eligible'])}\nevt={int(row['n_events'])}")
            if not xs:
                continue
            yerr = np.array([np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)])
            ax.errorbar(
                xs, ys, yerr=yerr, marker="o", capsize=3, label=centre, color=_CENTRE_COLOR[centre]
            )
            for x, y, note in zip(xs, ys, notes, strict=False):
                ax.annotate(
                    note,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -24),
                    fontsize=6,
                    ha="center",
                    color=_CENTRE_COLOR[centre],
                )
        ax.set_xticks(range(len(_THRESHOLD_ORDER)))
        ax.set_xticklabels(_THRESHOLD_ORDER, rotation=20, fontsize=7)
        ax.set_ylabel("pooled minimal_cox concordance", fontsize=8)
        ax.set_title(endpoint, fontsize=9)
        ax.legend(fontsize=6)
        ax.margins(y=0.25)
    fig.suptitle(
        "Sequencing-lag sensitivity (headline) -- pooled minimal_cox concordance by threshold",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Truncation diagnostic
# ---------------------------------------------------------------------------


def render_truncation_diagnostic(
    entry_time_distribution: pd.DataFrame,
    km_by_entry_quartile: pd.DataFrame,
    lag_km_curve: pd.DataFrame,
    truncation_kendall_tau: pd.DataFrame,
    path: Path,
) -> None:
    """Row 1: entry-time distribution by institution (a box plot built from the persisted
    percentiles, per `sensitivity-analyses-spec.md` item 2's frozen "entry-time distribution plot"
    decision). Row 2: delayed-entry survival by entry-time quartile (`km_by_entry_quartile_rows`) --
    the frozen decision's own "KM curves stratified by entry-time quartile" requirement. Row 3:
    delayed-entry survival by locked sequencing-lag group, at the frozen `IBS_GRID` resolution
    (`lag_km_curve_rows`) -- an additional view requested alongside the frozen one, not a
    replacement for it. Each panel is annotated with the pooled conditional Kendall's tau and its
    comparable-pair count -- the numeric companion to every plot here."""
    if entry_time_distribution.empty and km_by_entry_quartile.empty and lag_km_curve.empty:
        return
    endpoints = sorted(
        set(entry_time_distribution.get("endpoint", pd.Series(dtype=str)))
        | set(km_by_entry_quartile.get("endpoint", pd.Series(dtype=str)))
        | set(lag_km_curve.get("endpoint", pd.Series(dtype=str)))
    )
    if not endpoints:
        return

    fig, axes = plt.subplots(3, len(endpoints), figsize=(6 * len(endpoints), 12), squeeze=False)
    for col, endpoint in enumerate(endpoints):
        ax_box, ax_quartile, ax_curve = axes[0][col], axes[1][col], axes[2][col]

        dist = entry_time_distribution[
            (entry_time_distribution.get("endpoint") == endpoint)
            & (entry_time_distribution["group"].isin(CENTRES))
        ]
        stats = []
        for centre in CENTRES:
            rows = dist[dist["group"] == centre]
            if rows.empty:
                continue
            row = rows.iloc[0]
            p = {p: _num(row.get(f"entry_p{p}")) for p in (0, 25, 50, 75, 100)}
            if any(v is None for v in p.values()):
                continue
            stats.append(
                {
                    "label": centre,
                    "med": p[50],
                    "q1": p[25],
                    "q3": p[75],
                    "whislo": p[0],
                    "whishi": p[100],
                }
            )
        if stats:
            ax_box.bxp(stats, showfliers=False)
            for tick, box in zip(ax_box.get_xticklabels(), stats, strict=False):
                tick.set_color(_CENTRE_COLOR.get(box["label"], "black"))
        ax_box.set_ylabel("entry (days)", fontsize=8)
        ax_box.set_title(f"{endpoint}: entry-time distribution by institution", fontsize=8)

        quartile = km_by_entry_quartile[
            (km_by_entry_quartile.get("endpoint") == endpoint)
            & (km_by_entry_quartile.get("status") == "ok")
        ].sort_values("entry_quartile")
        if not quartile.empty:
            ax_quartile.plot(
                quartile["entry_quartile"], quartile["survival_365"], marker="o", label="365d"
            )
            ax_quartile.plot(
                quartile["entry_quartile"], quartile["survival_730"], marker="s", label="730d"
            )
            ax_quartile.legend(fontsize=6)
        ax_quartile.set_xlabel("entry-time quartile", fontsize=8)
        ax_quartile.set_ylabel("delayed-entry KM survival", fontsize=8)
        ax_quartile.set_title(f"{endpoint}: survival by entry-time quartile", fontsize=8)

        curve = lag_km_curve[lag_km_curve.get("endpoint") == endpoint]
        for label in _THRESHOLD_ORDER:
            series = curve[curve["threshold"] == label].sort_values("time_days")
            if series.empty:
                continue
            ax_curve.step(
                series["time_days"], series["survival"], where="post", label=label, linewidth=1.4
            )
        ax_curve.set_xlabel("days since FOLFOX initiation", fontsize=8)
        ax_curve.set_ylabel("delayed-entry KM survival", fontsize=8)
        ax_curve.set_title(f"{endpoint}: survival by sequencing-lag group", fontsize=8)
        ax_curve.legend(fontsize=6)

        tau_rows = truncation_kendall_tau[
            (truncation_kendall_tau.get("endpoint") == endpoint)
            & (truncation_kendall_tau.get("stratum") == "pooled")
        ]
        if not tau_rows.empty:
            tau_row = tau_rows.iloc[0]
            tau_value = _num(tau_row["tau"])
            note = (
                f"pooled tau = {tau_value:.3f}" if tau_value is not None else "pooled tau: n/a"
            ) + f"\ncomparable pairs = {int(tau_row['n_comparable_pairs'])}"
            ax_curve.annotate(
                note,
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                fontsize=6,
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Complete-case sensitivity forest plot
# ---------------------------------------------------------------------------


def _forest_panel(ax, rows: pd.DataFrame, row_labels: list[str]) -> None:
    ax.axvline(0.0, color="black", linewidth=0.8)
    y_positions = np.arange(len(row_labels))
    for y, label in zip(y_positions, row_labels, strict=False):
        match = rows[rows["_label"] == label]
        if match.empty:
            continue
        row = match.iloc[0]
        diff = _num(row["diff_estimate"])
        if diff is None:
            continue
        lo, hi = _num(row["diff_ci_lower"]), _num(row["diff_ci_upper"])
        color = _CENTRE_COLOR.get(row["centre"], "black")
        marker = _MODEL_MARKER.get(row["model"], "d")
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.2)
        ax.plot([diff], [y], marker=marker, color=color, markersize=6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("complete-case minus full-cohort", fontsize=8)


def render_complete_case_forest(
    complete_case: pd.DataFrame, primary_path: Path, supplementary_path: Path
) -> None:
    """`sensitivity-analyses-spec.md` item 3's forest plot of `complete-case - full` with paired
    bootstrap 95% CIs (see `complete_case.py`'s `_paired_difference`). Primary figure: 730-day OS,
    `minimal_cox`/`spline_cox`, pooled design, across the three centres, for `concordance` and
    `brier_730` (the headline discrimination and calibration-adjacent metrics). Supplementary: the
    365-day OS and 730-day TTNT-D concordance analogues, deferring every other
    metric/horizon/design/endpoint combination to the CSV -- a disclosed scope restriction, not an
    omission (see `implement.md`)."""
    if complete_case.empty:
        return
    frame = complete_case.copy()
    frame["_label"] = frame["centre"] + " / " + frame["model"]

    def _select(endpoint: str, metric: str, design: str = "pooled") -> pd.DataFrame:
        return frame[
            (frame["endpoint"] == endpoint)
            & (frame["metric"] == metric)
            & (frame["design"] == design)
        ]

    row_labels = [f"{c} / {m}" for c in CENTRES for m in ("minimal_cox", "spline_cox")]

    primary_metrics = [("concordance", "concordance"), ("brier_730", "Brier (730d)")]
    available_primary = [(k, t) for k, t in primary_metrics if not _select("OS", k).empty]
    if available_primary:
        fig, axes = plt.subplots(
            1, len(available_primary), figsize=(6 * len(available_primary), 4.5), squeeze=False
        )
        for ax, (metric_key, title) in zip(axes[0], available_primary, strict=False):
            _forest_panel(ax, _select("OS", metric_key), row_labels)
            ax.set_title(f"OS, 730d pooled: {title}", fontsize=9)
        fig.suptitle("Complete-case assessment-subset sensitivity (primary)", fontsize=9)
        fig.tight_layout()
        fig.savefig(primary_path, dpi=110)
        plt.close(fig)

    supplementary_specs = [
        ("OS", "brier_365", "OS, 365d pooled, Brier (supplementary horizon)"),
        ("TTNT-D", "concordance", "TTNT-D, pooled, concordance (supportive endpoint)"),
    ]
    available_supplementary = [
        (ep, mk, title) for ep, mk, title in supplementary_specs if not _select(ep, mk).empty
    ]
    if available_supplementary:
        fig, axes = plt.subplots(
            1,
            len(available_supplementary),
            figsize=(6 * len(available_supplementary), 4.5),
            squeeze=False,
        )
        for ax, (endpoint, metric_key, title) in zip(
            axes[0], available_supplementary, strict=False
        ):
            _forest_panel(ax, _select(endpoint, metric_key), row_labels)
            ax.set_title(title, fontsize=9)
        fig.suptitle("Complete-case assessment-subset sensitivity (supplementary)", fontsize=9)
        fig.tight_layout()
        fig.savefig(supplementary_path, dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Supportive endpoint robustness comparison
# ---------------------------------------------------------------------------


def render_endpoint_robustness(endpoint_robustness: pd.DataFrame, path: Path) -> None:
    """Side-by-side OS-versus-TTNT-D effect bars for every classified comparison, coloured by
    concordant/discordant/indeterminate. TTNT-D bars are drawn hatched, and the panel is titled with
    the word "supportive", so the endpoint's subordinate role stays visible in the image itself, not
    only in surrounding prose (`mission.md` non-negotiable 1). Restricted to the `concordance`
    metric for the `complete_case` sensitivity (every `truncation_kendall_tau` row is kept, since
    there are only four) -- the real published frame carries 10 metrics x 2 designs x 3 centres x 2
    models of complete-case comparisons, which would make a single one-row-per-comparison figure
    several dozen inches tall; every comparison, on every metric, remains in
    `supportive_endpoint_robustness.csv` (a disclosed scope restriction, not an omission)."""
    if endpoint_robustness.empty:
        return
    frame = endpoint_robustness[
        (endpoint_robustness["sensitivity"] != "complete_case")
        | (endpoint_robustness["metric"] == "concordance")
    ].copy()
    frame["_label"] = (
        frame["sensitivity"]
        + " | "
        + frame["centre"]
        + " | "
        + frame["model"]
        + " | "
        + frame["metric"]
    )
    frame = frame[frame["_label"].str.len() > 0]
    if frame.empty:
        return

    y_positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(frame) + 1.0)))
    for y, (_, row) in zip(y_positions, frame.iterrows(), strict=False):
        color = _CLASSIFICATION_COLOR.get(row["concordance_classification"], "tab:gray")
        os_effect, ttnt_effect = _num(row["os_effect"]), _num(row["ttnt_effect"])
        if os_effect is not None:
            ax.barh(y + 0.18, os_effect, height=0.32, color=color, alpha=0.9, label="_nolegend_")
        if ttnt_effect is not None:
            ax.barh(
                y - 0.18,
                ttnt_effect,
                height=0.32,
                color=color,
                alpha=0.55,
                hatch="//",
                label="_nolegend_",
            )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["_label"], fontsize=6)
    ax.set_xlabel("effect (complete-case diff, or tau)", fontsize=8)
    ax.set_title(
        "Supportive endpoint robustness -- OS (solid) vs. TTNT-D (hatched, supportive-only)",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_CLASSIFICATION_COLOR[key], label=key)
        for key in (CONCORDANCE_CLASSIFICATION_CONCORDANT, CONCORDANCE_CLASSIFICATION_DISCORDANT)
    ] + [plt.Rectangle((0, 0), 1, 1, color="tab:gray", label="indeterminate")]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Calibration before/after local adaptation
# ---------------------------------------------------------------------------


def _recalibration_curve(risk_grid: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(risk_grid, RISK_CLIP, 1.0 - RISK_CLIP)
    cloglog = np.log(-np.log(1.0 - clipped))
    return 1.0 - np.exp(-np.exp(intercept + slope * cloglog))


def render_calibration_drift(
    calibration_drift: pd.DataFrame, path: Path, *, endpoint: str = "OS", model: str = "minimal_cox"
) -> None:
    """`sensitivity-analyses-spec.md` item 5's four curves (untouched, intercept-only,
    intercept-plus-slope, ideal 45-degree) reconstructed from each step's own fitted
    `(citl, slope)` calibration-GLM parameters, applied to a risk grid -- the calibration line each
    step's `horizon_calibration` fit implies, not a rebinning of patient-level predictions
    (sensitivity-analysis workflow
    persists no patient-level artifact). Scoped to `endpoint`/`model` (default OS/minimal_cox, the
    primary headline pair) across the three centres and both horizons; a step whose fit failed (e.g.
    VICC's support gate) is a labelled "not computable" panel, never a fabricated line."""
    if calibration_drift.empty:
        return
    frame = calibration_drift[
        (calibration_drift["endpoint"] == endpoint) & (calibration_drift["model"] == model)
    ]
    if frame.empty:
        return
    horizons = sorted(set(frame["horizon"]))
    grid = np.linspace(0.01, 0.95, 50)
    ideal = grid

    fig, axes = plt.subplots(
        len(horizons),
        len(CENTRES),
        figsize=(4.2 * len(CENTRES), 4.0 * len(horizons)),
        squeeze=False,
    )
    for row, horizon in enumerate(horizons):
        for col, centre in enumerate(CENTRES):
            ax = axes[row][col]
            cell = frame[(frame["centre"] == centre) & (frame["horizon"] == horizon)]
            ax.plot(grid, ideal, color="black", linestyle="--", linewidth=1.0, label="ideal")
            drew_any = False
            for step, curve_label, color in (
                (STEP_UNTOUCHED, "untouched", "tab:gray"),
                (STEP_INTERCEPT_ONLY, "intercept-only", "tab:blue"),
                (STEP_INTERCEPT_SLOPE, "intercept+slope", "tab:red"),
            ):
                step_rows = cell[cell["step"] == step]
                if step_rows.empty:
                    continue
                step_row = step_rows.iloc[0]
                citl, slope = _num(step_row["citl_after"]), 1.0
                if step == STEP_INTERCEPT_SLOPE:
                    slope = _num(step_row["slope_after"]) or 1.0
                if citl is None:
                    continue
                ax.plot(
                    grid, _recalibration_curve(grid, citl, slope), color=color, label=curve_label
                )
                drew_any = True
            if not drew_any:
                ax.annotate(
                    "not computable\n(support gate)",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=7,
                )
            ax.set_title(f"{centre}, {horizon:g}d", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
    fig.suptitle(f"Calibration before/after local adaptation ({endpoint}, {model})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Scaled Brier / no-information reference
# ---------------------------------------------------------------------------


def render_scaled_brier(
    scaled_brier: pd.DataFrame, path: Path, *, model: str = "minimal_cox", design: str = "pooled"
) -> None:
    """Model Brier/IBS against the delayed-entry KM no-information reference, and the scaled Brier
    score, with a zero reference line -- explicitly labelled a descriptive aid, never a new primary
    metric, per `sensitivity-analyses-spec.md` item 6. Scoped to pooled `minimal_cox` (the headline
    design/model pair); every other cell remains in `scaled_brier.csv`."""
    if scaled_brier.empty:
        return
    frame = scaled_brier[(scaled_brier["model"] == model) & (scaled_brier["design"] == design)]
    if frame.empty:
        return
    endpoints = [e for e in ("OS", "TTNT-D") if e in set(frame["endpoint"])]
    if not endpoints:
        return
    metrics = ["brier_365", "brier_730", "ibs"]

    fig, axes = plt.subplots(1, len(endpoints), figsize=(6 * len(endpoints), 4.2), squeeze=False)
    for ax, endpoint in zip(axes[0], endpoints, strict=False):
        sub = frame[frame["endpoint"] == endpoint]
        width = 0.8 / max(len(CENTRES), 1)
        x_base = np.arange(len(metrics))
        for i, centre in enumerate(CENTRES):
            values = []
            for metric_key in metrics:
                rows = sub[(sub["centre"] == centre) & (sub["metric"] == metric_key)]
                values.append(_num(rows.iloc[0]["scaled_brier"]) if not rows.empty else None)
            xs = x_base + (i - (len(CENTRES) - 1) / 2) * width
            ys = [v if v is not None else 0.0 for v in values]
            colors = [_CENTRE_COLOR[centre] if v is not None else "none" for v in values]
            ax.bar(xs, ys, width=width, color=colors, label=centre, edgecolor=_CENTRE_COLOR[centre])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x_base)
        ax.set_xticklabels(metrics, fontsize=7)
        ax.set_ylabel("scaled Brier (1 - Brier/Brier_KM)", fontsize=8)
        ax.set_title(f"{endpoint} -- descriptive aid, not a primary metric", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Local-adaptation learning curve
# ---------------------------------------------------------------------------


def render_local_adaptation_curve(
    local_recalibration: pd.DataFrame,
    calibration_drift: pd.DataFrame,
    path: Path,
    *,
    endpoint: str = "OS",
    model: str = "minimal_cox",
    horizon: float = 730.0,
) -> None:
    """0%/10%/20%/30% stratified local-adaptation fractions against post-adaptation `|CITL|` on
    the held-out evaluation subset, with the unused-evaluation N/events annotated beneath each
    point. The 0% point is the untouched step from `calibration_drift` (no adaptation subset
    withheld yet); a centre with no feasible fraction (VICC's support gate failing at every
    fraction) is skipped
    entirely, never drawn with a fabricated point (`sensitivity-analyses-spec.md` item 5)."""
    if local_recalibration.empty:
        return
    local = local_recalibration[
        (local_recalibration["endpoint"] == endpoint)
        & (local_recalibration["model"] == model)
        & (local_recalibration["horizon"] == horizon)
    ]
    untouched = calibration_drift[
        (calibration_drift["endpoint"] == endpoint)
        & (calibration_drift["model"] == model)
        & (calibration_drift["horizon"] == horizon)
        & (calibration_drift["step"] == STEP_UNTOUCHED)
    ]
    if local.empty and untouched.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fractions_pct = [0] + [round(f * 100) for f in LOCAL_ADAPTATION_FRACTIONS]
    any_centre_drawn = False
    for centre in CENTRES:
        xs, ys, notes = [], [], []
        base = untouched[untouched["centre"] == centre]
        if not base.empty:
            citl0 = _num(base.iloc[0]["citl_before"])
            if citl0 is not None:
                xs.append(0)
                ys.append(abs(citl0))
                notes.append(
                    f"n={int(base.iloc[0]['n_eligible'])}\nevt={int(base.iloc[0]['n_events'])}"
                )
        centre_rows = local[local["centre"] == centre].set_index("fraction")
        for fraction in LOCAL_ADAPTATION_FRACTIONS:
            if fraction not in centre_rows.index:
                continue
            row = centre_rows.loc[fraction]
            if not bool(row["feasible"]):
                continue
            citl_after = _num(row["citl_after"])
            if citl_after is None:
                continue
            xs.append(round(fraction * 100))
            ys.append(abs(citl_after))
            notes.append(f"n={int(row['n_evaluation'])}\nevt={int(row['n_evaluation_events'])}")
        if len(xs) < 2:
            continue
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        notes = [notes[i] for i in order]
        ax.plot(xs, ys, marker="o", color=_CENTRE_COLOR[centre], label=centre)
        for x, y, note in zip(xs, ys, notes, strict=False):
            ax.annotate(
                note,
                (x, y),
                textcoords="offset points",
                xytext=(0, -26),
                fontsize=6,
                ha="center",
                color=_CENTRE_COLOR[centre],
            )
        any_centre_drawn = True
    if not any_centre_drawn:
        plt.close(fig)
        return
    ax.set_xticks(fractions_pct)
    ax.set_xticklabels([f"{p}%" for p in fractions_pct])
    ax.set_xlabel("local-adaptation fraction", fontsize=8)
    ax.set_ylabel("|CITL| on unused evaluation subset", fontsize=8)
    ax.set_title(f"Local-adaptation learning curve ({endpoint}, {model}, {horizon:g}d)", fontsize=9)
    ax.margins(y=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


__all__ = [
    "render_calibration_drift",
    "render_complete_case_forest",
    "render_endpoint_robustness",
    "render_local_adaptation_curve",
    "render_scaled_brier",
    "render_sequencing_lag_headline",
    "render_truncation_diagnostic",
]

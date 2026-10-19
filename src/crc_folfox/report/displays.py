"""Manuscript-reporting display items: the eight prespecified main figures and tables.

Every figure is regenerated at >= 300 dpi from its published aggregate frame -- never by upscaling
an existing PNG, because an upscaled PNG carries the resolution of the run that made it and
silently decouples the paper from the bundle it claims to report.

Every table renders the three reporting statuses as three distinct categories. Collapsing
`estimable`, `estimable_imprecise`, and `not_computable` into one bucket of bad results is the
specific misreading `the evaluation-diagnosis note` exists to prevent: an imprecise cell is a valid
estimate that must be shown with its interval, and a not-computable cell has a model-level cause.

Every display records the scope restriction it makes, the way sensitivity-analysis workflow's
`plots.py` does, so a
reader of the artifact can see what was left out rather than inferring it from the picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

from crc_folfox.preprocessing.contracts import INSTITUTIONS  # noqa: E402
from crc_folfox.report.contracts import (  # noqa: E402
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
    ReportContractError,
)
from crc_folfox.report.frames import (  # noqa: E402
    EXPECTED_ELIGIBLE,
    assert_descriptive_and_aggregate,
)
from crc_folfox.report.inputs import CanonicalInputs  # noqa: E402

FIGURE_DPI = 300
#: Figures are drawn at the same width used by the manuscript renderer.  This matters more than a
#: nominal font size: drawing at 12 inches and placing at 6.5 inches turns 9 pt source text into
#: roughly 5 pt print text.  At this width, the source font size is the final printed font size.
MANUSCRIPT_FIGURE_WIDTH_IN = 6.5
FIGURE_FONT_MIN_PT = 8.0
FIGURE_FONT_MAX_PT = 10.0
FIGURE_FONT_SCALE = 1.0
FIGURE_INK = "#262626"
FIGURE_GRID = "#b8b8b8"
PRIMARY_MODELS = ("minimal_cox", "spline_cox")
PRIMARY_HORIZON = 730.0

_STATUS_MARKER = {
    "estimable": "o",
    REPORTING_IMPRECISE: "s",
    REPORTING_NOT_COMPUTABLE: "x",
}


@dataclass
class DisplayRecord:
    """One rendered display and every scope restriction it applied."""

    name: str
    path: str
    source: str
    restrictions: list[str] = field(default_factory=list)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save at final manuscript size with print-safe type and strokes."""

    for text in fig.findobj(matplotlib.text.Text):
        scaled = text.get_fontsize() * FIGURE_FONT_SCALE
        text.set_fontsize(min(FIGURE_FONT_MAX_PT, max(FIGURE_FONT_MIN_PT, scaled)))
        if text.get_color() in {"#5f6368", "#616161", "#777777", "#9e9e9e"}:
            text.set_color(FIGURE_INK)
    for ax in fig.axes:
        ax.tick_params(
            axis="both",
            which="both",
            colors=FIGURE_INK,
            labelsize=FIGURE_FONT_MIN_PT,
            width=1.0,
        )
        for spine in ax.spines.values():
            spine.set_color(FIGURE_INK)
            spine.set_linewidth(max(spine.get_linewidth(), 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_panel_header(
    ax: plt.Axes,
    letter: str,
    text: str,
    *,
    fontsize: float = 8.5,
    pad: float = 7.0,
) -> None:
    """Add a bold panel letter separately from its descriptive header."""

    panel_label = f"{letter}.\n" if "\n" in text else f"{letter}."
    ax.set_title(
        panel_label,
        loc="left",
        fontsize=fontsize,
        fontweight="bold",
        color=FIGURE_INK,
        pad=pad,
    )
    ax.set_title(text, loc="center", fontsize=fontsize, color=FIGURE_INK, pad=pad)


# --- Figure 1: cohort flow -----------------------------------------------------------------------


def figure1_cohort_flow(flow: pd.DataFrame, output_dir: Path) -> DisplayRecord:
    """Draw one source-cohort path followed by two independent endpoint branches."""

    fig, ax = plt.subplots(figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 6.2))
    ax.axis("off")

    main = flow[flow["scope"].isin({"release", "analysis_cohort"})].reset_index(drop=True)
    x_main = 0.34
    y_positions = np.linspace(0.94, 0.47, len(main))
    previous_count: int | None = None
    exclusion_reasons = (
        "No documented advanced index CRC",
        "No linked post-metastatic regimen",
        "First post-metastatic regimen did not contain a FOLFOX backbone",
        "Did not meet the primary FOLFOX/adult definition",
    )

    for position, ((_, row), y) in enumerate(zip(main.iterrows(), y_positions, strict=False)):
        count = int(row["patients"])
        centres = [c for c in INSTITUTIONS if pd.notna(row.get(c))]
        detail = "\n" + " | ".join(f"{c} {int(row[c])}" for c in centres) if centres else ""
        is_primary = row["scope"] == "analysis_cohort"
        ax.text(
            x_main,
            y,
            f"{row['step']}: n = {int(row['patients'])}{detail}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=FIGURE_INK,
            fontweight="bold" if is_primary else "normal",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#e8f0fe" if is_primary else "#f5f5f5",
                "edgecolor": "#1a73e8" if is_primary else "#4a4a4a",
                "linewidth": 1.8 if is_primary else 1.2,
            },
        )
        if position:
            prior_y = y_positions[position - 1]
            ax.annotate(
                "",
                xy=(x_main, y + 0.035),
                xytext=(x_main, prior_y - 0.035),
                arrowprops={"arrowstyle": "->", "color": FIGURE_INK, "linewidth": 1.4},
            )
            excluded = int(previous_count - count) if previous_count is not None else 0
            midpoint = (prior_y + y) / 2
            ax.annotate(
                "",
                xy=(0.66, midpoint),
                xytext=(x_main + 0.03, midpoint),
                arrowprops={"arrowstyle": "->", "color": FIGURE_INK, "linewidth": 1.3},
            )
            ax.text(
                0.79,
                midpoint,
                f"Excluded: n = {excluded}\n{exclusion_reasons[position - 1]}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=FIGURE_INK,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#fff8e1",
                    "edgecolor": "#4a4a4a",
                    "linewidth": 1.2,
                },
            )
        previous_count = count

    primary_y = float(y_positions[-1])
    branch_y = 0.20
    branch_specs = (
        ("OS", 0.22, "deaths", "entry at or after OS exit"),
        ("TTNT-D", 0.70, "events", "entry at or after TTNT-D exit"),
    )
    primary_count = int(main.iloc[-1]["patients"])
    for endpoint, x, event_word, exclusion_reason in branch_specs:
        source = flow[flow["step"] == f"Source cohort, {endpoint} events"].iloc[0]
        eligible = flow[flow["step"] == f"Delayed-entry eligible, {endpoint}"].iloc[0]
        event_count = int(sum(int(source[c]) for c in INSTITUTIONS))
        eligible_n = int(eligible["patients"])
        excluded_n = primary_count - eligible_n
        centre_pairs = [
            f"{c} {int(eligible[c])}/{int(EXPECTED_ELIGIBLE[endpoint][c][1])}" for c in INSTITUTIONS
        ]
        centre_detail = (
            f"{centre_pairs[0]} | {centre_pairs[1]}\n{centre_pairs[2]} (patients/events)"
        )
        ax.annotate(
            "",
            xy=(x, branch_y + 0.07),
            xytext=(x_main, primary_y - 0.04),
            arrowprops={"arrowstyle": "->", "color": FIGURE_INK, "linewidth": 1.5},
        )
        ax.text(
            x,
            branch_y,
            f"{endpoint} analysis: n = {eligible_n}, {event_word} = "
            f"{EXPECTED_ELIGIBLE[endpoint]['events']}\n{centre_detail}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=FIGURE_INK,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#e8f0fe",
                "edgecolor": "#1a73e8",
                "linewidth": 1.8,
            },
        )
        ax.text(
            x,
            0.06,
            f"Source cohort: {event_count} {event_word}\nExcluded: n = {excluded_n}; "
            f"{exclusion_reason}",
            ha="center",
            va="center",
            fontsize=8.0,
            color=FIGURE_INK,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fff8e1",
                "edgecolor": "#4a4a4a",
                "linewidth": 1.2,
            },
        )

    fig.tight_layout(pad=0.35)
    path = output_dir / "figure1_cohort_flow.png"
    _save(fig, path)
    return DisplayRecord(
        name="Figure 1",
        path=str(path),
        source="manuscript/cohort_flow.csv (computed, reconciled to tech-stack.md section 5)",
        restrictions=[
            "release-level steps are frozen constants, not recomputed from the 678-row dataset",
            "OS and TTNT-D populations branch independently from the 678-patient primary cohort",
            "every transition states its excluded count and reason",
            "the endpoint populations branch without panel lettering; each branch box names its "
            "endpoint directly",
        ],
    )


# --- Figure 2: delayed-entry-adjusted survival ----------------------------------------------------


def build_delayed_entry_km_frame(inputs: CanonicalInputs) -> pd.DataFrame:
    """Compute descriptive left-truncation-aware OS curves without exposing patient rows."""

    dataset = inputs.frame("phase2_dataset", "analysis_dataset.csv")
    eligible = dataset[dataset["os_eligible"].astype(str).str.lower().eq("true")].copy()
    rows: list[dict] = []
    strata = {
        "presentation": (
            "metastatic_presentation",
            ("De novo Stage IV", "Recurrent/metachronous"),
        ),
        "centre": ("institution", INSTITUTIONS),
    }
    for stratification, (column, levels) in strata.items():
        for level in levels:
            cell = eligible[eligible[column].astype(str) == level]
            if cell.empty:
                continue
            km = KaplanMeierFitter(alpha=0.05).fit(
                cell["os_time"].to_numpy(float),
                event_observed=cell["os_event"].to_numpy(int),
                entry=cell["entry"].to_numpy(float),
                label=str(level),
            )
            survival = km.survival_function_.iloc[:, 0]
            interval = km.confidence_interval_
            for time, estimate in survival.items():
                rows.append(
                    {
                        "stratification": stratification,
                        "group": str(level),
                        "time": float(time),
                        "survival": float(estimate),
                        "ci_lower": float(interval.loc[time].iloc[0]),
                        "ci_upper": float(interval.loc[time].iloc[1]),
                        "n_patients": int(len(cell)),
                        "n_events": int(cell["os_event"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def figure2_survival(km_frame: pd.DataFrame, output_dir: Path) -> DisplayRecord:
    """Display absolute OS by presentation and centre using delayed-entry risk sets."""

    if km_frame.empty:
        raise ReportContractError("Figure 2 has no delayed-entry OS survival rows")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 3.25),
        sharex=True,
        sharey=True,
    )
    colours = {
        "De novo Stage IV": "#d93025",
        "Recurrent/metachronous": "#1a73e8",
        "DFCI": "#1a73e8",
        "MSK": "#188038",
        "VICC": "#d93025",
    }
    for ax, (stratification, title) in zip(
        axes,
        (("presentation", "A. By metastatic presentation"), ("centre", "B. By centre")),
        strict=False,
    ):
        panel = km_frame[km_frame["stratification"] == stratification]
        for group in panel["group"].drop_duplicates():
            cell = panel[panel["group"] == group].sort_values("time")
            support = cell.iloc[0]
            label = f"{group}\n(n={int(support['n_patients'])}, events={int(support['n_events'])})"
            colour = colours.get(str(group), "#5f6368")
            ax.step(
                cell["time"],
                cell["survival"],
                where="post",
                color=colour,
                lw=2.2,
                label=label,
            )
            ax.fill_between(
                cell["time"].to_numpy(float),
                cell["ci_lower"].to_numpy(float),
                cell["ci_upper"].to_numpy(float),
                step="post",
                color=colour,
                alpha=0.18,
                linewidth=0.6,
            )
        panel_letter, panel_title = title.split(". ", maxsplit=1)
        _set_panel_header(ax, panel_letter, panel_title, fontsize=9.0)
        ax.set_xlim(0, 1460)
        ax.set_ylim(0, 1.02)
        ax.set_xticks((0, 365, 730, 1095, 1460))
        ax.grid(color=FIGURE_GRID, alpha=0.65, linewidth=0.8)
        ax.legend(fontsize=8.0, loc="lower left", frameon=False)
        ax.set_xlabel("Days from FOLFOX initiation", fontsize=9.0)
    axes[0].set_ylabel("Overall survival probability", fontsize=9.0)
    fig.tight_layout()
    path = output_dir / "figure2_delayed_entry_survival.png"
    _save(fig, path)
    return DisplayRecord(
        name="Figure 2",
        path=str(path),
        source="analysis_dataset.csv; lifelines KaplanMeierFitter(entry=...) aggregate curve",
        restrictions=[
            "descriptive OS curves only; no causal or subgroup-effect interpretation",
            "risk sets incorporate delayed entry from FOLFOX initiation",
            "curves are truncated at four years to avoid visually emphasizing sparse tails",
        ],
    )


# --- Figure 3: calibration, held-out OS at 730 days -----------------------------------------------


def figure3_calibration(inputs: CanonicalInputs, output_dir: Path) -> DisplayRecord:
    """Held-out OS calibration at 730 days: three centres, both risk scales, no recalibration."""

    curves = inputs.frame("phase5_evaluation", "calibration_curves.csv")
    performance = inputs.frame("phase5_evaluation", "performance_summary.csv")

    subset = curves[
        (curves["endpoint"] == "OS")
        & (curves["design"] == "iecv")
        & (curves["horizon"] == PRIMARY_HORIZON)
        & (curves["model"] == "minimal_cox")
    ]
    if subset.empty:
        raise ReportContractError("Figure 3 has no held-out OS calibration rows at 730 days")

    scales = ("unconditional", "entry_conditional")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 5.1),
        sharex=True,
        sharey=True,
    )

    legend_handles: dict[str, object] = {}
    for row_index, scale in enumerate(scales):
        for col_index, centre in enumerate(INSTITUTIONS):
            ax = axes[row_index][col_index]
            panel_letter = chr(ord("A") + row_index * len(INSTITUTIONS) + col_index)
            cell = subset[(subset["centre"] == centre) & (subset["risk_scale"] == scale)]
            ideal_line = ax.plot(
                [0, 0.8],
                [0, 0.8],
                linestyle="--",
                color="#3f3f3f",
                linewidth=1.6,
                zorder=1,
                label="Ideal",
            )[0]
            legend_handles.setdefault("Ideal", ideal_line)
            if not cell.empty:
                low = cell["low_support"].astype(str).str.lower().eq("true")
                ax.errorbar(
                    cell["mean_predicted_risk"],
                    cell["observed_risk"],
                    yerr=cell["observed_risk_se"],
                    fmt="o-",
                    color="#1a73e8",
                    ecolor="#376d9f",
                    capsize=3.5,
                    capthick=1.5,
                    elinewidth=1.5,
                    linewidth=2.0,
                    markersize=6.2,
                    markeredgewidth=1.2,
                    zorder=3,
                )
                if low.any():
                    low_support_marker = ax.scatter(
                        cell.loc[low, "mean_predicted_risk"],
                        cell.loc[low, "observed_risk"],
                        facecolors="none",
                        edgecolors="#d93025",
                        s=110,
                        linewidths=2.0,
                        zorder=4,
                        label="low support",
                    )
                    legend_handles.setdefault("Low support", low_support_marker)
                # The frozen curve artifact contains quintile counts rather than patient-level
                # predictions. Thin bars show that distribution without leaking row-level risks.
                max_n = max(float(cell["n"].max()), 1.0)
                ax.bar(
                    cell["mean_predicted_risk"],
                    0.065 * cell["n"].astype(float) / max_n,
                    width=0.018,
                    color="#595959",
                    edgecolor="#262626",
                    linewidth=0.8,
                    alpha=0.48,
                    zorder=0,
                )

            slope = performance[
                (performance["endpoint"] == "OS")
                & (performance["design"] == "iecv")
                & (performance["centre"] == centre)
                & (performance["model"] == "minimal_cox")
                & (performance["metric"] == "slope")
                & (performance["horizon"] == PRIMARY_HORIZON)
                & (performance["risk_scale"] == scale)
            ]
            slope_text = "Slope not computable"
            if not slope.empty:
                srow = slope.iloc[0]
                status = _status_label(str(srow["reporting_status"]))
                slope_text = (
                    f"Slope {srow['estimate']:.3f}\n"
                    f"[{srow['ci_lower']:.3f}, {srow['ci_upper']:.3f}]\n{status}"
                )
            panel_title = f"{centre}\n{scale.replace('_', '-')} risk"
            _set_panel_header(ax, panel_letter, panel_title, fontsize=8.0, pad=5.0)
            ax.text(
                0.04,
                0.95,
                slope_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=FIGURE_INK,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#3f3f3f",
                    "linewidth": 1.0,
                    "alpha": 0.92,
                },
                zorder=6,
            )
            ax.set_xlim(0, 0.8)
            ax.set_ylim(0, 0.8)
            ax.grid(color=FIGURE_GRID, alpha=0.55, linewidth=0.8)
            if col_index == 0:
                ax.set_ylabel("Observed risk", fontsize=9.0)
            if row_index == 1:
                ax.set_xlabel("Mean predicted risk", fontsize=9.0)

    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1.4, w_pad=0.8)
    path = output_dir / "figure3_calibration_os_730.png"
    _save(fig, path)
    return DisplayRecord(
        name="Figure 3",
        path=str(path),
        source="evaluation/calibration_curves.csv + performance_summary.csv",
        restrictions=[
            "minimal_cox only; spline_cox calibration goes to the supplement",
            "held-out (iecv) design only; pooled is the within-mixture comparator",
            "both risk scales shown; neither is presented as the single calibration answer",
            "panels A-C show DFCI, MSK, and VICC on the unconditional risk scale; panels D-F "
            "show the same centres on the entry-conditional risk scale",
            "each panel carries its slope interval and reporting status",
            "bottom bars show the number of observations in each published prediction quintile",
        ],
    )


def supplementary_calibration(
    inputs: CanonicalInputs,
    output_dir: Path,
    *,
    endpoint: str,
    model: str,
    item: str,
) -> DisplayRecord:
    """Render an S-numbered 730-day calibration grid with published support counts."""

    curves = inputs.frame("phase5_evaluation", "calibration_curves.csv")
    performance = inputs.frame("phase5_evaluation", "performance_summary.csv")
    subset = curves[
        (curves["endpoint"] == endpoint)
        & (curves["design"] == "iecv")
        & (curves["horizon"] == PRIMARY_HORIZON)
        & (curves["model"] == model)
    ]
    if subset.empty:
        raise ReportContractError(f"{item} has no {endpoint}/{model} calibration rows")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 5.1),
        sharex=True,
        sharey=True,
    )
    for row_index, scale in enumerate(("unconditional", "entry_conditional")):
        for col_index, centre in enumerate(INSTITUTIONS):
            ax = axes[row_index][col_index]
            panel_letter = chr(ord("A") + row_index * len(INSTITUTIONS) + col_index)
            cell = subset[(subset["centre"] == centre) & (subset["risk_scale"] == scale)]
            ax.plot([0, 0.8], [0, 0.8], "--", color="#3f3f3f", linewidth=1.6)
            if not cell.empty:
                ax.errorbar(
                    cell["mean_predicted_risk"],
                    cell["observed_risk"],
                    yerr=cell["observed_risk_se"],
                    fmt="o-",
                    color="#1a73e8",
                    ecolor="#376d9f",
                    capsize=3.5,
                    capthick=1.5,
                    elinewidth=1.5,
                    linewidth=2.0,
                    markersize=6.2,
                    markeredgewidth=1.2,
                )
                max_n = max(float(cell["n"].max()), 1.0)
                ax.bar(
                    cell["mean_predicted_risk"],
                    0.065 * cell["n"].astype(float) / max_n,
                    width=0.018,
                    color="#595959",
                    edgecolor="#262626",
                    linewidth=0.8,
                    alpha=0.48,
                )
            slope = performance[
                (performance["endpoint"] == endpoint)
                & (performance["design"] == "iecv")
                & (performance["centre"] == centre)
                & (performance["model"] == model)
                & (performance["metric"] == "slope")
                & (performance["horizon"] == PRIMARY_HORIZON)
                & (performance["risk_scale"] == scale)
            ]
            slope_text = "Slope not computable"
            if not slope.empty:
                srow = slope.iloc[0]
                slope_text = (
                    f"Slope {srow['estimate']:.3f}\n"
                    f"[{srow['ci_lower']:.3f}, {srow['ci_upper']:.3f}]\n"
                    f"{_status_label(srow['reporting_status'])}"
                )
            panel_title = f"{centre}\n{scale.replace('_', '-')} risk"
            _set_panel_header(ax, panel_letter, panel_title, fontsize=8.0, pad=5.0)
            ax.text(
                0.04,
                0.95,
                slope_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=FIGURE_INK,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#3f3f3f",
                    "linewidth": 1.0,
                    "alpha": 0.92,
                },
                zorder=6,
            )
            ax.set_xlim(0, 0.8)
            ax.set_ylim(0, 0.8)
            ax.grid(color=FIGURE_GRID, alpha=0.55, linewidth=0.8)
            if col_index == 0:
                ax.set_ylabel("Observed risk", fontsize=9.0)
            if row_index == 1:
                ax.set_xlabel("Mean predicted risk", fontsize=9.0)
    fig.tight_layout(h_pad=1.4, w_pad=0.8)
    safe_endpoint = endpoint.lower().replace("-", "_")
    path = output_dir / f"{item.lower().replace(' ', '_')}_{safe_endpoint}_{model}.png"
    _save(fig, path)
    return DisplayRecord(
        name=item,
        path=str(path),
        source="evaluation/calibration_curves.csv + performance_summary.csv",
        restrictions=[
            "held-out IECV estimates at 730 days before recalibration",
            "both unconditional and entry-conditional risk scales shown",
            "bottom bars show published prediction-quintile support",
        ],
    )


# --- Supplementary attribution figure ------------------------------------------------------------


def supplementary_attribution(inputs: CanonicalInputs, output_dir: Path) -> DisplayRecord:
    """Panel A: grouped importance. Panel B: cross-centre / cross-endpoint agreement."""

    importance = inputs.frame("phase6_explainability", "importance_summary.csv")
    stability = inputs.frame("phase6_explainability", "attribution_stability.csv")

    panel_a = importance[
        (importance["endpoint"] == "OS")
        & (importance["model"] == "minimal_cox")
        & (importance["metric"].astype(str) == "730")
    ].copy()
    if panel_a.empty:
        raise ReportContractError("Supplementary attribution panel has no 730-day importance")

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 3.7),
    )

    centres = [c for c in INSTITUTIONS if c in set(panel_a["centre"].astype(str))]
    groups = sorted(panel_a["feature_group"].astype(str).unique())
    offsets = {centre: (i - (len(centres) - 1) / 2) * 0.22 for i, centre in enumerate(centres)}
    colours = {"DFCI": "#1a73e8", "MSK": "#188038", "VICC": "#d93025"}

    for centre in centres:
        cell = panel_a[panel_a["centre"].astype(str) == centre]
        ys, xs = [], []
        for group in groups:
            match = cell[cell["feature_group"].astype(str) == group]
            if match.empty:
                continue
            row = match.iloc[0]
            ys.append(groups.index(group) + offsets[centre])
            xs.append(float(row["importance"]))
        ax_a.scatter(
            xs,
            ys,
            color=colours.get(centre, "#5f6368"),
            s=48,
            linewidths=1.2,
            label=centre,
        )

    ax_a.axvline(0.0, color="#3f3f3f", linestyle="--", linewidth=1.5)
    ax_a.set_yticks(range(len(groups)))
    ax_a.set_yticklabels(groups, fontsize=8.5)
    ax_a.set_xlabel("Decrease in 730-day AUC after grouped permutation", fontsize=8.5)
    _set_panel_header(ax_a, "A", "Grouped permutation importance, OS", fontsize=8.5)
    ax_a.legend(fontsize=8, title="Development scope", title_fontsize=8)

    minimal = stability[stability["model_name"].astype(str) == "minimal_cox"].copy()
    minimal = minimal[
        minimal["comparison_type"].astype(str).isin({"cross_centre", "cross_endpoint"})
    ]
    if minimal.empty:
        raise ReportContractError("Figure 3 panel B has no minimal_cox agreement rows")

    # Order is fixed rather than left to frame order, and every label carries its endpoint: a
    # `cross_centre` pair appears once per endpoint, so "DFCI vs MSK" alone names two different
    # comparisons and would print twice with no way to tell them apart.
    minimal = minimal.assign(
        _order=minimal["comparison_type"].map({"cross_centre": 0, "cross_endpoint": 1})
    ).sort_values(["_order", "endpoint", "key_a", "key_b"])

    labels, taus, kinds, statuses = [], [], [], []
    for _, row in minimal.iterrows():
        if str(row["comparison_type"]) == "cross_centre":
            labels.append(f"{row['key_a']} vs {row['key_b']}\n({row['endpoint']})")
        else:
            labels.append(f"{row['key_a']}\nvs {row['key_b']}")
        taus.append(float(row["kendall_tau"]) if pd.notna(row["kendall_tau"]) else float("nan"))
        kinds.append(str(row["comparison_type"]))
        statuses.append(str(row["comparison_status"]))

    bar_colours = ["#1a73e8" if k == "cross_centre" else "#f9ab00" for k in kinds]
    positions = list(range(len(labels)))
    ax_b.barh(positions, [0.0 if pd.isna(t) else t for t in taus], color=bar_colours, height=0.62)
    ax_b.axvline(0.0, color="#3f3f3f", linewidth=1.5)
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(labels, fontsize=8.0)
    ax_b.set_xlim(-1.15, 1.15)
    ax_b.set_xlabel("Kendall's tau (attribution rank agreement)", fontsize=8.5)
    _set_panel_header(
        ax_b,
        "B",
        "Attribution rank agreement\nblue = centres; amber = endpoints",
        fontsize=8.5,
    )
    for index, (tau, status) in enumerate(zip(taus, statuses, strict=False)):
        # A bar of length zero and an absent bar look identical. Both are labelled so the reader
        # never has to guess which one they are looking at.
        if pd.isna(tau) or status != "comparable":
            ax_b.annotate(
                "not comparable" if status != "comparable" else "not computable",
                xy=(0.02, index),
                fontsize=8,
                va="center",
                color="#5f6368",
                style="italic",
            )
        elif abs(tau) < 1e-9:
            ax_b.annotate(
                "tau = 0.000",
                xy=(0.02, index),
                fontsize=8,
                va="center",
                color="#3c4043",
                style="italic",
            )
        elif tau <= -0.999:
            ax_b.annotate(
                "complete reversal",
                xy=(tau + 0.04, index),
                fontsize=8,
                va="center",
                color="#d93025",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.62)
    path = output_dir / "supplementary_figure_s25_attribution.png"
    _save(fig, path)
    return DisplayRecord(
        name="Supplementary Figure S25",
        path=str(path),
        source="explainability/importance_summary.csv + attribution_stability.csv",
        restrictions=[
            "minimal_cox only (Tier 1); Tier 2/3 attribution goes to the supplement",
            "panel A uses grouped permutation importance, avoiding collapse of categorical "
            "coefficients",
            "panel A restricted to OS; panel B carries OS and OS-vs-TTNT-D comparisons",
            "VICC OS-vs-TTNT-D reversal is annotated rather than left for the reader to notice",
        ],
    )


# --- Figure 4: sequencing-lag headline curve -----------------------------------------------------


def figure4_sequencing_lag(inputs: CanonicalInputs, output_dir: Path) -> DisplayRecord:
    """The headline sequencing-lag curve with a prespecified minimum-support display rule."""

    support = inputs.frame("phase7_sensitivity", "sequencing_lag_support.csv")
    km = inputs.frame("phase7_sensitivity", "sequencing_lag_km.csv")

    support = support[support["endpoint"] == "OS"]
    km = km[km["endpoint"] == "OS"]
    if km.empty:
        raise ReportContractError("Figure 4 has no OS sequencing-lag rows")

    order = [
        t for t in ("by_folfox", "90d", "180d", "365d", "unrestricted") if t in set(km["threshold"])
    ]
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(MANUSCRIPT_FIGURE_WIDTH_IN, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    minimum_events = 20
    colours = {"DFCI": "#1a73e8", "MSK": "#188038", "VICC": "#d93025", "All": "#5f6368"}
    for group in [*INSTITUTIONS, "All"]:
        cell = km[km["group"].astype(str) == group]
        if cell.empty:
            continue
        support_cell = support[support["group"].astype(str) == group]
        survival_by_threshold = {
            str(row["threshold"]): float(row["survival_730"])
            for _, row in cell.iterrows()
            if str(row["threshold"]) in order
        }
        events_by_threshold = {
            str(row["threshold"]): int(row["n_events"])
            for _, row in support_cell.iterrows()
            if str(row["threshold"]) in order
        }
        xs = list(range(len(order)))
        ys = [
            survival_by_threshold.get(threshold, np.nan)
            if events_by_threshold.get(threshold, 0) >= minimum_events
            else np.nan
            for threshold in order
        ]
        ax_top.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.1 if group != "All" else 2.7,
            linestyle="-" if group != "All" else "--",
            color=colours.get(group, "#5f6368"),
            label=group,
            markersize=6.5,
            markeredgewidth=1.2,
        )

    ax_top.set_ylabel("730-day survival (Kaplan-Meier)", fontsize=9.0)
    _set_panel_header(
        ax_top,
        "A",
        "730-day OS point estimates (shown only with >=20 events)",
        fontsize=9.0,
    )
    ax_top.legend(fontsize=8, title="Centre", title_fontsize=8)
    ax_top.grid(color=FIGURE_GRID, alpha=0.65, linewidth=0.8)
    ax_top.text(
        0.01,
        0.04,
        "Blank where fewer than 20 eligible OS events are retained",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=FIGURE_INK,
    )

    width = 0.2
    for index, group in enumerate(INSTITUTIONS):
        cell = support[support["group"].astype(str) == group]
        xs, heights, events = [], [], []
        for threshold in order:
            match = cell[cell["threshold"].astype(str) == threshold]
            if match.empty:
                continue
            xs.append(order.index(threshold) + (index - 1) * width)
            heights.append(int(match.iloc[0]["n_patients"]))
            events.append(int(match.iloc[0]["n_events"]))
        bars = ax_bottom.bar(
            xs,
            heights,
            width=width,
            color=colours.get(group, "#5f6368"),
            edgecolor=FIGURE_INK,
            linewidth=0.8,
            label=group,
        )
        for bar, event_count in zip(bars, events, strict=False):
            ax_bottom.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(event_count),
                ha="center",
                fontsize=8.0,
                color=FIGURE_INK,
            )

    ax_bottom.set_ylabel("Retained patients\n(bar label = events)", fontsize=9.0)
    _set_panel_header(
        ax_bottom,
        "B",
        "Retained patients and eligible OS events (all thresholds)",
        fontsize=9.0,
    )
    ax_bottom.set_xticks(range(len(order)))
    ax_bottom.set_xticklabels(
        ["By FOLFOX", "<= 90 d", "<= 180 d", "<= 365 d", "Unrestricted"][: len(order)]
    )
    ax_bottom.set_xlabel("Maximum permitted post-FOLFOX sequencing lag", fontsize=9.0)
    ax_bottom.grid(color=FIGURE_GRID, alpha=0.65, axis="y", linewidth=0.8)

    fig.tight_layout(h_pad=1.4)
    path = output_dir / "figure4_sequencing_lag.png"
    _save(fig, path)
    return DisplayRecord(
        name="Figure 4",
        path=str(path),
        source="sensitivity/sequencing_lag_km.csv + sequencing_lag_support.csv",
        restrictions=[
            "OS only; the TTNT-D lag curve goes to the supplement",
            "centre-specific and pooled 730-day survival point estimates are shown only when at "
            "least 20 eligible OS events are retained",
            "line segments break across thresholds that fail the minimum-support rule",
            "the counts panel retains all five thresholds; bar heights are patients and numeric "
            "labels are events",
            "no confidence intervals are displayed; the survival panel is a support sensitivity "
            "diagnostic",
        ],
    )


# --- Tables 2-4 ----------------------------------------------------------------------------------


def _status_label(status: str) -> str:
    return {
        "estimable": "estimable",
        REPORTING_IMPRECISE: "imprecise",
        REPORTING_NOT_COMPUTABLE: "not computable",
    }.get(str(status), str(status))


def _interval(row: pd.Series, estimate: str, lower: str, upper: str, decimals: int) -> str:
    if str(row.get("reporting_status", "")) == REPORTING_NOT_COMPUTABLE or pd.isna(row[estimate]):
        return "not computable"
    return (
        f"{float(row[estimate]):.{decimals}f} "
        f"[{float(row[lower]):.{decimals}f}, {float(row[upper]):.{decimals}f}]"
    )


def table2_primary_performance(inputs: CanonicalInputs) -> pd.DataFrame:
    """Compact held-out OS and TTNT-D performance for the two primary Cox models."""

    performance = inputs.frame("phase5_evaluation", "performance_summary.csv")
    subset = performance[
        (performance["endpoint"].isin(("OS", "TTNT-D")))
        & (performance["design"] == "iecv")
        & (performance["model"].isin(PRIMARY_MODELS))
        & (performance["risk_scale"] == "unconditional")
    ]
    rows: list[dict] = []
    metrics = (
        ("concordance", None, 3, "concordance_95ci"),
        ("auc", 365.0, 3, "auc_365d_95ci"),
        ("auc", 730.0, 3, "auc_730d_95ci"),
        ("slope", 730.0, 3, "slope_730d_95ci"),
        ("ibs", None, 4, "ibs_365_730d_95ci"),
    )
    for endpoint in ("OS", "TTNT-D"):
        for model in PRIMARY_MODELS:
            for centre in INSTITUTIONS:
                result: dict[str, object] = {
                    "endpoint": endpoint,
                    "model": model,
                    "centre": centre,
                }
                status_notes: list[str] = []
                support = subset[
                    (subset["endpoint"] == endpoint)
                    & (subset["model"] == model)
                    & (subset["centre"] == centre)
                    & (subset["metric"] == "concordance")
                    & subset["horizon"].isna()
                ]
                if not support.empty:
                    srow = support.iloc[0]
                    result["patients/events"] = f"{int(srow['n_eligible'])}/{int(srow['n_events'])}"
                for metric, horizon, decimals, column in metrics:
                    cell = subset[
                        (subset["endpoint"] == endpoint)
                        & (subset["model"] == model)
                        & (subset["centre"] == centre)
                        & (subset["metric"] == metric)
                    ]
                    cell = (
                        cell[cell["horizon"].isna()]
                        if horizon is None
                        else cell[cell["horizon"] == horizon]
                    )
                    if cell.empty:
                        result[column] = "not computable"
                        status_notes.append(f"{column}: not computable")
                        continue
                    metric_row = cell.iloc[0]
                    result[column] = _interval(
                        metric_row, "estimate", "ci_lower", "ci_upper", decimals
                    )
                    status = _status_label(metric_row["reporting_status"])
                    if status != "estimable":
                        status_notes.append(f"{column.replace('_95ci', '')}: {status}")
                result["status_notes"] = (
                    "; ".join(status_notes) if status_notes else "all estimable"
                )
                rows.append(result)
    return pd.DataFrame(rows)


def table3_transportability(inputs: CanonicalInputs) -> pd.DataFrame:
    """Paired pooled-minus-held-out differences for OS, with their paired intervals."""

    gaps = inputs.frame("phase5_evaluation", "transportability_summary.csv")
    subset = gaps[(gaps["endpoint"] == "OS") & (gaps["model"].isin(PRIMARY_MODELS))]
    rows: list[dict] = []
    for _, row in subset.iterrows():
        decimals = 4 if str(row["metric"]) in {"ibs", "brier"} else 3
        horizon = "" if pd.isna(row["horizon"]) else f" @ {int(float(row['horizon']))}d"
        computable = pd.notna(row["gap"])
        rows.append(
            {
                "centre": row["centre"],
                "model": row["model"],
                "metric": f"{row['metric']}{horizon}",
                "pooled": f"{float(row['pooled_estimate']):.{decimals}f}"
                if pd.notna(row["pooled_estimate"])
                else "not computable",
                "held_out": f"{float(row['iecv_estimate']):.{decimals}f}"
                if pd.notna(row["iecv_estimate"])
                else "not computable",
                "difference_95ci": (
                    f"{float(row['gap']):.{decimals}f} "
                    f"[{float(row['gap_ci_lower']):.{decimals}f}, "
                    f"{float(row['gap_ci_upper']):.{decimals}f}]"
                )
                if computable
                else "not computable",
                "reason": "" if computable else str(row.get("gap_reason", "")),
            }
        )
    return pd.DataFrame(rows)


def table4_complexity_value(inputs: CanonicalInputs) -> pd.DataFrame:
    """Incremental value of complexity for OS, with the locked three-way classification."""

    value = inputs.frame("phase5_evaluation", "complexity_value_summary.csv")
    subset = value[value["endpoint"] == "OS"]
    rows: list[dict] = []
    for _, row in subset.iterrows():
        decimals = 4 if str(row["gain"]) in {"ibs", "brier"} else 3
        horizon = "" if pd.isna(row["horizon"]) else f" @ {int(float(row['horizon']))}d"
        computable = pd.notna(row["value"])
        rows.append(
            {
                "design": row["design"],
                "centre": row["centre"],
                "fuller_model": row["fuller_model"],
                "gain": f"{row['gain']}{horizon}",
                "gain_95ci": (
                    f"{float(row['value']):.{decimals}f} "
                    f"[{float(row['ci_lower']):.{decimals}f}, "
                    f"{float(row['ci_upper']):.{decimals}f}]"
                )
                if computable
                else "not computable",
                "classification": row["classification"],
                "net_benefit_gain": row.get("net_benefit_gain", "not_applicable"),
            }
        )
    return pd.DataFrame(rows)


def supplementary_minimal_cox_coefficients(inputs: CanonicalInputs) -> pd.DataFrame:
    """Full minimal-Cox coefficient specification with explicit units and reference rows."""

    coefficients = inputs.frame("phase6_explainability", "coefficient_summary.csv")
    dataset = inputs.frame("phase2_dataset", "analysis_dataset.csv")
    eligible = dataset[dataset["os_eligible"].astype(str).str.lower().eq("true")]
    subset = coefficients[
        (coefficients["endpoint"] == "OS") & (coefficients["model"] == "minimal_cox")
    ]
    rows: list[dict] = []
    for centre in INSTITUTIONS:
        development = eligible[eligible["institution"].astype(str) != centre]
        age_sd = float(development["age_at_folfox"].std(ddof=0))
        site_sd = float(development["number_metastatic_sites"].std(ddof=0))
        reference = sorted(
            development["ca_d_site"]
            .astype(str)
            .value_counts()
            .loc[lambda counts: counts == counts.max()]
            .index
        )[0]
        cell = subset[subset["centre"] == centre]
        for _, row in cell.iterrows():
            design = str(row["design_column"])
            if design == "age_at_folfox":
                predictor = f"Age at FOLFOX (per {age_sd:.1f} years; 1 training-scope SD)"
            elif design == "number_metastatic_sites":
                predictor = (
                    f"Number of metastatic sites (per {site_sd:.2f} sites; 1 training-scope SD)"
                )
            elif design.startswith("metastatic_presentation__"):
                predictor = "De novo stage IV vs recurrent metastatic presentation"
            elif design.startswith("ca_d_site__"):
                level = design.split("__", 1)[1].replace("__RARE__", "pooled rare levels")
                predictor = f"Primary site {level} vs {reference}"
            else:
                predictor = design
            rows.append(
                {
                    "development_scope": f"{centre} held out",
                    "predictor_contrast_or_unit": predictor,
                    "coefficient_log_hr": f"{float(row['coefficient']):.4f}",
                    "hazard_ratio_95ci": (
                        f"{float(row['hazard_ratio']):.3f} "
                        f"[{math.exp(float(row['ci_lower'])):.3f}, "
                        f"{math.exp(float(row['ci_upper'])):.3f}]"
                    ),
                    "standard_error": f"{float(row['standard_error']):.4f}",
                    "interval": str(row["interval_provenance"]),
                }
            )
        rows.append(
            {
                "development_scope": f"{centre} held out",
                "predictor_contrast_or_unit": f"Primary site {reference} (reference)",
                "coefficient_log_hr": "0",
                "hazard_ratio_95ci": "1.000 (reference)",
                "standard_error": "not applicable",
                "interval": "model_based",
            }
        )
    return pd.DataFrame(rows)


def supplementary_sex_stratified_performance(inputs: CanonicalInputs) -> pd.DataFrame:
    """Held-out sex-stratified OS concordance with support and reporting status (Table S30)."""

    performance = inputs.frame("fairness_subgroups", "sex_stratified_performance.csv")
    rows: list[dict] = []
    for _, row in performance.iterrows():
        rows.append(
            {
                "centre": row["centre"],
                "group": row["group"],
                "patients/events": f"{int(row['n_patients'])}/{int(row['n_events'])}",
                "concordance_95ci": _interval(row, "estimate", "ci_lower", "ci_upper", 3),
                "reporting_status": _status_label(row["reporting_status"]),
            }
        )
    table = pd.DataFrame(rows)
    assert_descriptive_and_aggregate(table, "Supplementary Table S30 (performance)")
    return table


def supplementary_sociodemographic_support(inputs: CanonicalInputs) -> pd.DataFrame:
    """OS-analysis patients and deaths by centre for each sociodemographic level (Table S30)."""

    support = inputs.frame("fairness_subgroups", "sociodemographic_support.csv")
    rows: list[dict] = []
    for (variable, label), cell in support.groupby(["variable", "level_label"], sort=True):
        indexed = cell.set_index("centre")
        row: dict = {
            "characteristic": "Sex" if variable == "naaccr_sex_code" else "Race/ethnicity",
            "level": label,
        }
        for centre in (*INSTITUTIONS, "All"):
            entry = indexed.loc[centre]
            row[centre] = f"{int(entry['n_patients'])}/{int(entry['n_events'])}"
        rows.append(row)
    table = pd.DataFrame(rows)
    assert_descriptive_and_aggregate(table, "Supplementary Table S30 (support)")
    return table


def table4_model_summary(inputs: CanonicalInputs) -> pd.DataFrame:
    """Key adjusted associations plus a compact centre-by-model complexity classification."""

    coefficients = inputs.frame("phase6_explainability", "coefficient_summary.csv")

    def hr_cell(centre: str, model: str, feature_group: str) -> str:
        match = coefficients[
            (coefficients["endpoint"] == "OS")
            & (coefficients["centre"] == centre)
            & (coefficients["model"] == model)
            & (coefficients["feature_group"] == feature_group)
        ]
        if len(match) != 1:
            return "not estimable"
        row = match.iloc[0]
        return (
            f"{float(row['hazard_ratio']):.2f} "
            f"[{math.exp(float(row['ci_lower'])):.2f}, {math.exp(float(row['ci_upper'])):.2f}]"
        )

    rows = [
        {
            "panel": "A. Adjusted associations",
            "item": "De novo stage IV vs recurrent (minimal Cox)",
            **{
                centre: hr_cell(centre, "minimal_cox", "metastatic_presentation")
                for centre in INSTITUTIONS
            },
        },
        {
            "panel": "A. Adjusted associations",
            "item": "Metastatic-site count, per training-scope SD (minimal Cox)",
            **{
                centre: hr_cell(centre, "minimal_cox", "number_metastatic_sites")
                for centre in INSTITUTIONS
            },
        },
        {
            "panel": "A. Adjusted associations",
            "item": "Abdominal/peritoneal metastasis present vs absent (spline Cox)",
            **{centre: hr_cell(centre, "spline_cox", "dmets_abdomen") for centre in INSTITUTIONS},
        },
    ]

    value = inputs.frame("phase5_evaluation", "complexity_value_summary.csv")
    iecv = value[(value["endpoint"] == "OS") & (value["design"] == "iecv")]
    for model in ("core_cox", "spline_cox", "elastic_net_cox"):
        cells: dict[str, str] = {}
        symbols = {
            "clear_added_value": "A",
            "no_demonstrated_advantage": "N",
            "trade_off_uncertain": "U",
        }
        for centre in INSTITUTIONS:
            labels = (
                iecv[(iecv["fuller_model"] == model) & (iecv["centre"] == centre)]["classification"]
                .dropna()
                .astype(str)
                .unique()
            )
            cells[centre] = symbols.get(labels[0], "?") if len(labels) == 1 else "?"
        rows.append(
            {
                "panel": "B. Added-complexity classification",
                "item": model,
                **cells,
            }
        )
    return pd.DataFrame(rows)


def complexity_status_counts(table4: pd.DataFrame) -> dict[str, int]:
    """The 18-cell OS summary shown alongside Table 4's detail."""

    iecv = table4[table4["design"] == "iecv"]
    return {str(k): int(v) for k, v in iecv["classification"].value_counts().items()}


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as editable GitHub-flavoured Markdown, not as an image."""

    if frame.empty:
        return "_(no rows)_"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])

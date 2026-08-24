"""Forest-model-retry workflow figures: five plots, one function each, written under
``analysis/out/current/the forest-retry figure directory``.

forest-model-retry workflow published its CSVs with no figures at all, while every other analysis
workflow ships eight or
nine. Three of the five below exist specifically because a number that was hard to read in a CSV got
misread, or could not be read at all, in prose:

* ``calibration_slope_intervals`` draws the calibration slope **with its interval** against the
  ideal value of 1. The 2026-08-16 correction to `implement.md` records the claim it replaces —
  "calibration is the forest's clear weakness", asserted from horizon-averaged point estimates while
  10 of 12 intervals contained 1. A point estimate without its interval is what made that mistake
  possible; this figure cannot be drawn without the interval.
* ``calibration_curves`` shows the same question non-parametrically, so a slope near 1 can be
  checked against the actual predicted-versus-observed shape rather than trusted.
* ``scaled_brier`` draws zero as the score of a model using no covariate information at all. The
  report quotes raw integrated Brier scores, from which it is impossible to tell which side of that
  line a model falls on (§10.6).

Every function reads already-computed aggregates or already-fitted models, recomputes no metric, and
is a **no-op** when its source is empty — a reduced synthetic exercise never fails on a missing
figure. Conventions follow `sensitivity/plots.py`: ``Agg`` backend, ``dpi=110``,
``fig.tight_layout()`` before saving, small legend/label fonts, no seaborn and no shared rcParams.

The calibration curve is the one figure that needs per-patient risk rather than a published
aggregate, so it re-predicts from the frozen bundles through `forest.evaluation`'s own cell builders
— the same code path the published performance numbers came from. Nothing is refitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crc_folfox.evaluate.calibration import calibration_curve_groups  # noqa: E402
from crc_folfox.evaluate.contracts import CENTRES, IBS_GRID  # noqa: E402
from crc_folfox.forest.workflow import CANDIDATE  # noqa: E402

BENCHMARK = "minimal_cox"

#: `IBS_GRID` runs 365 → 730 inclusive, so the two SAP horizons are its endpoints.
_HORIZON_INDEX = {365.0: 0, 730.0: len(IBS_GRID) - 1}

_MODEL_STYLE = {
    CANDIDATE: {"color": "tab:green", "marker": "o", "label": "ltrc_cif (forest)"},
    BENCHMARK: {"color": "tab:blue", "marker": "s", "label": "minimal_cox"},
}
_REPORTED = ("estimable", "estimable_imprecise")


def _num(value: object) -> float | None:
    """`None`/`""`/NaN all mean "not reported" in these CSV-round-tripped frames."""
    if value is None or value == "":
        return None
    value = float(value)
    return None if np.isnan(value) else value


# ---------------------------------------------------------------------------
# 1. Calibration slope with intervals -- the figure the 2026-08-16 correction needed
# ---------------------------------------------------------------------------


def render_calibration_slope_intervals(performance: pd.DataFrame, path: Path) -> None:
    """Calibration slope and 95% interval per (centre, horizon), both models, against ideal = 1.

    Drawn for the held-out (IECV) design on the unconditional scale — the primary estimand. An
    interval crossing the dashed ideal line is a cell that does not demonstrate miscalibration,
    which is the great majority of them; that is the point of the figure.
    """
    if performance.empty:
        return
    frame = performance[
        (performance["metric"] == "slope")
        & (performance["design"] == "iecv")
        & (performance["risk_scale"] == "unconditional")
        & (performance["reporting_status"].isin(_REPORTED))
    ]
    if frame.empty:
        return

    cells = [(centre, horizon) for centre in CENTRES for horizon in (365.0, 730.0)]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    offsets = {CANDIDATE: -0.16, BENCHMARK: 0.16}
    for model, style in _MODEL_STYLE.items():
        xs, ys, lo, hi = [], [], [], []
        for position, (centre, horizon) in enumerate(cells):
            row = frame[
                (frame["centre"] == centre)
                & (frame["horizon"] == horizon)
                & (frame["model"] == model)
            ]
            if row.empty:
                continue
            estimate = _num(row.iloc[0]["estimate"])
            lower, upper = _num(row.iloc[0]["ci_lower"]), _num(row.iloc[0]["ci_upper"])
            if estimate is None:
                continue
            xs.append(position + offsets[model])
            ys.append(estimate)
            lo.append(estimate - (lower if lower is not None else estimate))
            hi.append((upper if upper is not None else estimate) - estimate)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            fmt=style["marker"],
            color=style["color"],
            markersize=5,
            capsize=3,
            linewidth=1.2,
            label=style["label"],
        )

    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.9)
    ax.text(len(cells) - 0.55, 1.0, "ideal", fontsize=6, va="bottom", ha="right", color="black")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{centre}\n{int(horizon)} d" for centre, horizon in cells], fontsize=7)
    ax.set_ylabel("calibration slope (95% CI)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(
        "Phase 9 held-out calibration slope, unconditional scale — every cell "
        "`estimable_imprecise`",
        fontsize=8,
    )
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Calibration curves -- predicted versus observed, non-parametric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCurveInput:
    """One model's held-out predictions at one centre, ready for `calibration_curve_groups`."""

    centre: str
    model: str
    entry: np.ndarray
    exit: np.ndarray
    event: np.ndarray
    risk_at_horizon: np.ndarray


def render_calibration_curves(
    curves: list[CalibrationCurveInput], horizon: float, path: Path
) -> None:
    """One panel per centre: mean predicted risk versus Kaplan–Meier observed risk by risk quintile.

    Groups are formed by `evaluate.calibration.calibration_curve_groups`, imported unchanged — the
    same quintile rule, support floors, and delayed-entry KM primary-evaluation workflow uses.
    Low-support groups are
    drawn hollow so a point resting on two patients is visibly different from one resting on fifty.
    """
    if not curves:
        return
    centres = [centre for centre in CENTRES if any(c.centre == centre for c in curves)]
    if not centres:
        return

    # Compute every panel's groups first: the axes are shared across panels and scaled to the
    # realized data range rather than fixed at [0, 1]. Predicted risks here occupy roughly the lower
    # half of the unit square, and plotting that against a full-width diagonal compresses every
    # departure from it into an unreadable band -- which is the departure the figure exists to show.
    computed: dict[str, list[tuple[CalibrationCurveInput, list]]] = {}
    values: list[float] = []
    for centre in centres:
        panel: list[tuple[CalibrationCurveInput, list]] = []
        for curve in (c for c in curves if c.centre == centre):
            result = calibration_curve_groups(
                curve.entry, curve.exit, curve.event, curve.risk_at_horizon, horizon
            )
            if result.status != "ok":
                continue
            supported = [g for g in result.groups if g.observed_risk is not None]
            if not supported:
                continue
            panel.append((curve, supported))
            values.extend(g.mean_predicted_risk for g in supported)
            values.extend(float(g.observed_risk) for g in supported)
        computed[centre] = panel
    if not values:
        return
    pad = 0.05 * max(max(values) - min(values), 0.1)
    low, high = max(0.0, min(values) - pad), min(1.0, max(values) + pad)

    # Extra height: the equal aspect ratio below pins panel width to panel height, so `tight_layout`
    # cannot reclaim space for the x-label by shrinking the axes.
    fig, axes = plt.subplots(1, len(centres), figsize=(3.4 * len(centres), 4.2), squeeze=False)
    for column, centre in enumerate(centres):
        ax = axes[0][column]
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=0.8, zorder=1)
        for curve, supported in computed[centre]:
            style = _MODEL_STYLE[curve.model]
            xs = [g.mean_predicted_risk for g in supported]
            ys = [float(g.observed_risk) for g in supported]
            ax.plot(
                xs, ys, "-", color=style["color"], linewidth=1.1, label=style["label"], zorder=2
            )
            for group, x, y in zip(supported, xs, ys, strict=True):
                ax.plot(
                    x,
                    y,
                    style["marker"],
                    color=style["color"],
                    markersize=5,
                    markerfacecolor="none" if group.low_support else style["color"],
                    zorder=3,
                )
        ax.set_title(centre, fontsize=8)
        ax.set_xlabel("mean predicted risk", fontsize=7)
        if column == 0:
            ax.set_ylabel(f"observed risk at {int(horizon)} d", fontsize=7)
            ax.legend(fontsize=6.5, loc="upper left")
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Phase 9 held-out calibration, {int(horizon)} d (hollow markers = low-support group)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. The forest's own permutation importance
# ---------------------------------------------------------------------------


def render_forest_importance(importance: pd.DataFrame, horizon_label: str, path: Path) -> None:
    """Grouped horizontal bars, one bar per centre within each feature group.

    Groups are ordered by their across-centre mean, so the reader sees a ranking rather than an
    alphabetical list. Bars carry **no error bars, deliberately**: forest-model-retry workflow
    computed no bootstrap
    importance intervals (~125 hours of refits), and drawing a bare bar is the honest rendering of a
    point estimate. Negative importances — permuting the group *improved* the score — are real and
    drawn as-is rather than clipped at zero.
    """
    if importance.empty:
        return
    frame = importance[(importance["model"] == CANDIDATE) & (importance["metric"] == horizon_label)]
    if frame.empty:
        return
    wide = frame.pivot(index="feature_group", columns="centre", values="importance")
    wide = wide.reindex(wide.mean(axis=1).sort_values().index)
    centres = [centre for centre in CENTRES if centre in wide.columns]
    if not centres:
        return

    positions = np.arange(len(wide))
    height = 0.8 / len(centres)
    colors = dict(zip(CENTRES, ("tab:blue", "tab:orange", "tab:green"), strict=False))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(wide) + 1.6))
    for index, centre in enumerate(centres):
        ax.barh(
            positions + index * height - 0.4 + height / 2,
            wide[centre].to_numpy(dtype=float),
            height=height,
            color=colors.get(centre, "grey"),
            label=centre,
        )
    ax.axvline(0.0, color="black", linewidth=0.8)

    # A group with importance exactly 0.0 at every centre draws no visible bar, which reads as
    # missing data. It is the opposite: permuting a column changes nothing only when no tree in the
    # ensemble ever split on it, so the absent bar is a finding and is labelled as one.
    for position, group_name in enumerate(wide.index):
        if np.all(wide.loc[group_name].to_numpy(dtype=float) == 0.0):
            ax.text(
                0.0,
                position,
                "  exactly 0 — never split on",
                fontsize=6,
                va="center",
                ha="left",
                style="italic",
                color="dimgrey",
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(wide.index, fontsize=7)
    ax.set_xlabel(
        f"increase in delayed-entry Brier score when permuted ({horizon_label} d)", fontsize=7.5
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        "Phase 9 grouped permutation importance, ltrc_cif — point estimates, no intervals computed",
        fontsize=8,
    )
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Rank agreement between the forest and the Cox models
# ---------------------------------------------------------------------------


def render_rank_agreement(agreement: pd.DataFrame, path: Path) -> None:
    """Kendall tau-b by centre and metric, across every comparator.

    Three series, because they carry different weight. `spline_cox` is the primary comparison over
    the full 15-group core inventory. The five further core-set models are drawn small and hollow —
    individually redundant, collectively the answer to "does the forest agree with the Cox *family*,
    or only with `spline_cox`?". `minimal_cox` is drawn as a distinct hollow triangle and annotated
    with its shared-group count, because it rests on three shared groups and is `restricted_scopes`;
    the artifact says so in a column, and the figure has to say so too or the plot will be read as
    equal comparisons.
    """
    if agreement.empty:
        return
    metrics = ["365", "730", "ibs"]
    cells = [(centre, metric) for centre in CENTRES for metric in metrics]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    for role, marker, filled, color in (
        ("primary", "o", True, "tab:green"),
        ("supporting", "o", False, "tab:olive"),
        ("descriptive", "^", False, "tab:grey"),
    ):
        subset = agreement[agreement["role"] == role]
        if subset.empty:
            continue
        # Five supporting comparators at one x position render as a single marker, hiding both
        # their number and their spread. A small deterministic offset per comparator (never random:
        # the figure must redraw identically) makes the cluster legible as a cluster.
        order = sorted(set(subset["model_b"]))
        span = 0.34
        offsets = {
            name: 0.0 if len(order) == 1 else -span / 2 + span * index / (len(order) - 1)
            for index, name in enumerate(order)
        }
        xs, ys, shared = [], [], []
        for position, (centre, metric) in enumerate(cells):
            rows = subset[(subset["centre"] == centre) & (subset["metric"].astype(str) == metric)]
            for row in rows.itertuples(index=False):
                tau = _num(row.kendall_tau)
                if tau is None:
                    continue
                xs.append(position + offsets[str(row.model_b)])
                ys.append(tau)
                shared.append(int(row.n_shared_groups))
        if not xs:
            continue
        comparators = sorted(set(subset["model_b"]))
        label = (
            f"vs {comparators[0]} ({role}, {shared[0]} shared groups)"
            if len(comparators) == 1
            else f"vs {len(comparators)} further core models ({role}, {shared[0]} shared groups)"
        )
        ax.plot(
            xs,
            ys,
            marker,
            color=color,
            markersize=6 if role != "supporting" else 4,
            markerfacecolor=color if filled else "none",
            linestyle="none",
            label=label,
            zorder=3 if role == "primary" else 2,
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{centre}\n{metric}" for centre, metric in cells], fontsize=6.5)
    ax.set_ylabel("Kendall tau-b vs the forest's ranking", fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(
        "Phase 9 rank agreement — forest versus Cox feature-group importance ordering", fontsize=8
    )
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Scaled Brier against the no-information reference (§10.6)
# ---------------------------------------------------------------------------


def render_scaled_brier(forest_scaled: pd.DataFrame, cox_scaled: pd.DataFrame, path: Path) -> None:
    """Scaled Brier by centre and metric, forest against `minimal_cox`, with zero drawn as the
    no-information line.

    Zero is not a cosmetic gridline here: it is the score of a model that uses **no covariate
    information at all**, only the population Kaplan–Meier. A point near zero means the covariates
    bought almost nothing, and a point below it means they actively hurt. That reading is the whole
    reason §10.6 was run — the forest-retry report quotes raw integrated Brier scores, from which it
    is
    impossible to tell which side of that line a model is on.

    `cox_scaled` is sensitivity-analysis workflow's **published** `minimal_cox` frame, read and
    never recomputed.
    """
    if forest_scaled.empty:
        return
    designs = [d for d in ("iecv", "pooled") if d in set(forest_scaled["design"])]
    metrics = [m for m in ("brier_365", "brier_730", "ibs") if m in set(forest_scaled["metric"])]
    if not designs or not metrics:
        return

    fig, axes = plt.subplots(1, len(designs), figsize=(4.6 * len(designs), 3.9), squeeze=False)
    cells = [(centre, metric) for centre in CENTRES for metric in metrics]
    for column, design in enumerate(designs):
        ax = axes[0][column]
        for frame, style in (
            (forest_scaled, _MODEL_STYLE[CANDIDATE]),
            (cox_scaled, _MODEL_STYLE[BENCHMARK]),
        ):
            if frame is None or frame.empty:
                continue
            subset = frame[frame["design"] == design]
            xs, ys = [], []
            for position, (centre, metric) in enumerate(cells):
                row = subset[(subset["centre"] == centre) & (subset["metric"] == metric)]
                if row.empty:
                    continue
                value = _num(row.iloc[0]["scaled_brier"])
                if value is None:
                    continue
                xs.append(position)
                ys.append(value)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    style["marker"],
                    color=style["color"],
                    markersize=5,
                    linestyle="none",
                    label=style["label"],
                )
        ax.axhline(0.0, linestyle="--", color="black", linewidth=0.9)
        ax.text(
            len(cells) - 0.4,
            0.0,
            "no-information (Kaplan–Meier)",
            fontsize=5.5,
            va="bottom",
            ha="right",
            color="black",
        )
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(
            [f"{centre}\n{metric.replace('brier_', '')}" for centre, metric in cells], fontsize=6
        )
        ax.set_title(design, fontsize=8)
        if column == 0:
            ax.set_ylabel("scaled Brier  (1 − Brier / Brier$_{KM}$)", fontsize=7.5)
            ax.legend(fontsize=6.5, loc="upper left")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(
        "Phase 9 scaled Brier, OS — descriptive aid; both families sit within a few percent "
        "of no information",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _curve_inputs(
    phase2: Path, phase9: Path, phase4: Path, endpoint: str, horizon: float
) -> list[CalibrationCurveInput]:
    """Per-patient held-out risk at one horizon, for both models at all three centres.

    This is the only figure input that cannot come from a published aggregate — a calibration curve
    needs the individual predictions. They are produced by `forest.evaluation`'s own cell builders,
    the identical code path the published `performance_summary.csv` came from, so the curve and the
    slope describe the same predictions. Nothing is refitted.
    """
    from crc_folfox.evaluate.workflow import _predict_iecv, _verify_phase4
    from crc_folfox.forest.evaluation import _forest_iecv_cell, _load_forest_registry
    from crc_folfox.forest.r_adapter import RConditionalForestAdapter
    from crc_folfox.pipeline import load_clean_analysis_dataset
    from crc_folfox.preprocessing.contracts import build_endpoint_view
    from crc_folfox.preprocessing.splits import build_iecv_splits

    index = _HORIZON_INDEX[horizon]
    frame = load_clean_analysis_dataset(phase2 / "analysis_dataset.csv")
    cox_registry, _fit_summary, _hashes = _verify_phase4(phase4)
    forest_registry = _load_forest_registry(phase9)
    view = build_endpoint_view(frame, endpoint)
    adapter = RConditionalForestAdapter()

    inputs: list[CalibrationCurveInput] = []
    for split in build_iecv_splits(view):
        centre = str(split.held_out_institution)
        cells = {
            BENCHMARK: _predict_iecv(view, split, BENCHMARK, cox_registry, phase4 / "models"),
            CANDIDATE: _forest_iecv_cell(view, split, forest_registry, phase9 / "models", adapter),
        }
        for model, cell in cells.items():
            if cell.status != "ok" or cell.risk_grid_unconditional is None:
                continue
            inputs.append(
                CalibrationCurveInput(
                    centre=centre,
                    model=model,
                    entry=np.asarray(cell.entry, dtype=float),
                    exit=np.asarray(cell.exit, dtype=float),
                    event=np.asarray(cell.event, dtype=int),
                    risk_at_horizon=cell.risk_grid_unconditional[:, index],
                )
            )
    return inputs


def run_forest_figures(
    phase9_directory: str | Path,
    *,
    phase2_directory: str | Path | None = None,
    phase4_directory: str | Path | None = None,
    endpoint: str = "OS",
    skip_calibration_curves: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Draw the forest-model-retry workflow figure set; return the paths actually written.

    `skip_calibration_curves` exists for environments without the R runtime — the other three
    figures read published CSVs only and always work.
    """
    phase9 = Path(phase9_directory).resolve()
    phase2 = Path(phase2_directory).resolve() if phase2_directory else phase9.parent
    phase4 = Path(phase4_directory).resolve() if phase4_directory else phase2 / "model_training"
    figures = phase9 / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def _read(name: str) -> pd.DataFrame:
        path = phase9 / name
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    written: list[Path] = []

    def _record(path: Path) -> None:
        if path.is_file():
            written.append(path)
            if progress:
                print(f"  wrote {path.name}", flush=True)

    performance = _read("performance_summary.csv")
    slope_path = figures / f"calibration_slope_intervals_{endpoint}_iecv.png"
    render_calibration_slope_intervals(performance, slope_path)
    _record(slope_path)

    importance = _read("importance_summary.csv")
    for horizon_label in ("730", "365"):
        importance_path = figures / f"permutation_importance_ltrc_cif_{horizon_label}.png"
        render_forest_importance(importance, horizon_label, importance_path)
        _record(importance_path)

    agreement = _read("importance_agreement.csv")
    agreement_path = figures / "rank_agreement_forest_vs_cox.png"
    render_rank_agreement(agreement, agreement_path)
    _record(agreement_path)

    # sensitivity-analysis workflow's published `minimal_cox` scaled Brier is the comparison series,
    # read rather than
    # recomputed — the same rule §9 follows for explainability workflow's importances.
    phase7_scaled = phase2 / "sensitivity" / "scaled_brier.csv"
    cox_scaled = pd.DataFrame()
    if phase7_scaled.is_file():
        published = pd.read_csv(phase7_scaled)
        cox_scaled = published[
            (published["endpoint"] == endpoint) & (published["model"] == BENCHMARK)
        ]
    scaled_path = figures / f"scaled_brier_reference_{endpoint}.png"
    render_scaled_brier(_read("scaled_brier.csv"), cox_scaled, scaled_path)
    _record(scaled_path)

    if not skip_calibration_curves:
        for horizon in (365.0, 730.0):
            if progress:
                print(f"  predicting for the {int(horizon)} d calibration curve ...", flush=True)
            curve_path = figures / f"calibration_{endpoint}_iecv_{int(horizon)}.png"
            render_calibration_curves(
                _curve_inputs(phase2, phase9, phase4, endpoint, horizon), horizon, curve_path
            )
            _record(curve_path)

    return written


__all__ = [
    "BENCHMARK",
    "CalibrationCurveInput",
    "render_calibration_curves",
    "render_calibration_slope_intervals",
    "render_forest_importance",
    "render_rank_agreement",
    "render_scaled_brier",
    "run_forest_figures",
]

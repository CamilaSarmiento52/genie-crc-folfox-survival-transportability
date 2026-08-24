"""One-off figure generation for the published explainability bundle.

Reads only `analysis/out/current/explainability/*.csv` (the already-published, corrected real
inventory) and writes PNGs into that bundle's existing `figures/` directory -- no model loading,
no refitting, no re-running `run_explainability_workflow`. Follows the same small-multiple,
`dpi=110`, `Agg`-backend style as `crc_folfox.evaluate.workflow._render_figures`, for visual
consistency with the primary evaluation's published figures.

Usage:
    uv run python scripts/generate_explainability_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "analysis" / "out" / "current" / "explainability"
FIGURES_DIR = BUNDLE_DIR / "figures"

CENTRES = ("DFCI", "MSK", "VICC")
ENDPOINTS = ("OS", "TTNT-D")


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(BUNDLE_DIR / name)


def _clean_log_xaxis(ax) -> None:
    """A narrow log-scale range makes matplotlib's default minor-tick labels overlap into
    unreadable text; keep only a handful of clean major ticks."""
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))


def _forest_plot(coefficient_summary: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: minimal_cox hazard-ratio forest plot, one row per real design column, faceted by
    (endpoint, centre). CI bounds are on the log-hazard-ratio scale in the CSV -- exponentiate."""
    mc = coefficient_summary[coefficient_summary["model"] == "minimal_cox"].copy()
    mc = mc.sort_values(["feature_group", "design_column"])

    fig, axes = plt.subplots(
        len(ENDPOINTS),
        len(CENTRES),
        figsize=(4.2 * len(CENTRES), 3.4 * len(ENDPOINTS)),
        squeeze=False,
    )
    for row_index, endpoint in enumerate(ENDPOINTS):
        for col_index, centre in enumerate(CENTRES):
            ax = axes[row_index][col_index]
            cell = mc[(mc["endpoint"] == endpoint) & (mc["centre"] == centre)]
            if cell.empty:
                ax.set_visible(False)
                continue
            y = np.arange(len(cell))
            hr = cell["hazard_ratio"].to_numpy(dtype=float)
            hr_lower = np.exp(cell["ci_lower"].to_numpy(dtype=float))
            hr_upper = np.exp(cell["ci_upper"].to_numpy(dtype=float))
            xerr = np.vstack([hr - hr_lower, hr_upper - hr])
            ax.errorbar(
                hr, y, xerr=xerr, fmt="o", markersize=3, linewidth=1, capsize=2, color="#2b6cb0"
            )
            ax.axvline(1.0, linestyle="--", linewidth=0.8, color="gray")
            ax.set_yticks(y)
            ax.set_yticklabels(cell["design_column"], fontsize=6)
            ax.set_xscale("log")
            _clean_log_xaxis(ax)
            ax.set_title(f"{endpoint}/{centre}", fontsize=8)
            ax.set_xlabel("hazard ratio (log scale)", fontsize=6)
            ax.tick_params(axis="x", labelsize=6)
            ax.invert_yaxis()
    fig.suptitle("minimal_cox coefficient forest plot (design-column level, model-based 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "coefficient_forest_minimal_cox.png", dpi=110)
    plt.close(fig)


def _importance_plot(
    importance_summary: pd.DataFrame, bootstrap_attribution_stability: pd.DataFrame, out_dir: Path
) -> None:
    """Figure 2: minimal_cox 730-day permutation importance, point estimate from
    importance_summary.csv, uncertainty interval from bootstrap_attribution_stability.csv."""
    imp = importance_summary[
        (importance_summary["model"] == "minimal_cox") & (importance_summary["metric"] == "730")
    ]
    boot = bootstrap_attribution_stability[
        (bootstrap_attribution_stability["model"] == "minimal_cox")
        & (bootstrap_attribution_stability["metric"] == "730")
    ]
    merged = imp.merge(
        boot[["endpoint", "centre", "feature_group", "bootstrap_ci_lower", "bootstrap_ci_upper"]],
        on=["endpoint", "centre", "feature_group"],
        how="left",
    ).sort_values("feature_group")

    fig, axes = plt.subplots(
        len(ENDPOINTS),
        len(CENTRES),
        figsize=(4.2 * len(CENTRES), 2.8 * len(ENDPOINTS)),
        squeeze=False,
    )
    for row_index, endpoint in enumerate(ENDPOINTS):
        for col_index, centre in enumerate(CENTRES):
            ax = axes[row_index][col_index]
            cell = merged[(merged["endpoint"] == endpoint) & (merged["centre"] == centre)]
            if cell.empty:
                ax.set_visible(False)
                continue
            y = np.arange(len(cell))
            point = cell["importance"].to_numpy(dtype=float)
            lower = cell["bootstrap_ci_lower"].to_numpy(dtype=float)
            upper = cell["bootstrap_ci_upper"].to_numpy(dtype=float)
            xerr = np.vstack([point - lower, upper - point])
            xerr = np.clip(xerr, 0, None)
            ax.errorbar(
                point, y, xerr=xerr, fmt="o", markersize=3, linewidth=1, capsize=2, color="#2f855a"
            )
            ax.axvline(0.0, linestyle="--", linewidth=0.8, color="gray")
            ax.set_yticks(y)
            ax.set_yticklabels(cell["feature_group"], fontsize=6)
            ax.set_title(f"{endpoint}/{centre}", fontsize=8)
            ax.set_xlabel("730d Brier-increase importance", fontsize=6)
            ax.tick_params(axis="x", labelsize=6)
            ax.invert_yaxis()
    fig.suptitle(
        "minimal_cox permutation importance, 730d "
        "(point: E2; interval: bootstrap-refit distribution)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance_minimal_cox_730.png", dpi=110)
    plt.close(fig)


def _partial_effect_grid(
    partial_effects: pd.DataFrame,
    model_name: str,
    out_name: str,
    title: str,
    out_dir: Path,
    *,
    by_presentation_level: bool,
) -> None:
    """Shared small-multiple layout for the two age partial-effect figures (Figure 3a/3b)."""
    subset = partial_effects[partial_effects["model"] == model_name]
    fig, axes = plt.subplots(
        len(ENDPOINTS),
        len(CENTRES),
        figsize=(4.2 * len(CENTRES), 3.0 * len(ENDPOINTS)),
        squeeze=False,
    )
    for row_index, endpoint in enumerate(ENDPOINTS):
        for col_index, centre in enumerate(CENTRES):
            ax = axes[row_index][col_index]
            cell = subset[(subset["endpoint"] == endpoint) & (subset["centre"] == centre)]
            if cell.empty:
                ax.set_visible(False)
                continue
            groups = cell.groupby("presentation_level") if by_presentation_level else [(None, cell)]
            colors = ("#c05621", "#2b6cb0")
            for level_index, (level, curve) in enumerate(groups):
                curve = curve.sort_values("age_value")
                color = colors[level_index % len(colors)]
                ax.plot(
                    curve["age_value"], curve["effect"], color=color, linewidth=1.2, label=level
                )
                if curve["ci_lower"].notna().any():
                    ax.fill_between(
                        curve["age_value"],
                        curve["ci_lower"],
                        curve["ci_upper"],
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                    )
            ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
            ax.set_title(f"{endpoint}/{centre}", fontsize=8)
            ax.set_xlabel("age at FOLFOX start", fontsize=6)
            ax.set_ylabel("log relative hazard", fontsize=6)
            ax.tick_params(labelsize=6)
            if by_presentation_level:
                ax.legend(fontsize=5, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=110)
    plt.close(fig)


def _rank_agreement_cross_centre(attribution_stability: pd.DataFrame, out_dir: Path) -> None:
    """Figure 4a: minimal_cox cross-centre Kendall's tau, one 3x3 heatmap per endpoint."""
    rows = attribution_stability[
        (attribution_stability["model_name"] == "minimal_cox")
        & (attribution_stability["comparison_type"] == "cross_centre")
    ]
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(4.2 * len(ENDPOINTS), 3.6))
    for panel_index, endpoint in enumerate(ENDPOINTS):
        ax = axes[panel_index]
        matrix = pd.DataFrame(1.0, index=CENTRES, columns=CENTRES)
        subset = rows[rows["endpoint"] == endpoint]
        for _, row in subset.iterrows():
            matrix.loc[row["key_a"], row["key_b"]] = row["kendall_tau"]
            matrix.loc[row["key_b"], row["key_a"]] = row["kendall_tau"]
        im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(CENTRES)))
        ax.set_xticklabels(CENTRES, fontsize=7)
        ax.set_yticks(range(len(CENTRES)))
        ax.set_yticklabels(CENTRES, fontsize=7)
        for i in range(len(CENTRES)):
            for j in range(len(CENTRES)):
                ax.text(
                    j,
                    i,
                    f"{matrix.iat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        ax.set_title(f"{endpoint}", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Kendall's tau")
    fig.suptitle("minimal_cox cross-centre attribution-rank agreement (n_groups = 4)")
    fig.savefig(out_dir / "rank_agreement_cross_centre.png", dpi=110)
    plt.close(fig)


def _rank_agreement_cross_endpoint(attribution_stability: pd.DataFrame, out_dir: Path) -> None:
    """Figure 4b: minimal_cox cross-endpoint (OS vs TTNT-D) Kendall's tau by centre, highlighting
    the real VICC rank reversal."""
    rows = attribution_stability[
        (attribution_stability["model_name"] == "minimal_cox")
        & (attribution_stability["comparison_type"] == "cross_endpoint")
    ].copy()
    rows["centre"] = rows["key_a"].str.split("@").str[1]
    rows = rows.set_index("centre").reindex(CENTRES).reset_index()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    colors = ["#c53030" if centre == "VICC" else "#2b6cb0" for centre in rows["centre"]]
    ax.barh(rows["centre"], rows["kendall_tau"], color=colors)
    ax.axvline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Kendall's tau (OS vs TTNT-D attribution rank)")
    vicc_tau = rows.loc[rows["centre"] == "VICC", "kendall_tau"].iloc[0]
    vicc_y = rows.index[rows["centre"] == "VICC"][0]
    # The VICC bar can span the full plotted range (tau = -1.0 reaches the left edge), so the
    # label is anchored in the white space on the opposite side of zero, with an arrow pointing
    # back to the bar -- never drawn on top of the bar's own fill colour.
    label_x = 0.08 if vicc_tau < 0 else -0.75
    ax.annotate(
        "VICC cross-endpoint\nrank reversal",
        xy=(vicc_tau, vicc_y),
        xytext=(label_x, vicc_y),
        fontsize=7,
        color="black",
        va="center",
        ha="left" if vicc_tau < 0 else "right",
        arrowprops={"arrowstyle": "->", "color": "#c53030", "linewidth": 1},
    )
    fig.suptitle("minimal_cox cross-endpoint attribution-rank agreement\n(n_groups = 4 -- coarse)")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_agreement_cross_endpoint.png", dpi=110)
    plt.close(fig)


def _cross_family_spread(cross_family_prediction_spread: pd.DataFrame, out_dir: Path) -> None:
    """Figure 5: median and 90th-percentile cross-family prediction spread by (endpoint, centre),
    one panel per horizon, restricted (availability-confounded) cells marked with hatching."""
    horizons = sorted(cross_family_prediction_spread["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.0 * len(horizons), 3.6), squeeze=False)
    axes = axes[0]
    for panel_index, horizon in enumerate(horizons):
        ax = axes[panel_index]
        cell = cross_family_prediction_spread[
            cross_family_prediction_spread["horizon"] == horizon
        ].copy()
        cell["label"] = cell["endpoint"] + "/" + cell["centre"]
        x = np.arange(len(cell))
        width = 0.35
        bars_median = ax.bar(x - width / 2, cell["median_spread"], width, label="median spread")
        bars_p90 = ax.bar(x + width / 2, cell["p90_spread"], width, label="p90 spread")
        for bar_group in (bars_median, bars_p90):
            for bar, restricted in zip(bar_group, cell["availability_confounded"], strict=True):
                if restricted:
                    bar.set_hatch("//")
                    bar.set_edgecolor("#c53030")
        ax.set_xticks(x)
        ax.set_xticklabels(cell["label"], fontsize=6, rotation=45, ha="right")
        ax.set_title(f"{int(horizon)}d horizon", fontsize=9)
        ax.set_ylabel("cross-model prediction spread", fontsize=7)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Cross-family absolute-risk prediction spread (distribution summaries, not CIs)\n"
        "hatched = restricted / availability_confounded",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_dir / "cross_family_prediction_spread.png", dpi=110)
    plt.close(fig)


def _coefficient_path_elastic_net(coefficient_paths: pd.DataFrame, out_dir: Path) -> None:
    """Figure 6a: elastic_net_cox coefficient trajectories vs penalizer, faceted by l1_ratio, for
    one representative scope (OS/DFCI) -- the tuning axes are penalizer x l1_ratio, distinct from
    glmnet_enet_core's single-axis lambda path, so the two are never drawn on a shared x-axis."""
    en = coefficient_paths[
        (coefficient_paths["model"] == "elastic_net_cox")
        & (coefficient_paths["endpoint"] == "OS")
        & (coefficient_paths["centre"] == "DFCI")
        & (coefficient_paths["fit_status"] == "succeeded")
    ]
    l1_ratios = sorted(en["l1_ratio"].unique())
    fig, axes = plt.subplots(1, len(l1_ratios), figsize=(4.2 * len(l1_ratios), 3.6), squeeze=False)
    axes = axes[0]
    columns = sorted(en["design_column"].unique())
    cmap = plt.get_cmap("tab20")
    for panel_index, l1_ratio in enumerate(l1_ratios):
        ax = axes[panel_index]
        panel = en[en["l1_ratio"] == l1_ratio]
        for col_index, design_column in enumerate(columns):
            line = panel[panel["design_column"] == design_column].sort_values("penalizer")
            if line.empty:
                continue
            ax.plot(
                line["penalizer"],
                line["coefficient"],
                color=cmap(col_index % 20),
                linewidth=1,
                label=design_column,
            )
            selected = line[line["is_selected_configuration"]]
            if not selected.empty:
                ax.scatter(
                    selected["penalizer"],
                    selected["coefficient"],
                    color=cmap(col_index % 20),
                    marker="*",
                    s=60,
                    zorder=5,
                )
        ax.set_xscale("log")
        _clean_log_xaxis(ax)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
        ax.set_title(f"l1_ratio = {l1_ratio}", fontsize=8)
        ax.set_xlabel("penalizer (log scale)", fontsize=6)
        ax.set_ylabel("coefficient", fontsize=6)
        ax.tick_params(labelsize=6)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        fontsize=5,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "elastic_net_cox coefficient path "
        "(OS/DFCI, representative scope; ★ = selected configuration)"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "coefficient_path_elastic_net_cox.png", dpi=110)
    plt.close(fig)


def _coefficient_path_glmnet(coefficient_paths: pd.DataFrame, out_dir: Path) -> None:
    """Figure 6b: glmnet_enet_core (post_hoc_exploratory) coefficient trajectories vs log(lambda),
    one representative scope (OS/DFCI). A separate x-axis from elastic_net_cox's penalizer/l1_ratio
    grid -- the two model families' tuning axes are never plotted together."""
    ge = coefficient_paths[
        (coefficient_paths["model"] == "glmnet_enet_core")
        & (coefficient_paths["endpoint"] == "OS")
        & (coefficient_paths["centre"] == "DFCI")
        & (coefficient_paths["fit_status"] == "succeeded")
    ]
    columns = sorted(ge["design_column"].unique())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col_index, design_column in enumerate(columns):
        line = ge[ge["design_column"] == design_column].sort_values("lambda_value")
        if line.empty:
            continue
        ax.plot(
            np.log(line["lambda_value"]),
            line["coefficient"],
            color=cmap(col_index % 20),
            linewidth=0.9,
        )
        selected = line[line["is_selected_configuration"]]
        if not selected.empty:
            ax.scatter(
                np.log(selected["lambda_value"]),
                selected["coefficient"],
                color=cmap(col_index % 20),
                marker="*",
                s=60,
                zorder=5,
            )
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("log(lambda)")
    ax.set_ylabel("coefficient")
    fig.suptitle(
        "glmnet_enet_core coefficient path -- post_hoc_exploratory\n"
        "(OS/DFCI, representative scope; ★ = selected configuration)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "coefficient_path_glmnet_enet_core.png", dpi=110)
    plt.close(fig)


def main() -> None:
    if not BUNDLE_DIR.is_dir():
        raise SystemExit(f"Published bundle not found at {BUNDLE_DIR}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coefficient_summary = _load("coefficient_summary.csv")
    importance_summary = _load("importance_summary.csv")
    bootstrap_attribution_stability = _load("bootstrap_attribution_stability.csv")
    partial_effects = _load("partial_effects.csv")
    attribution_stability = _load("attribution_stability.csv")
    cross_family_prediction_spread = _load("cross_family_prediction_spread.csv")
    coefficient_paths = _load("coefficient_paths.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _forest_plot(coefficient_summary, FIGURES_DIR)
        _importance_plot(importance_summary, bootstrap_attribution_stability, FIGURES_DIR)
        _partial_effect_grid(
            partial_effects,
            "spline_cox",
            "partial_effect_age_spline_cox.png",
            "spline_cox age partial effect",
            FIGURES_DIR,
            by_presentation_level=False,
        )
        _partial_effect_grid(
            partial_effects,
            "penalized_nonlinear_cox",
            "partial_effect_age_penalized_nonlinear_cox.png",
            "penalized_nonlinear_cox age partial effect -- post_hoc_exploratory",
            FIGURES_DIR,
            by_presentation_level=True,
        )
        _rank_agreement_cross_centre(attribution_stability, FIGURES_DIR)
        _rank_agreement_cross_endpoint(attribution_stability, FIGURES_DIR)
        _cross_family_spread(cross_family_prediction_spread, FIGURES_DIR)
        _coefficient_path_elastic_net(coefficient_paths, FIGURES_DIR)
        _coefficient_path_glmnet(coefficient_paths, FIGURES_DIR)

    written = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    print(f"Wrote {len(written)} figures to {FIGURES_DIR}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()

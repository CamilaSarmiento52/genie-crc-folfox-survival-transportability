"""Manuscript-reporting workflow's two computed frames -- the only new numbers this workflow is
permitted to produce.

`manuscript-spec.md` section 4.3 admits exactly two exceptions to "every number is extracted, never
typed": the cohort-flow diagram counts and the Table 1 descriptives. Both are computed here from the
frozen data-cleaning workflow canonical dataset, written to their own aggregate artifacts, and then
consumed
through the same token mechanism as everything else.

The cohort flow reproduces `tech-stack.md` section 5 **exactly** or the build fails. That gate is
the point of the frame: the research proposal's own Table 1 was back-computed from the source cohort
and does not describe the delayed-entry analysis populations, so a manuscript that transcribed it
would overstate every fold. The reconciliation makes that error impossible to commit silently.

Table 1 carries **no cross-centre p-value**. A significance test of baseline imbalance between
held-out centres would invite a descriptive table to be read as a hypothesis test, and no such test
is prespecified anywhere in this study.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import pandas as pd

from crc_folfox.features.manifest import LOCKED_CORE
from crc_folfox.preprocessing.contracts import INSTITUTIONS
from crc_folfox.report.contracts import ReportContractError
from crc_folfox.report.inputs import CanonicalInputs, sha256_of

# --- Frozen expectations from tech-stack.md section 5 -------------------------------------------
#
# These are the regression-test constants the cohort flow must reproduce. They are duplicated here
# deliberately: a computed frame that reconciles against a value imported from the same code that
# computed it would reconcile against itself. The specification is the independent authority.

COHORT_FLOW_STEPS: tuple[tuple[str, int], ...] = (
    ("All release patients", 1485),
    ("Documented advanced index CRC", 1282),
    ("At least one linked post-metastatic regimen", 1150),
    ("First regimen contains FOLFOX backbone", 679),
    ("Primary FOLFOX definition and adult", 678),
)

EXPECTED_COHORT_TOTAL = 678
EXPECTED_PHENOTYPE = {"exact_folfox": 439, "folfox_plus_biologic": 239}
#: Source-cohort endpoint totals across all 678 patients. These are the numbers the research
#: proposal quotes, and they are *not* the analysis populations -- both are checked so the
#: manuscript shows the difference rather than inheriting the proposal's conflation of the two.
EXPECTED_SOURCE_TOTALS = {"OS": 377, "TTNT-D": 600}
#: Patients whose delayed entry falls at or after their OS exit, and who therefore cannot enter the
#: OS analysis at all: 678 - 644.
EXPECTED_ENTRY_AT_OR_AFTER_OS_EXIT = 34
EXPECTED_SOURCE_ENDPOINTS = {
    "DFCI": {"patients": 246, "os_events": 141, "ttnt_d_events": 212},
    "MSK": {"patients": 337, "os_events": 179, "ttnt_d_events": 301},
    "VICC": {"patients": 95, "os_events": 57, "ttnt_d_events": 87},
}
#: The delayed-entry-eligible analysis populations (tech-stack.md section 5, audit finding B).
EXPECTED_ELIGIBLE = {
    "OS": {
        "total": 644,
        "events": 361,
        "DFCI": (227, 129),
        "MSK": (330, 178),
        "VICC": (87, 54),
    },
    "TTNT-D": {
        "total": 338,
        "events": 264,
        "DFCI": (126, 95),
        "MSK": (180, 145),
        "VICC": (32, 24),
    },
}

ENDPOINTS = ("OS", "TTNT-D")


@dataclass
class ReconciliationResult:
    """Every check the cohort flow made, kept whether it passed or failed."""

    checks: list[dict] = field(default_factory=list)

    def record(self, name: str, expected: int, observed: int) -> None:
        self.checks.append(
            {
                "check": name,
                "expected": int(expected),
                "observed": int(observed),
                "agrees": bool(int(expected) == int(observed)),
            }
        )

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["agrees"]]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)


def _endpoint_columns(endpoint: str) -> tuple[str, str, str]:
    if endpoint == "OS":
        return "os_eligible", "os_event", "OS"
    if endpoint == "TTNT-D":
        return "ttnt_d_eligible", "ttnt_d_event", "TTNT-D"
    raise ReportContractError(f"{endpoint!r} is not a study endpoint")


def build_cohort_flow(inputs: CanonicalInputs) -> tuple[pd.DataFrame, ReconciliationResult]:
    """Compute the release-to-analysis-population flow and reconcile it against the frozen spec."""

    dataset = inputs.frame("phase2_dataset", "analysis_dataset.csv")
    reconciliation = ReconciliationResult()
    rows: list[dict] = []

    # Steps 1-4 are release-level counts that this dataset (already filtered to 678) cannot
    # recompute. They are carried from the frozen specification and labelled as such, rather than
    # silently presented as though manuscript-reporting workflow had derived them.
    for label, count in COHORT_FLOW_STEPS[:-1]:
        rows.append(
            {
                "step": label,
                "scope": "release",
                "patients": count,
                "DFCI": pd.NA,
                "MSK": pd.NA,
                "VICC": pd.NA,
                "provenance": "tech-stack.md section 5 (frozen constant)",
            }
        )

    observed_total = int(len(dataset))
    reconciliation.record("cohort_total", EXPECTED_COHORT_TOTAL, observed_total)
    per_centre = dataset["institution"].astype(str).value_counts()
    for centre in INSTITUTIONS:
        reconciliation.record(
            f"cohort_patients_{centre}",
            EXPECTED_SOURCE_ENDPOINTS[centre]["patients"],
            int(per_centre.get(centre, 0)),
        )
    rows.append(
        {
            "step": COHORT_FLOW_STEPS[-1][0],
            "scope": "analysis_cohort",
            "patients": observed_total,
            **{c: int(per_centre.get(c, 0)) for c in INSTITUTIONS},
            "provenance": "computed by Phase 8 from the frozen Phase 2 dataset",
        }
    )

    # Patients excluded from the OS analysis by delayed entry alone. Checked explicitly because it
    # is the single number that explains why the analysis population is not the source cohort.
    os_eligible_mask = dataset["os_eligible"].astype(str).str.lower().eq("true")
    reconciliation.record(
        "entry_at_or_after_os_exit",
        EXPECTED_ENTRY_AT_OR_AFTER_OS_EXIT,
        int((~os_eligible_mask).sum()),
    )

    # Phenotype split.
    biologic = dataset["biologic_with_folfox"].astype(str).str.lower().eq("true")
    reconciliation.record(
        "exact_folfox", EXPECTED_PHENOTYPE["exact_folfox"], int((~biologic).sum())
    )
    reconciliation.record(
        "folfox_plus_biologic", EXPECTED_PHENOTYPE["folfox_plus_biologic"], int(biologic.sum())
    )

    # Source-cohort endpoint counts, then the delayed-entry-eligible populations. Both appear, in
    # that order, because the difference between them is exactly what the proposal's Table 1 got
    # wrong and what a reader of the manuscript must be able to see.
    for endpoint in ENDPOINTS:
        eligible_col, event_col, label = _endpoint_columns(endpoint)
        reconciliation.record(
            f"source_{label}_events_total",
            EXPECTED_SOURCE_TOTALS[label],
            int(dataset[event_col].sum()),
        )
        for centre in INSTITUTIONS:
            subset = dataset[dataset["institution"].astype(str) == centre]
            key = "os_events" if endpoint == "OS" else "ttnt_d_events"
            reconciliation.record(
                f"source_{label}_events_{centre}",
                EXPECTED_SOURCE_ENDPOINTS[centre][key],
                int(subset[event_col].sum()),
            )
        rows.append(
            {
                "step": f"Source cohort, {label} events",
                "scope": "source_cohort",
                "patients": observed_total,
                **{
                    c: int(dataset[dataset["institution"].astype(str) == c][event_col].sum())
                    for c in INSTITUTIONS
                },
                "provenance": "computed by Phase 8; events, not patients",
            }
        )

        eligible = dataset[dataset[eligible_col].astype(str).str.lower().eq("true")]
        expected = EXPECTED_ELIGIBLE[label]
        reconciliation.record(f"eligible_{label}_total", expected["total"], int(len(eligible)))
        reconciliation.record(
            f"eligible_{label}_events", expected["events"], int(eligible[event_col].sum())
        )
        for centre in INSTITUTIONS:
            subset = eligible[eligible["institution"].astype(str) == centre]
            exp_patients, exp_events = expected[centre]
            reconciliation.record(f"eligible_{label}_patients_{centre}", exp_patients, len(subset))
            reconciliation.record(
                f"eligible_{label}_events_{centre}", exp_events, int(subset[event_col].sum())
            )
        rows.append(
            {
                "step": f"Delayed-entry eligible, {label}",
                "scope": "analysis_population",
                "patients": int(len(eligible)),
                **{c: int((eligible["institution"].astype(str) == c).sum()) for c in INSTITUTIONS},
                "provenance": "computed by Phase 8; the population that enters modelling",
            }
        )

    flow = pd.DataFrame(rows)
    return flow, reconciliation


def assert_cohort_flow_reconciles(reconciliation: ReconciliationResult) -> None:
    """Fail the build on any mismatch against `tech-stack.md` section 5."""

    failures = reconciliation.failures
    if failures:
        detail = "; ".join(
            f"{f['check']}: expected {f['expected']}, observed {f['observed']}" for f in failures
        )
        raise ReportContractError(
            "cohort flow does not reconcile with tech-stack.md section 5 -- the manuscript may not "
            f"be built from a cohort that disagrees with the frozen constants: {detail}"
        )


# --- Table 1 -------------------------------------------------------------------------------------

#: Core predictors that are summarized as categories rather than as a continuous distribution.
_CONTINUOUS = {"age_at_folfox"}
_COUNT_LIKE = {"number_metastatic_sites"}


def _category_rows(dataset: pd.DataFrame, column: str, centres: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    series = dataset[column]
    levels = sorted({str(v) for v in series.dropna().unique()})
    for level in levels:
        row: dict = {"variable": column, "level": level, "summary": "n (%)"}
        for centre in (*centres, "Total"):
            subset = dataset if centre == "Total" else dataset[dataset["institution"] == centre]
            denominator = len(subset)
            count = int((subset[column].astype(str) == level).sum())
            pct = (100.0 * count / denominator) if denominator else float("nan")
            row[centre] = f"{count} ({pct:.1f}%)"
        rows.append(row)
    missing = int(series.isna().sum())
    if missing:
        row = {"variable": column, "level": "Missing", "summary": "n (%)"}
        for centre in (*centres, "Total"):
            subset = dataset if centre == "Total" else dataset[dataset["institution"] == centre]
            denominator = len(subset)
            count = int(subset[column].isna().sum())
            pct = (100.0 * count / denominator) if denominator else float("nan")
            row[centre] = f"{count} ({pct:.1f}%)"
        rows.append(row)
    return rows


def _continuous_row(dataset: pd.DataFrame, column: str, centres: tuple[str, ...]) -> list[dict]:
    row: dict = {"variable": column, "level": "median (IQR)", "summary": "median (IQR)"}
    for centre in (*centres, "Total"):
        subset = dataset if centre == "Total" else dataset[dataset["institution"] == centre]
        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        if values.empty:
            row[centre] = "not computable"
            continue
        row[centre] = (
            f"{values.median():.1f} ({values.quantile(0.25):.1f}-{values.quantile(0.75):.1f})"
        )
    return [row]


def build_table1(inputs: CanonicalInputs) -> pd.DataFrame:
    """Baseline characteristics by institution over the 15 core predictors, with no p-values."""

    dataset = inputs.frame("phase2_dataset", "analysis_dataset.csv")
    dataset["institution"] = dataset["institution"].astype(str)
    centres = INSTITUTIONS

    rows: list[dict] = []
    header: dict = {"variable": "Patients", "level": "n", "summary": "n"}
    for centre in (*centres, "Total"):
        subset = dataset if centre == "Total" else dataset[dataset["institution"] == centre]
        header[centre] = str(len(subset))
    rows.append(header)

    for column in LOCKED_CORE:
        if column not in dataset.columns:
            raise ReportContractError(
                f"core predictor {column!r} is absent from the Phase 2 dataset"
            )
        if column in _CONTINUOUS:
            rows.extend(_continuous_row(dataset, column, centres))
        elif column in _COUNT_LIKE:
            rows.extend(_continuous_row(dataset, column, centres))
            rows.extend(_category_rows(dataset, column, centres))
        else:
            rows.extend(_category_rows(dataset, column, centres))

    # The delayed-entry-eligible rows required by manuscript-spec.md section 5's Table 1 entry, so
    # the analysis populations are visible beside the source cohort rather than only in the flow.
    for endpoint in ENDPOINTS:
        eligible_col, event_col, label = _endpoint_columns(endpoint)
        eligible = dataset[dataset[eligible_col].astype(str).str.lower().eq("true")]
        patients_row: dict = {
            "variable": f"Delayed-entry eligible ({label})",
            "level": "patients",
            "summary": "n",
        }
        events_row: dict = {
            "variable": f"Delayed-entry eligible ({label})",
            "level": "events",
            "summary": "n",
        }
        for centre in (*centres, "Total"):
            subset = eligible if centre == "Total" else eligible[eligible["institution"] == centre]
            patients_row[centre] = str(len(subset))
            events_row[centre] = str(int(subset[event_col].sum()))
        rows.append(patients_row)
        rows.append(events_row)

    table = pd.DataFrame(rows)
    assert_descriptive_and_aggregate(table, "Table 1")
    return table


#: Columns that would make a published frame patient-level rather than aggregate.
_PATIENT_LEVEL_COLUMNS = {
    "patient_key",
    "patient_id",
    "record_id",
    "entry",
    "os_time",
    "ttnt_d_time",
}


def assert_descriptive_and_aggregate(frame: pd.DataFrame, name: str) -> None:
    """Refuse a published frame that carries a p-value or any patient-level column.

    Both checks are assertions rather than conventions because both failures are invisible in a
    rendered table: a stray p-value invites a descriptive summary to be read as a hypothesis test,
    and a patient-level column is a privacy breach that looks like an ordinary extra column.
    """

    lowered = {str(c).lower() for c in frame.columns}
    if {"p", "p_value", "pvalue", "p value"} & lowered:
        raise ReportContractError(f"{name} must carry no cross-centre p-value")
    leaked = _PATIENT_LEVEL_COLUMNS & lowered
    if leaked:
        raise ReportContractError(f"{name} carries patient-level column(s): {sorted(leaked)}")


def build_manuscript_facts(
    inputs: CanonicalInputs,
    cohort_flow: pd.DataFrame,
    table1: pd.DataFrame,
) -> pd.DataFrame:
    """Index already-published quantities for transcription-safe prose tokens.

    This is a provenance index, not a third analytic result frame: each row points to one of the
    two permitted manuscript-reporting workflow frames, a frozen manifest, or a published aggregate
    primary-evaluation workflow-9 frame.
    """

    rows: list[dict] = []

    def add(fact: str, value: float, source: str, *, metric: str = "") -> None:
        rows.append(
            {
                "fact": fact,
                "kind": "metric" if metric else "count",
                "metric": metric,
                "value": float(value),
                "source_ref": source,
            }
        )

    dataset = inputs.frame("phase2_dataset", "analysis_dataset.csv")
    biologic = dataset["biologic_with_folfox"].astype(str).str.lower().eq("true")
    add("cohort_total", len(dataset), "phase8/table1_baseline.csv: Patients/Total")
    add("genomic_available", 41, "tech-stack.md section 5: pre-FOLFOX genomic availability")
    add("exact_folfox", (~biologic).sum(), "phase8/table1_baseline.csv: biologic=False")
    add("folfox_plus_biologic", biologic.sum(), "phase8/table1_baseline.csv: biologic=True")
    add("median_entry_days", dataset["entry"].median(), "phase2_dataset: entry median")

    flow = cohort_flow.set_index("step")
    centres = ["DFCI", "MSK", "VICC"]
    add(
        "source_os_events",
        flow.loc["Source cohort, OS events", centres].sum(),
        "phase8/cohort_flow.csv: source OS events",
    )
    add(
        "source_ttntd_events",
        flow.loc["Source cohort, TTNT-D events", centres].sum(),
        "phase8/cohort_flow.csv: source TTNT-D events",
    )

    def table_count(variable: str, level: str, column: str = "Total") -> int:
        match = table1[(table1["variable"] == variable) & (table1["level"] == level)]
        if len(match) != 1:
            raise ReportContractError(f"Table 1 fact {variable!r}/{level!r} is not unique")
        return int(str(match.iloc[0][column]).split()[0])

    for endpoint, key in (("OS", "os"), ("TTNT-D", "ttntd")):
        variable = f"Delayed-entry eligible ({endpoint})"
        add(
            f"{key}_eligible",
            table_count(variable, "patients"),
            f"phase8/table1_baseline.csv: {endpoint}/patients/Total",
        )
        add(
            f"{key}_eligible_events",
            table_count(variable, "events"),
            f"phase8/table1_baseline.csv: {endpoint}/events/Total",
        )
    add(
        "ttntd_vicc_patients",
        table_count("Delayed-entry eligible (TTNT-D)", "patients", "VICC"),
        "phase8/table1_baseline.csv: TTNT-D/patients/VICC",
    )
    add(
        "ttntd_vicc_events",
        table_count("Delayed-entry eligible (TTNT-D)", "events", "VICC"),
        "phase8/table1_baseline.csv: TTNT-D/events/VICC",
    )
    # Sociodemographic cohort counts for the TRIPOD+AI fairness statement. Both are indexed from
    # the already-published Table 1 race/ethnicity rows; the corresponding OS-analysis event
    # counts come from the fairness bundle's `subgroup` namespace, not from here.
    add(
        "nhb_cohort",
        table_count("race_ethnicity", "Non-Hispanic Black"),
        "phase8/table1_baseline.csv: race_ethnicity/Non-Hispanic Black/Total",
    )
    add(
        "hispanic_cohort",
        table_count("race_ethnicity", "Hispanic/Latinx"),
        "phase8/table1_baseline.csv: race_ethnicity/Hispanic/Latinx/Total",
    )

    coefficients = inputs.frame("phase6_explainability", "coefficient_summary.csv")

    def add_hr_facts(
        stem: str,
        *,
        centre: str,
        model: str,
        feature_group: str,
        expected_provenance: str,
    ) -> None:
        match = coefficients[
            (coefficients["endpoint"] == "OS")
            & (coefficients["centre"] == centre)
            & (coefficients["model"] == model)
            & (coefficients["feature_group"] == feature_group)
        ]
        if len(match) != 1:
            raise ReportContractError(
                "coefficient fact is not unique: "
                f"{centre}/{model}/{feature_group} matched {len(match)} rows"
            )
        row = match.iloc[0]
        if row["interval_provenance"] != expected_provenance:
            raise ReportContractError(
                "coefficient interval provenance changed for "
                f"{centre}/{model}/{feature_group}: {row['interval_provenance']!r}"
            )
        source = (
            "phase6 coefficient_summary.csv: "
            f"OS/{centre}/{model}/{feature_group}; CI exponentiated to hazard-ratio scale"
        )
        add(f"{stem}_hr", row["hazard_ratio"], source, metric="slope")
        add(f"{stem}_hr_lower", math.exp(float(row["ci_lower"])), source, metric="slope")
        add(f"{stem}_hr_upper", math.exp(float(row["ci_upper"])), source, metric="slope")

    for centre, centre_key in (("DFCI", "dfci"), ("MSK", "msk")):
        add_hr_facts(
            f"minimal_{centre_key}_de_novo",
            centre=centre,
            model="minimal_cox",
            feature_group="metastatic_presentation",
            expected_provenance="model_based",
        )
        add_hr_facts(
            f"minimal_{centre_key}_site_count",
            centre=centre,
            model="minimal_cox",
            feature_group="number_metastatic_sites",
            expected_provenance="model_based",
        )

    for centre, centre_key in (("DFCI", "dfci"), ("MSK", "msk"), ("VICC", "vicc")):
        add_hr_facts(
            f"spline_{centre_key}_abdomen",
            centre=centre,
            model="spline_cox",
            feature_group="dmets_abdomen",
            expected_provenance="bootstrap_refit",
        )

    stability = inputs.frame("phase6_explainability", "prediction_stability.csv")
    minimal_730 = stability[
        (stability["endpoint"] == "OS")
        & (stability["model"] == "minimal_cox")
        & (stability["horizon"] == 730.0)
    ]
    if sorted(minimal_730["centre"]) != ["DFCI", "MSK", "VICC"]:
        raise ReportContractError("prediction stability must cover exactly the three centres")
    stability_source = "phase6 prediction_stability.csv: OS/minimal_cox/730 across centres"
    add(
        "stability_os730_width_min",
        minimal_730["median_interval_width"].min(),
        stability_source,
        metric="risk_width",
    )
    add(
        "stability_os730_width_max",
        minimal_730["median_interval_width"].max(),
        stability_source,
        metric="risk_width",
    )
    add(
        "stability_os730_exceed020_min",
        minimal_730["proportion_exceeding_020"].min() * 100.0,
        stability_source,
        metric="percentage",
    )
    add(
        "stability_os730_exceed020_max",
        minimal_730["proportion_exceeding_020"].max() * 100.0,
        stability_source,
        metric="percentage",
    )

    spread = inputs.frame("phase6_explainability", "cross_family_prediction_spread.csv")
    spread_730 = spread[(spread["endpoint"] == "OS") & (spread["horizon"] == 730.0)]
    if sorted(spread_730["centre"]) != ["DFCI", "MSK", "VICC"]:
        raise ReportContractError("cross-family spread must cover exactly the three centres")
    if not (spread_730["model_family_status"] == "complete").all():
        raise ReportContractError(
            "cross-family spread facts require complete model-family coverage in every centre"
        )
    spread_source = "phase6 cross_family_prediction_spread.csv: OS/730 across centres"
    add(
        "spread_os730_median_min",
        spread_730["median_spread"].min(),
        spread_source,
        metric="risk_width",
    )
    add(
        "spread_os730_median_max",
        spread_730["median_spread"].max(),
        spread_source,
        metric="risk_width",
    )

    support = inputs.frame("phase5_evaluation", "entry_support_summary.csv")
    dca = support[(support["endpoint"] == "OS") & (support["day"] == 84)]
    if (dca["group"] == "All").any():
        dca = dca[dca["group"] == "All"]
    add("deaths_by_84_days", dca.iloc[0]["n_events"], "phase5 entry support: OS/day=84")

    primary_value = inputs.frame("phase5_evaluation", "complexity_value_summary.csv")
    primary_cells = (
        primary_value[primary_value["endpoint"] == "OS"]
        .groupby(["design", "centre", "fuller_model"], dropna=False)["classification"]
        .first()
    )
    add("phase5_os_complexity_cells", len(primary_cells), "phase5 complexity-value summary")
    add(
        "phase5_os_clear_added",
        (primary_cells == "clear_added_value").sum(),
        "phase5 complexity-value summary",
    )

    phase4_fit = inputs.frame("phase4_training", "fit_summary.csv")
    for model, stem in (("core_cox", "core_cox"), ("elastic_net_cox", "elastic_net")):
        subset = phase4_fit[phase4_fit["model_name"] == model]
        add(f"{stem}_fit_total", len(subset), f"phase4 fit summary: {model}")
        add(
            f"{stem}_fit_failed_before",
            (subset["fit_status"] == "fit_failed").sum(),
            f"phase4 fit summary: {model}",
        )

    correction_manifest = json.loads(
        inputs.path("phase5a_correction", "evaluation_manifest.json").read_text()
    )
    correction_fit_path = inputs.path(
        "phase5a_correction", "../model_training/fit_summary.csv"
    ).resolve()
    expected_fit_hash = correction_manifest["input_hashes"]["fit_summary.csv"]
    if sha256_of(correction_fit_path) != expected_fit_hash:
        raise ReportContractError("Phase 5A correction fit summary does not match its manifest")
    correction_fit = pd.read_csv(correction_fit_path)
    corrected_enet = correction_fit[correction_fit["model_name"] == "elastic_net_cox"]
    add(
        "elastic_net_fit_failed_after",
        (corrected_enet["fit_status"] == "fit_failed").sum(),
        "phase5a correction fit summary verified by evaluation manifest",
    )

    optimization = inputs.frame("phase5a_optimization", "complexity_value_summary.csv")
    opt_cells = (
        optimization[(optimization["endpoint"] == "OS") & (optimization["design"] == "iecv")]
        .groupby(["centre", "fuller_model"], dropna=False)["classification"]
        .first()
    )
    opt_models = opt_cells.groupby(level="fuller_model").apply(
        lambda values: bool((values == "clear_added_value").any())
    )
    add("phase5a_os_candidates", len(opt_models), "phase5a optimization complexity summary")
    add(
        "phase5a_os_candidates_clear_any",
        opt_models.sum(),
        "phase5a optimization complexity summary",
    )

    phase9_manifest = json.loads(inputs.path("phase9_forest", "phase9_manifest.json").read_text())
    add("phase9_scopes", phase9_manifest["scopes"], "phase9 manifest: scopes")
    add("phase9_fit_succeeded", phase9_manifest["fit_succeeded"], "phase9 manifest: fit_succeeded")
    add("phase9_trees", phase9_manifest["ntree"], "phase9 manifest: ntree")
    add(
        "bootstrap_resamples",
        phase9_manifest["evaluation"]["bootstrap_resamples"],
        "phase9 manifest: evaluation.bootstrap_resamples",
    )

    phase9_value = inputs.frame("phase9_forest", "complexity_value_summary.csv")
    iecv9 = phase9_value[phase9_value["design"] == "iecv"]
    zero9 = (iecv9["ci_lower"] <= 0) & (iecv9["ci_upper"] >= 0)
    add("phase9_iecv_gain_intervals", len(iecv9), "phase9 complexity summary: IECV")
    add("phase9_iecv_gain_intervals_contain_zero", zero9.sum(), "phase9 complexity summary: IECV")
    centre_classes = iecv9.groupby("centre")["classification"].first()
    add("phase9_iecv_centres", len(centre_classes), "phase9 complexity summary: IECV")
    add(
        "phase9_iecv_centres_no_advantage",
        (centre_classes == "no_demonstrated_advantage").sum(),
        "phase9 complexity summary: IECV",
    )

    performance9 = inputs.frame("phase9_forest", "performance_summary.csv")
    slopes = performance9[
        (performance9["model"] == "ltrc_cif")
        & (performance9["metric"] == "slope")
        & (performance9["risk_scale"] == "unconditional")
    ]
    add("phase9_forest_slope_intervals", len(slopes), "phase9 performance: forest slopes")
    add(
        "phase9_forest_slope_intervals_contain_one",
        ((slopes["ci_lower"] <= 1) & (slopes["ci_upper"] >= 1)).sum(),
        "phase9 performance: forest slopes",
    )
    calibration = iecv9[iecv9["gain"].isin(["slope", "citl"])]
    add("phase9_calibration_gain_intervals", len(calibration), "phase9 complexity: calibration")
    add(
        "phase9_calibration_gain_intervals_contain_zero",
        ((calibration["ci_lower"] <= 0) & (calibration["ci_upper"] >= 0)).sum(),
        "phase9 complexity: calibration",
    )

    agreement = inputs.frame("phase9_forest", "importance_agreement.csv")
    family = agreement[agreement["role"].isin(["primary", "supporting"])]
    add("phase9_rank_comparisons", len(family), "phase9 importance agreement")
    add(
        "phase9_rank_tau_min",
        family["kendall_tau"].min(),
        "phase9 importance agreement",
        metric="tau",
    )
    add(
        "phase9_rank_tau_median",
        family["kendall_tau"].median(),
        "phase9 importance agreement",
        metric="tau",
    )
    add(
        "phase9_rank_tau_max",
        family["kendall_tau"].max(),
        "phase9 importance agreement",
        metric="tau",
    )

    complete9 = inputs.frame("phase9_forest", "complete_case_assessment_subset.csv")
    complete9 = complete9[complete9["design"] == "iecv"]
    add("phase9_complete_case_differences", len(complete9), "phase9 complete-case: IECV")
    add(
        "phase9_complete_case_differences_contain_zero",
        ((complete9["diff_ci_lower"] <= 0) & (complete9["diff_ci_upper"] >= 0)).sum(),
        "phase9 complete-case: IECV",
    )

    forest_scaled = inputs.frame("phase9_forest", "scaled_brier.csv")["scaled_brier"]
    cox_scaled = inputs.frame("phase7_sensitivity", "scaled_brier.csv")
    cox_scaled = cox_scaled[
        (cox_scaled["endpoint"] == "OS") & (cox_scaled["model"] == "minimal_cox")
    ]["scaled_brier"]
    scaled = pd.concat([forest_scaled, cox_scaled], ignore_index=True)
    add("scaled_brier_min", scaled.min(), "phase7 and phase9 scaled Brier", metric="scaled_brier")
    add("scaled_brier_max", scaled.max(), "phase7 and phase9 scaled Brier", metric="scaled_brier")
    add("scaled_brier_below_zero", (scaled < 0).sum(), "phase7 and phase9 scaled Brier")

    facts = pd.DataFrame(rows).sort_values("fact").reset_index(drop=True)
    assert_descriptive_and_aggregate(facts, "Manuscript facts index")
    return facts

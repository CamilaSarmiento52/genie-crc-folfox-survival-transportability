"""Manuscript-reporting workflow's single orchestration boundary: verify inputs, compute frames,
render displays, build.

The notebook and every test call this module rather than reassembling the steps themselves, so
there is one path from frozen bundles to manuscript artifacts and no second, subtly different one.

Ordering is a contract, not a convenience. Verification happens before any value is read; the
cohort-flow reconciliation happens before any display is rendered; and the token resolution that
produces the manuscript happens last, so a stale bundle fails the build before it can reach prose.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from crc_folfox.report.assembly import render_docx, render_pdf
from crc_folfox.report.contracts import (
    ABSTRACT_WORD_CEILING,
    BODY_WORD_CEILING,
    BODY_WORD_MINIMUM,
    BODY_WORD_SECTIONS,
    MANUSCRIPT_OUTPUT_DIRNAME,
    MANUSCRIPT_SECTION_FILES,
    MAX_COMBINED_DISPLAYS,
    PLACEHOLDER_PATTERN,
    TOKEN_PATTERN,
    ReportContractError,
)
from crc_folfox.report.displays import (
    DisplayRecord,
    build_delayed_entry_km_frame,
    figure1_cohort_flow,
    figure2_survival,
    figure3_calibration,
    figure4_sequencing_lag,
    supplementary_attribution,
    supplementary_calibration,
    supplementary_minimal_cox_coefficients,
    supplementary_sex_stratified_performance,
    supplementary_sociodemographic_support,
    table2_primary_performance,
    table3_transportability,
    table4_complexity_value,
    table4_model_summary,
    to_markdown_table,
)
from crc_folfox.report.frames import (
    assert_cohort_flow_reconciles,
    build_cohort_flow,
    build_manuscript_facts,
    build_table1,
)
from crc_folfox.report.inputs import CanonicalInputs, verify_canonical_inputs
from crc_folfox.report.tokens import ResolvedToken, resolve_token, status_rule_violations


@dataclass
class BuildResult:
    """Everything one manuscript-reporting workflow build produced, including what it could not
    resolve."""

    output_root: str
    verifications: dict
    reconciliation_checks: int
    reconciliation_failures: int
    displays: list[DisplayRecord] = field(default_factory=list)
    resolved_tokens: list[ResolvedToken] = field(default_factory=list)
    unresolved_placeholders: list[str] = field(default_factory=list)
    word_counts: dict[str, int] = field(default_factory=dict)
    prose_present: bool = False
    status_violations: list[str] = field(default_factory=list)
    docx_path: str = ""
    pdf_path: str = ""
    pdf_unavailable_reason: str = ""

    @property
    def body_words(self) -> int:
        return sum(self.word_counts.get(name, 0) for name in BODY_WORD_SECTIONS)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _count_words(markdown: str) -> int:
    """Count body words under the frozen counting rule: prose only, no tokens, no headings."""

    lines = [
        line for line in markdown.splitlines() if not line.strip().startswith(("#", "|", ">", "!["))
    ]
    text = TOKEN_PATTERN.sub("X", "\n".join(lines))
    return len([w for w in text.split() if any(ch.isalnum() for ch in w)])


def _software_versions() -> dict[str, str]:
    packages = ("crc-folfox", "pandas", "matplotlib", "python-docx", "lifelines")
    recorded = {"python": platform.python_version()}
    for package in packages:
        try:
            recorded[package] = version(package)
        except PackageNotFoundError:
            recorded[package] = "not installed as a distribution"
    return recorded


def _resolve_prose(
    source_dir: Path,
    inputs: CanonicalInputs,
    replacements: dict[str, str] | None = None,
) -> tuple[str, list[ResolvedToken], list[str], dict[str, int], list[str]]:
    """Resolve every token in every existing section file and assemble `manuscript.md`."""

    assembled: list[str] = []
    resolved: list[ResolvedToken] = []
    placeholders: list[str] = []
    counts: dict[str, int] = {}
    violations: list[str] = []

    for name in MANUSCRIPT_SECTION_FILES:
        path = source_dir / name
        if not path.exists():
            continue
        markdown = path.read_text()
        for marker, replacement in (replacements or {}).items():
            markdown = markdown.replace(marker, replacement)
        counts[name] = _count_words(markdown)

        section_resolved: list[ResolvedToken] = []
        for raw in TOKEN_PATTERN.findall(markdown):
            if PLACEHOLDER_PATTERN.fullmatch("{{" + raw + "}}"):
                placeholders.append(raw)
                continue
            token = resolve_token(raw, inputs)
            section_resolved.append(token)
            markdown = markdown.replace("{{" + raw + "}}", token.rendered)

        for paragraph in markdown.split("\n\n"):
            in_paragraph = [t for t in section_resolved if t.rendered in paragraph]
            violations.extend(status_rule_violations(paragraph, in_paragraph))

        resolved.extend(section_resolved)
        assembled.append(markdown.rstrip())

    return "\n\n".join(assembled), resolved, placeholders, counts, violations


def run_manuscript_build(
    current_root: str | Path = "analysis/out/current",
    source_dir: str | Path | None = None,
    *,
    render_displays: bool = True,
) -> BuildResult:
    """Run the whole manuscript-reporting workflow build. Safe to call before any prose exists."""

    inputs = verify_canonical_inputs(current_root)
    output_root = Path(current_root).expanduser().resolve() / MANUSCRIPT_OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    if source_dir is None:
        source_dir = Path(__file__).resolve().parents[3] / (
            "specs/phases/phase-8-manuscript-draft/manuscript"
        )
    source_dir = Path(source_dir)

    # WP2 -- computed frames, reconciled before anything downstream may use them.
    flow, reconciliation = build_cohort_flow(inputs)
    assert_cohort_flow_reconciles(reconciliation)
    _write_frame(flow, output_root / "cohort_flow.csv")
    _write_frame(reconciliation.frame(), output_root / "cohort_flow_reconciliation.csv")
    table1 = build_table1(inputs)
    _write_frame(table1, output_root / "table1_baseline.csv")

    needs_manuscript_facts = any(
        "{{fact_" in path.read_text()
        for name in MANUSCRIPT_SECTION_FILES
        if (path := source_dir / name).exists()
    )
    facts = (
        build_manuscript_facts(inputs, flow, table1)
        if needs_manuscript_facts
        else pd.DataFrame(columns=["fact", "value", "metric", "artifact", "source_ref"])
    )
    _write_frame(facts, output_root / "manuscript_facts.csv")
    inputs.register_generated_frame("manuscript_facts.csv", facts)

    # WP3 -- display items, each regenerated from its published frame.
    displays: list[DisplayRecord] = []
    table2 = pd.DataFrame()
    table3 = pd.DataFrame()
    table4 = pd.DataFrame()
    supplementary_coefficients = pd.DataFrame()
    supplementary_complexity = pd.DataFrame()
    if render_displays:
        figures_dir = output_root / "figures"
        km_frame = build_delayed_entry_km_frame(inputs)
        _write_frame(km_frame, output_root / "delayed_entry_km_curves.csv")
        displays.append(figure1_cohort_flow(flow, figures_dir))
        displays.append(figure2_survival(km_frame, figures_dir))
        displays.append(figure3_calibration(inputs, figures_dir))
        displays.append(figure4_sequencing_lag(inputs, figures_dir))
        supplementary_attribution(inputs, figures_dir)
        supplementary_calibration(
            inputs,
            figures_dir,
            endpoint="OS",
            model="spline_cox",
            item="Supplementary Figure S26",
        )
        supplementary_calibration(
            inputs,
            figures_dir,
            endpoint="TTNT-D",
            model="minimal_cox",
            item="Supplementary Figure S27",
        )

        table2 = table2_primary_performance(inputs)
        table3 = table3_transportability(inputs)
        table4 = table4_model_summary(inputs)
        supplementary_coefficients = supplementary_minimal_cox_coefficients(inputs)
        supplementary_complexity = table4_complexity_value(inputs)
        _write_frame(table2, output_root / "table2_primary_performance.csv")
        _write_frame(table3, output_root / "table3_transportability.csv")
        _write_frame(table4, output_root / "table4_model_summary.csv")
        _write_frame(
            supplementary_coefficients,
            output_root / "supplementary_minimal_cox_coefficients.csv",
        )
        _write_frame(
            supplementary_complexity,
            output_root / "supplementary_complexity_value_full.csv",
        )
        displays.extend(
            [
                DisplayRecord(
                    name="Table 1",
                    path=str(output_root / "table1_baseline.csv"),
                    source="analysis_dataset.csv plus reconciled endpoint populations",
                    restrictions=[
                        "descriptive baseline summaries only; no new performance analysis"
                    ],
                ),
                DisplayRecord(
                    name="Table 2",
                    path=str(output_root / "table2_primary_performance.csv"),
                    source="evaluation/performance_summary.csv",
                    restrictions=[
                        "primary models, held-out OS and TTNT-D, with reporting statuses kept "
                        "distinct"
                    ],
                ),
                DisplayRecord(
                    name="Table 3",
                    path=str(output_root / "table3_transportability.csv"),
                    source="evaluation/paired_comparisons.csv",
                    restrictions=["OS paired pooled-minus-IECV comparisons only"],
                ),
                DisplayRecord(
                    name="Table 4",
                    path=str(output_root / "table4_model_summary.csv"),
                    source=(
                        "explainability/coefficient_summary.csv + "
                        "evaluation/complexity_value_summary.csv"
                    ),
                    restrictions=[
                        "association panel is descriptive and non-causal",
                        "complexity panel gives one locked classification per model-centre cell",
                        "full coefficient and paired-gain tables are retained in the supplement",
                    ],
                ),
            ]
        )

        tables_markdown = "\n\n".join(
            [
                "## Table 2. Held-out OS and supportive TTNT-D performance",
                to_markdown_table(table2),
                "## Table 3. Transportability difference (pooled minus held-out), OS",
                to_markdown_table(table3),
                "## Table 4. Adjusted associations and classification of added complexity",
                to_markdown_table(table4),
                "## Supplementary Table S28. Full minimal-Cox coefficients",
                to_markdown_table(supplementary_coefficients),
                "## Supplementary Table S29. Full paired complexity-value results",
                to_markdown_table(supplementary_complexity),
            ]
        )
        (output_root / "tables.md").write_text(tables_markdown + "\n")

    # Revision-era fairness tables (Supplementary Table S30). Built whenever the post hoc
    # fairness bundle has been published by `scripts/generate_sex_stratified_performance.py`;
    # empty frames otherwise so a build without that bundle renders "(no rows)" rather than
    # failing on a table no prose token depends on.
    fairness_performance = pd.DataFrame()
    fairness_support = pd.DataFrame()
    if (inputs.bundle_root("fairness_subgroups") / "sex_stratified_performance.csv").exists():
        fairness_performance = supplementary_sex_stratified_performance(inputs)
        fairness_support = supplementary_sociodemographic_support(inputs)

    # WP1 -- token resolution over whatever prose exists yet.
    display_replacements = {
        "[[TABLE_1]]": to_markdown_table(table1),
        "[[TABLE_2]]": to_markdown_table(table2),
        "[[TABLE_3]]": to_markdown_table(table3),
        "[[TABLE_4]]": to_markdown_table(table4),
        "[[SUPP_HR_TABLE]]": to_markdown_table(supplementary_coefficients),
        "[[SUPP_COMPLEXITY_TABLE]]": to_markdown_table(supplementary_complexity),
        "[[SUPP_SEX_TABLE]]": to_markdown_table(fairness_performance),
        "[[SUPP_SUBGROUP_SUPPORT_TABLE]]": to_markdown_table(fairness_support),
    }
    references_path = source_dir / "references.yaml"
    if references_path.exists():
        references = json.loads(references_path.read_text())
        display_replacements["[[REFERENCES]]"] = "\n\n".join(
            f"{item['id']}. {item['citation']}" for item in references
        )
    assembled, resolved, placeholders, counts, violations = _resolve_prose(
        source_dir,
        inputs,
        replacements=display_replacements,
    )
    prose_present = bool(counts)
    rendered_docx = ""
    rendered_pdf = ""
    pdf_unavailable_reason = ""

    if prose_present:
        (output_root / "manuscript.md").write_text(assembled + "\n")
        _write_frame(
            pd.DataFrame(
                [
                    {
                        "token": t.token,
                        "rendered": t.rendered,
                        "raw_value": t.raw_value,
                        "artifact": t.artifact,
                        "source_cell": t.source_cell,
                        "reporting_status": t.status,
                    }
                    for t in resolved
                ]
            ),
            output_root / "resolved_values.csv",
        )
        _write_frame(
            pd.DataFrame(sorted(counts.items()), columns=["section", "words"]),
            output_root / "word_counts.csv",
        )
        docx_path = render_docx(assembled, output_root / "manuscript.docx")
        pdf_path, pdf_reason = render_pdf(docx_path)
        rendered_docx = str(docx_path)
        rendered_pdf = str(pdf_path) if pdf_path else ""
        pdf_unavailable_reason = pdf_reason

    result = BuildResult(
        output_root=str(output_root),
        verifications={k: asdict(v) for k, v in inputs.verifications.items()},
        reconciliation_checks=len(reconciliation.checks),
        reconciliation_failures=len(reconciliation.failures),
        displays=displays,
        resolved_tokens=resolved,
        unresolved_placeholders=sorted(set(placeholders)),
        word_counts=counts,
        prose_present=prose_present,
        status_violations=violations,
        docx_path=rendered_docx,
        pdf_path=rendered_pdf,
        pdf_unavailable_reason=pdf_unavailable_reason,
    )

    all_body_sections_present = all((source_dir / name).exists() for name in BODY_WORD_SECTIONS)
    if all_body_sections_present and not (
        BODY_WORD_MINIMUM <= result.body_words <= BODY_WORD_CEILING
    ):
        raise ReportContractError(
            f"body word count {result.body_words} is outside the "
            f"{BODY_WORD_MINIMUM}-{BODY_WORD_CEILING} limit"
        )
    if "01_abstract.md" in counts and counts["01_abstract.md"] > ABSTRACT_WORD_CEILING:
        raise ReportContractError(
            f"abstract word count {counts['01_abstract.md']} exceeds {ABSTRACT_WORD_CEILING}"
        )
    if render_displays and len(displays) != MAX_COMBINED_DISPLAYS:
        raise ReportContractError(
            f"main manuscript requires exactly {MAX_COMBINED_DISPLAYS} displays; "
            f"got {len(displays)}"
        )

    manifest = {
        "phase": "phase-8-manuscript-draft",
        "input_verification": {
            name: {
                "status": v["status"],
                "hashes_checked": v["hashes_checked"],
                "hashes_matched": v["hashes_matched"],
                "baseline_recorded": v["baseline_recorded"],
                "reason": v["reason"],
            }
            for name, v in result.verifications.items()
        },
        "cohort_flow_reconciliation": {
            "checks": result.reconciliation_checks,
            "failures": result.reconciliation_failures,
        },
        "displays": [asdict(d) for d in displays],
        "prose_present": prose_present,
        "resolved_token_count": len(resolved),
        "resolved_token_inventory": [
            {
                "token": item.token,
                "artifact": item.artifact,
                "source_cell": item.source_cell,
                "reporting_status": item.status,
            }
            for item in resolved
        ],
        "unresolved_placeholders": result.unresolved_placeholders,
        "status_rule_violations": violations,
        "docx": result.docx_path,
        "pdf": result.pdf_path,
        "pdf_unavailable_reason": result.pdf_unavailable_reason,
        "word_counts": counts,
        "body_words": result.body_words,
        "body_word_budget": [BODY_WORD_MINIMUM, BODY_WORD_CEILING],
        "software_versions": _software_versions(),
    }
    (output_root / "manuscript_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if violations:
        raise ReportContractError(
            "reporting-status rules violated in prose: " + "; ".join(violations[:5])
        )
    return result

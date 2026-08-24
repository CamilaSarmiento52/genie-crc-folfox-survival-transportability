"""Render the assembled, token-resolved Markdown into the submission `.docx`.

Pandoc and LibreOffice are not available in this environment, so the `.docx` is produced directly
with `python-docx` and the `.pdf` is produced only when a converter is present. A missing PDF is
recorded as unavailable rather than silently skipped: the PDF is a review convenience copy, and
claiming to have produced one that does not exist would be worse than not producing it.

This renderer deliberately understands only the subset of Markdown the manuscript source uses --
headings, paragraphs, pipe tables, and image references. A renderer that quietly accepted syntax it
could not represent would drop content out of the submission file without failing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

_HEADING_PREFIXES = ("######", "#####", "####", "###", "##", "#")
_IMAGE_PATTERN = re.compile(r"^!\[(?P<alt>.*)\]\((?P<path>.+)\)$")
_NUMBERED_PATTERN = re.compile(r"^\d+\.\s+(?P<text>.+)$")


def _set_font(run, size: float, *, bold: bool = False, color: str = "000000") -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def _clean_inline(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def _shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_cell_margins(
    cell, *, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120
) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _table_widths(rows: list[list[str]], total: int = 9360) -> list[int]:
    columns = len(rows[0])
    maxima = [max(len(str(row[index])) for row in rows) for index in range(columns)]
    weights = [max(8.0, min(32.0, float(value) ** 0.72)) for value in maxima]
    widths = [max(640, round(total * weight / sum(weights))) for weight in weights]
    scale = total / sum(widths)
    widths = [round(width * scale) for width in widths]
    widths[-1] += total - sum(widths)
    return widths


def _apply_table_geometry(table, widths: list[int]) -> None:
    table.autofit = False
    table.allow_autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths[index] / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[index]))
            tc_w.set(qn("w:type"), "dxa")
            _set_cell_margins(cell)


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    _set_font(run, 9, color="666666")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    heading_tokens = {
        "Heading 1": (16, 16, 8),
        "Heading 2": (13, 12, 6),
        "Heading 3": (12, 8, 4),
        "Heading 4": (11, 8, 4),
    }
    for name, (size, before, after) in heading_tokens.items():
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    list_style = document.styles["List Number"]
    list_style.font.name = "Times New Roman"
    list_style.font.size = Pt(9.5)
    list_style.paragraph_format.left_indent = Inches(0.375)
    list_style.paragraph_format.first_line_indent = Inches(-0.194)
    list_style.paragraph_format.space_after = Pt(2)
    list_style.paragraph_format.line_spacing = 1.05

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_run = header.add_run("FOLFOX prediction transportability | AACR Project GENIE BPC")
    _set_font(header_run, 9, color="666666")
    _add_page_number(section.footer.paragraphs[0])


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_table_rule(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(set(c) <= {"-", ":", " "} and "-" in c for c in cells)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_docx(markdown: str, path: Path) -> Path:
    """Write a restrained journal-style `.docx` with figures and editable tables."""

    document = Document()
    _configure_document(document)

    lines = markdown.splitlines()
    index = 0
    paragraph_buffer: list[str] = []

    top_level_headings = 0
    title_page = True

    def flush_paragraph() -> None:
        if paragraph_buffer:
            text = _clean_inline(" ".join(paragraph_buffer).strip())
            numbered = _NUMBERED_PATTERN.match(text)
            paragraph = document.add_paragraph(
                numbered.group("text") if numbered else text,
                style="List Number" if numbered else None,
            )
            if title_page and not numbered:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = next((p for p in _HEADING_PREFIXES if stripped.startswith(p + " ")), None)
        if heading:
            flush_paragraph()
            level = min(len(heading), 4)
            heading_text = _clean_inline(stripped[len(heading) :].strip())
            if level == 1:
                paragraph = document.add_heading(heading_text, level=1)
                if top_level_headings and heading_text != "Keywords":
                    # Applying the break to the heading avoids the empty paragraph/page that an
                    # explicit page-break run can create when the preceding section ends exactly
                    # at a page boundary.
                    paragraph.paragraph_format.page_break_before = True
                top_level_headings += 1
                if top_level_headings == 1:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        _set_font(run, 18, bold=True)
                else:
                    title_page = False
            else:
                paragraph = document.add_heading(heading_text, level=level)
                if (
                    level == 2
                    and heading_text.startswith("Figure ")
                    and not heading_text.startswith("Figure 1.")
                ):
                    paragraph.paragraph_format.page_break_before = True
            index += 1
            continue

        if _is_table_row(stripped):
            flush_paragraph()
            block = []
            while index < len(lines) and _is_table_row(lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            rows = [_split_row(r) for r in block if not _is_table_rule(r)]
            if rows:
                table = document.add_table(rows=len(rows), cols=len(rows[0]))
                table.style = "Table Grid"
                widths = _table_widths(rows)
                for r_index, row in enumerate(rows):
                    for c_index, value in enumerate(row[: len(rows[0])]):
                        cell = table.cell(r_index, c_index)
                        cell.text = _clean_inline(value)
                        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                        cell.paragraphs[0].alignment = (
                            WD_ALIGN_PARAGRAPH.LEFT
                            if c_index == 0 or len(value) > 24
                            else WD_ALIGN_PARAGRAPH.CENTER
                        )
                        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
                        cell.paragraphs[0].paragraph_format.line_spacing = 1.0
                        for run in cell.paragraphs[0].runs:
                            _set_font(run, 7.5, bold=r_index == 0)
                        if r_index == 0:
                            _shade_cell(cell, "F4F6F9")
                header_properties = table.rows[0]._tr.get_or_add_trPr()
                repeat = OxmlElement("w:tblHeader")
                repeat.set(qn("w:val"), "true")
                header_properties.append(repeat)
                for row in table.rows:
                    row_properties = row._tr.get_or_add_trPr()
                    cant_split = OxmlElement("w:cantSplit")
                    cant_split.set(qn("w:val"), "true")
                    row_properties.append(cant_split)
                _apply_table_geometry(table, widths)
                document.add_paragraph()
            continue

        image_match = _IMAGE_PATTERN.match(stripped)
        if image_match:
            flush_paragraph()
            image_path = Path(image_match.group("path"))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"manuscript figure does not exist: {image_path}")
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if document.paragraphs[:-1]:
                document.paragraphs[-2].paragraph_format.keep_with_next = True
            run = paragraph.add_run()
            # Full text width (8.5 in page, 1 in margins). Editorial feedback 2026-08-22: the
            # earlier 5-6 in placements left the figures smaller than the page allows.
            inline_shape = run.add_picture(str(image_path), width=Inches(6.5))
            inline_shape._inline.docPr.set("descr", image_match.group("alt"))
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path


def render_pdf(docx_path: Path) -> tuple[Path | None, str]:
    """Convert to PDF when a converter exists; otherwise say plainly that none does."""

    converter = shutil.which("soffice") or shutil.which("libreoffice")
    if converter is None:
        return None, (
            "no PDF converter available (neither soffice nor libreoffice is installed); the "
            "review PDF is a convenience copy and its absence does not affect the submission .docx"
        )
    try:
        # Convert in an isolated profile/output directory. LibreOffice otherwise fails when a
        # stale desktop profile lock or an earlier review PDF exists at the destination.
        with tempfile.TemporaryDirectory(prefix="crc-manuscript-pdf-") as temporary:
            temporary_path = Path(temporary)
            profile_uri = (temporary_path / "libreoffice-profile").as_uri()
            subprocess.run(
                [
                    converter,
                    f"-env:UserInstallation={profile_uri}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temporary_path),
                    str(docx_path),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            temporary_pdf = temporary_path / docx_path.with_suffix(".pdf").name
            if not temporary_pdf.exists():
                return None, "converter produced no PDF"
            shutil.copyfile(temporary_pdf, docx_path.with_suffix(".pdf"))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return None, f"PDF conversion failed: {error}"
    produced = docx_path.with_suffix(".pdf")
    return (produced, "") if produced.exists() else (None, "converter produced no PDF")

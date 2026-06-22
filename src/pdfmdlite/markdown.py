from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from statistics import median
from typing import Any

from .layout import (
    Line,
    PageLayout,
    normalize_line_text,
    split_table_cells,
    touching_cjk,
)

LIST_RE = re.compile(
    r"^\s*((?:[-*+•◦▪●])|(?:\(?\d+[\).])|(?:\(?[A-Za-z][\).])|(?:[IVXLCM]+[\).]))\s+"
)

# A lone section label that sits to the left of a heading on the same baseline:
# a number ("1", "5.2") or an appendix letter ("A", "C.1"). Used both to merge
# the label into the heading text and to protect it from artifact suppression.
SECTION_NUMBER_RE = re.compile(r"^(?:\d+|[A-Z])(?:\.\d+)*$")

# A bare equation-number label such as "(1)", "12.", or "3" sitting on its
# own. These appear in the right margin of display equations and must never
# become a math block or leak as body text inside an equation run.
EQUATION_NUMBER_RE = re.compile(r"^\(?\d+\)?[.)]?$")


def render_markdown(
    pages: list[PageLayout],
    *,
    page_breaks: bool = False,
    detect_tables: bool = True,
    artifacts: list[Any] | None = None,
    embed_artifacts: bool = False,
    asset_base_dir: str | Path | None = None,
    math_results: dict[int, Any] | None = None,
    inline_images: bool = False,
) -> str:
    """Render pages to Markdown.

    Display equations are emitted as fenced ``math`` blocks of reconstructed
    LaTeX and inline math as ``$...$``; the source glyph lines are suppressed so
    nothing is emitted twice.
    """
    parts: list[str] = []
    document_median_height = _document_median_word_height(pages)
    artifacts_by_page = _artifacts_by_page(artifacts or []) if embed_artifacts else {}
    base_dir = Path(asset_base_dir) if asset_base_dir is not None else None

    for page in pages:
        ordered = order_lines(page)
        page_math = math_results.get(page.number) if math_results is not None else None
        page_markdown = _render_page(
            ordered,
            median_word_height=document_median_height or page.median_word_height or 10.0,
            detect_tables=detect_tables,
            artifacts=artifacts_by_page.get(page.number, []),
            asset_base_dir=base_dir,
            page_width=page.width,
            page_math=page_math,
            page_lines=page.lines,
            inline_images=inline_images,
        )
        if page_markdown:
            if page_breaks and parts:
                parts.append(f"<!-- page {page.number} -->")
            parts.append(page_markdown)

    return "\n\n".join(parts).strip() + "\n"


def order_lines(page: PageLayout) -> list[Line]:
    lines = [line for line in page.lines if line.text]
    if not lines:
        return []

    content_width = max((line.x_max for line in lines), default=page.width) - min(
        (line.x_min for line in lines), default=0
    )
    if content_width <= 0:
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))

    full_width_threshold = content_width * 0.68
    ordered: list[Line] = []
    segment: list[Line] = []

    for line in sorted(lines, key=lambda item: (item.y_min, item.x_min)):
        is_full_width = line.width >= full_width_threshold
        if is_full_width and segment:
            ordered.extend(_order_segment(segment, page))
            segment = []
        if is_full_width:
            ordered.append(line)
        else:
            segment.append(line)
    if segment:
        ordered.extend(_order_segment(segment, page))
    return ordered


def _order_segment(lines: list[Line], page: PageLayout) -> list[Line]:
    if len(lines) < 6:
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))

    x_values = sorted(line.x_center for line in lines)
    midpoint = x_values[len(x_values) // 2]
    left = [line for line in lines if line.x_center <= midpoint]
    right = [line for line in lines if line.x_center > midpoint]
    if min(len(left), len(right)) < 3:
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))
    if (
        median(line.width for line in left) < page.width * 0.22
        or median(line.width for line in right) < page.width * 0.22
    ):
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))

    left_x = median(line.x_min for line in left)
    right_x = median(line.x_min for line in right)
    if right_x - left_x < page.width * 0.20:
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))

    if not _has_vertical_overlap(left, right):
        return sorted(lines, key=lambda line: (line.y_min, line.x_min))

    return sorted(left, key=lambda line: (line.y_min, line.x_min)) + sorted(
        right, key=lambda line: (line.y_min, line.x_min)
    )


def _has_vertical_overlap(left: list[Line], right: list[Line]) -> bool:
    left_min = min(line.y_min for line in left)
    left_max = max(line.y_max for line in left)
    right_min = min(line.y_min for line in right)
    right_max = max(line.y_max for line in right)
    return min(left_max, right_max) - max(left_min, right_min) > 0


def _render_page(
    lines: list[Line],
    *,
    median_word_height: float,
    detect_tables: bool,
    artifacts: list[Any],
    asset_base_dir: Path | None,
    page_width: float = 0.0,
    page_math: Any | None = None,
    page_lines: list[Line] | None = None,
    inline_images: bool = False,
) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    paragraph_key: tuple[int, int, str] | None = None
    index = 0
    anchored_artifacts = _anchored_artifacts(artifacts)
    emitted_artifact_ids: set[str] = set()
    suppressed_line_indices = _suppressed_line_indices(lines, artifacts, median_word_height)

    # Display-math: map each region to its anchor line (the last source line in
    # reading order) and suppress every source line so glyph text is not emitted
    # twice. Indices in math_results reference the page's original line order, so
    # they are translated to positions in this reordered `lines` list by Line
    # identity.
    display_by_anchor: dict[int, list[str]] = {}
    math_suppressed: set[int] = set()
    inline_by_anchor: dict[int, list[Any]] = {}
    if page_math is not None and page_lines is not None:
        position_of = {id(line): pos for pos, line in enumerate(lines)}
        for region in getattr(page_math, "display", ()):  # type: ignore[union-attr]
            positions = [
                position_of[id(page_lines[i])]
                for i in region.source_line_indices
                if 0 <= i < len(page_lines) and id(page_lines[i]) in position_of
            ]
            if not positions:
                continue
            math_suppressed.update(positions)
            anchor = max(positions)
            display_by_anchor.setdefault(anchor, []).append(region.latex)
        for region in getattr(page_math, "inline", ()):  # type: ignore[union-attr]
            positions = [
                position_of[id(page_lines[i])]
                for i in region.source_line_indices
                if 0 <= i < len(page_lines) and id(page_lines[i]) in position_of
            ]
            if not positions:
                continue
            anchor = max(positions)
            inline_by_anchor.setdefault(anchor, []).append(region)

    # A line reconstructed as LaTeX or covered by an artifact must not be eaten
    # into a gap-detected table by the scanner below, which would drop its math
    # block / crop and leak its glyph fragments as bogus cells. The scanner
    # stops at any such line.
    table_blocked = frozenset(math_suppressed).union(suppressed_line_indices)

    def emit_display_math(line_index: int) -> None:
        for latex in display_by_anchor.get(line_index, []):
            flush_paragraph()
            blocks.append("```math\n" + latex + "\n```")

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_key
        if paragraph:
            blocks.append(_join_paragraph(paragraph))
            paragraph = []
            paragraph_key = None

    def emit_artifacts_for_line(line_index: int) -> None:
        for artifact in anchored_artifacts.get(line_index, []):
            artifact_id = _artifact_identity(artifact)
            if artifact_id in emitted_artifact_ids:
                continue
            snippet = _artifact_markdown(artifact, asset_base_dir, inline_images)
            if snippet:
                flush_paragraph()
                blocks.append(snippet)
                emitted_artifact_ids.add(artifact_id)

    while index < len(lines):
        if index in suppressed_line_indices:
            flush_paragraph()
            emit_display_math(index)
            emit_artifacts_for_line(index)
            index += 1
            continue

        # A display-equation source line is reconstructed as LaTeX, not emitted
        # as glyph text: suppress it and emit the fenced math block at its anchor.
        if index in math_suppressed:
            flush_paragraph()
            emit_display_math(index)
            emit_artifacts_for_line(index)
            index += 1
            continue

        if detect_tables:
            table, consumed = _try_render_table(lines, index, table_blocked)
            if table:
                flush_paragraph()
                blocks.append(table)
                for table_index in range(index, index + consumed):
                    emit_artifacts_for_line(table_index)
                index += consumed
                continue

        line = lines[index]
        text = line.text.strip()
        if not text:
            flush_paragraph()
            index += 1
            continue

        if _is_rotated_marginal_stamp(line, page_width, median_word_height):
            flush_paragraph()
            emit_artifacts_for_line(index)
            index += 1
            continue

        # A lone equation number parked in the far-right margin (x past ~78%
        # of the page) labels a display equation; it is never body text and
        # leaks as a stray "(1)" line once the equation itself is suppressed
        # or rendered as an image. Drop it. The x guard keeps real bare-number
        # body lines, which never sit that far right, intact.
        if (
            page_width > 0
            and EQUATION_NUMBER_RE.match(text)
            and line.x_min >= page_width * 0.78
        ):
            flush_paragraph()
            emit_artifacts_for_line(index)
            index += 1
            continue

        if _is_orphan_section_number(
            lines, index, median_word_height, suppressed_line_indices
        ):
            flush_paragraph()
            heading_level = _heading_level(lines[index + 1], median_word_height)
            heading_text, consumed = _collect_heading(
                lines, index + 1, median_word_height, page_width, suppressed_line_indices
            )
            blocks.append(f"{'#' * heading_level} {text} {heading_text}")
            for consumed_index in range(index, index + 1 + consumed):
                emit_artifacts_for_line(consumed_index)
            index += 1 + consumed
            continue

        heading_level = _heading_level(line, median_word_height)
        if heading_level:
            flush_paragraph()
            heading_text, consumed = _collect_heading(
                lines, index, median_word_height, page_width, suppressed_line_indices
            )
            blocks.append(f"{'#' * heading_level} {heading_text}")
            for consumed_index in range(index, index + consumed):
                emit_artifacts_for_line(consumed_index)
            index += consumed
            continue
        elif _is_list_item(text):
            flush_paragraph()
            blocks.append(_normalize_list_item(text))
        else:
            if paragraph_key is not None and line.block_key != paragraph_key:
                flush_paragraph()
            line_regions = inline_by_anchor.get(index)
            if line_regions:
                paragraph.append(_splice_inline_math(line, line_regions))
            else:
                paragraph.append(text)
            paragraph_key = line.block_key
        emit_artifacts_for_line(index)
        index += 1

    flush_paragraph()
    for artifact in artifacts:
        artifact_id = _artifact_identity(artifact)
        if artifact_id in emitted_artifact_ids:
            continue
        snippet = _artifact_markdown(artifact, asset_base_dir, inline_images)
        if snippet:
            blocks.append(snippet)
            emitted_artifact_ids.add(artifact_id)
    return "\n\n".join(block for block in blocks if block.strip())


def _heading_level(line: Line, median_word_height: float) -> int:
    text = line.text.strip()
    if not text or len(text) > 120 or _is_list_item(text) or _is_formula_line(text):
        return 0
    if text.endswith((".", "。", ":", "：", ";", "；", ",")):
        return 0
    # Curly braces only appear in set/math notation (e.g. "{xi}nvec"), never in a
    # prose heading. Such fragments otherwise get promoted by the height ratio.
    if "{" in text or "}" in text:
        return 0

    # A line of nothing but digits/punctuation (an axis label "0.5", a leaked
    # figure measure "0.0 0.2 0.4") is never a heading, whatever its size.
    if re.fullmatch(r"[\d.,%\-\s]+", text):
        return 0

    ratio = line.median_word_height / median_word_height if median_word_height else 1.0
    numbered_heading = re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", text) is not None
    short_title = len(text) <= 80 and len(text.split()) <= 12

    if ratio >= 1.65 and short_title:
        return 1
    if ratio >= 1.28 and short_title:
        return 2
    if numbered_heading and short_title:
        return 2
    if text.isupper() and short_title and len(text) > 3:
        return 2
    return 0


def _is_rotated_marginal_stamp(
    line: Line, page_width: float, median_word_height: float
) -> bool:
    if page_width <= 0 or median_word_height <= 0:
        return False
    x_center_ratio = line.x_center / page_width
    in_extreme_margin = x_center_ratio < 0.06 or x_center_ratio > 0.94
    if not in_extreme_margin:
        return False
    if line.median_word_height / median_word_height >= 2.0:
        return True
    return line.text.strip().startswith("arXiv:")


def _is_orphan_section_number(
    lines: list[Line],
    index: int,
    median_word_height: float,
    suppressed_line_indices: set[int],
) -> bool:
    line = lines[index]
    text = line.text.strip()
    if not SECTION_NUMBER_RE.match(text):
        return False
    next_index = index + 1
    if next_index >= len(lines) or next_index in suppressed_line_indices:
        return False
    next_line = lines[next_index]
    if not _heading_level(next_line, median_word_height):
        return False
    # The number must sit on the same baseline as the heading it labels and
    # to its left (a number that is inline/after text is not a section label).
    if abs(next_line.y_min - line.y_min) > 0.6 * median_word_height:
        return False
    if line.x_max > next_line.x_min + 2.0:
        return False
    return True


def _collect_heading(
    lines: list[Line],
    index: int,
    median_word_height: float,
    page_width: float,
    suppressed_line_indices: set[int],
) -> tuple[str, int]:
    first = lines[index]
    level = _heading_level(first, median_word_height)
    pieces = [first.text.strip()]
    align_tolerance = max(6.0, 0.05 * page_width)
    j = index + 1
    while j < len(lines):
        nxt = lines[j]
        if j in suppressed_line_indices:
            break
        next_text = nxt.text.strip()
        if not next_text:
            break
        if _heading_level(nxt, median_word_height) != level:
            break
        previous = lines[j - 1]
        gap = nxt.y_min - previous.y_min
        max_gap = 2.0 * max(previous.median_word_height, nxt.median_word_height)
        if gap < 0 or gap > max_gap:
            break
        if abs(nxt.x_min - first.x_min) > align_tolerance:
            break
        pieces.append(next_text)
        j += 1
    return _join_heading_pieces(pieces), j - index


def _join_heading_pieces(pieces: list[str]) -> str:
    result = pieces[0].strip()
    for piece in pieces[1:]:
        next_text = piece.strip()
        if not next_text:
            continue
        if result.endswith("-"):
            result = result[:-1] + next_text
            continue
        separator = "" if touching_cjk(result, next_text) else " "
        result += separator + next_text
    return result


def _is_formula_line(text: str) -> bool:
    if not text:
        return False
    if text.lower().startswith(("where ", "therefore ", "assuming ")):
        return False
    math_chars = set("=∑∏√≤≥≠≈∈∀∂∆△⊤⊥λγµσθΩωαβζ∞·±−")
    math_count = sum(1 for char in text if char in math_chars)
    if math_count == 0:
        return False
    if re.search(r"\b(Table|Figure|Fig)\b", text, re.IGNORECASE):
        return False
    if len(text) > 60 and len(text.split()) > 8:
        return False
    if len(text) > 120:
        return False
    tokens = text.split()
    alpha_chars = sum(1 for char in text if char.isalpha())
    digit_chars = sum(1 for char in text if char.isdigit())
    symbolish = math_count + digit_chars
    if len(text) <= 90 and (symbolish >= 2 or math_count >= 2):
        return True
    if len(tokens) <= 8 and math_count >= 1 and alpha_chars <= 28:
        return True
    return False


def _is_list_item(text: str) -> bool:
    return LIST_RE.match(text) is not None


def _normalize_list_item(text: str) -> str:
    match = LIST_RE.match(text)
    if not match:
        return text
    marker = match.group(1)
    rest = text[match.end() :].strip()
    if marker in {"•", "◦", "▪", "●"}:
        marker = "-"
    return f"{marker} {rest}"


def _try_render_table(
    lines: list[Line], start: int, blocked: frozenset[int] = frozenset()
) -> tuple[str | None, int]:
    rows: list[list[str]] = []
    consumed = 0
    expected_cells = 0

    for offset, line in enumerate(lines[start : start + 20]):
        if start + offset in blocked:
            break
        cells = split_table_cells(line, normalize=True)
        if len(cells) < 2:
            break
        if expected_cells and abs(len(cells) - expected_cells) > 1:
            break
        expected_cells = expected_cells or len(cells)
        rows.append(cells)
        consumed += 1

    if len(rows) < 2:
        return None, 0

    cell_count = max(len(row) for row in rows)
    if cell_count < 2:
        return None, 0

    normalized = [row + [""] * (cell_count - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    table_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(cell_count)) + " |",
    ]
    for row in body:
        table_lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(table_lines), consumed


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


def _splice_inline_math(line: Line, regions: list[Any]) -> str:
    """Rebuild a body line replacing math-covered words with ``$latex$``.

    Each inline region's bbox covers a contiguous run of source words; those
    words are dropped and the region's LaTeX is inserted once, in reading order,
    where the run began. Words a region does not cover are emitted verbatim, so
    surrounding prose is preserved exactly.

    A region that covers no word *center* (it spans only part of a longer word,
    e.g. the "ℓ4" of "ℓ4-norm") is spliced INTO the word it x-overlaps by
    replacing the region's ``source_text`` substring, so the source glyphs are
    neither emitted twice nor the math appended out of place.
    """
    ordered_regions = sorted(regions, key=lambda r: r.bbox.x0)
    words = sorted(line.words, key=lambda w: w.x_min)
    pieces: list[str] = []
    piece_word: list[Any] = []  # the source Word of each piece, or None
    emitted: set[int] = set()
    for word in words:
        center = word.x_center
        covering = None
        for ri, region in enumerate(ordered_regions):
            if region.bbox.x0 - 1.0 <= center <= region.bbox.x1 + 1.0:
                covering = ri
                break
        if covering is None:
            pieces.append(word.text)
            piece_word.append(word)
            continue
        if covering not in emitted:
            pieces.append(f"${ordered_regions[covering].latex}$")
            piece_word.append(None)
            emitted.add(covering)
    for ri, region in enumerate(ordered_regions):
        if ri in emitted:
            continue
        # Partial-word coverage: splice the LaTeX into the overlapping word by
        # replacing the run's source characters.
        if _splice_into_word(pieces, piece_word, region):
            continue
        # Fallback: append so the math is never silently dropped.
        pieces.append(f"${region.latex}$")
        piece_word.append(None)
    return normalize_line_text(" ".join(piece for piece in pieces if piece))


def _splice_into_word(
    pieces: list[str], piece_word: list[Any], region: Any
) -> bool:
    """Replace ``region.source_text`` inside the word piece it x-overlaps.

    Returns True when the splice succeeded. The candidate is the verbatim word
    piece with the greatest x-overlap with the region's bbox that contains the
    region's source characters; only one occurrence is replaced.
    """
    source_text = getattr(region, "source_text", "")
    if not source_text:
        return False
    bbox = region.bbox
    best_index: int | None = None
    best_overlap = 0.0
    for pi, word in enumerate(piece_word):
        if word is None or source_text not in pieces[pi]:
            continue
        overlap = max(0.0, min(word.x_max, bbox.x1) - max(word.x_min, bbox.x0))
        if overlap > best_overlap:
            best_index = pi
            best_overlap = overlap
    if best_index is None or best_overlap <= 0.0:
        return False
    pieces[best_index] = pieces[best_index].replace(
        source_text, f"${region.latex}$", 1
    )
    return True


def _join_paragraph(lines: list[str]) -> str:
    if not lines:
        return ""

    result = lines[0].strip()
    for line in lines[1:]:
        next_text = line.strip()
        if not next_text:
            continue
        if result.endswith("-") and next_text[:1].islower():
            result = result[:-1] + next_text
            continue
        separator = "" if touching_cjk(result, next_text) else " "
        result += separator + next_text
    return result


def _document_median_word_height(pages: list[PageLayout]) -> float:
    heights = [
        word.height
        for page in pages
        for line in page.lines
        for word in line.words
        if word.height > 0
    ]
    return median(heights) if heights else 0.0


def _artifacts_by_page(artifacts: list[Any]) -> dict[int, list[Any]]:
    by_page: dict[int, list[Any]] = {}
    for artifact in artifacts:
        page = getattr(artifact, "page", None)
        if page is None:
            continue
        by_page.setdefault(page, []).append(artifact)
    for page_artifacts in by_page.values():
        page_artifacts.sort(
            key=lambda artifact: (
                getattr(getattr(artifact, "bbox", None), "y0", 0.0),
                getattr(getattr(artifact, "bbox", None), "x0", 0.0),
            )
        )
    return by_page


def _anchored_artifacts(artifacts: list[Any]) -> dict[int, list[Any]]:
    anchored: dict[int, list[Any]] = {}
    for artifact in artifacts:
        caption = getattr(artifact, "caption", None)
        line_indices = getattr(caption, "line_indices", ()) if caption else ()
        if not line_indices:
            line_indices = getattr(artifact, "anchor_line_indices", ())
        if not line_indices:
            continue
        anchor = max(line_indices)
        anchored.setdefault(anchor, []).append(artifact)
    return anchored


def _suppressed_line_indices(
    lines: list[Line], artifacts: list[Any], median_word_height: float = 0.0
) -> set[int]:
    suppressed: set[int] = set()
    caption_indices: set[int] = set()
    for artifact in artifacts:
        caption = getattr(artifact, "caption", None)
        caption_indices.update(getattr(caption, "line_indices", ()) if caption else ())

    # Section headings (and a leading section number that labels one) are
    # structural text, not part of any figure/table/equation crop, so they
    # are never suppressed even when an over-extracted artifact bbox overlaps.
    protected_indices: set[int] = set()
    if median_word_height > 0:
        for index, line in enumerate(lines):
            if _heading_level(line, median_word_height):
                protected_indices.add(index)
            elif SECTION_NUMBER_RE.match(line.text.strip()) and (
                index + 1 < len(lines)
                and _heading_level(lines[index + 1], median_word_height)
            ):
                protected_indices.add(index)

    for index, line in enumerate(lines):
        if index in caption_indices or index in protected_indices:
            continue
        for artifact in artifacts:
            bbox = getattr(artifact, "bbox", None)
            if bbox is None:
                continue
            if _line_overlaps_bbox(line, bbox):
                suppressed.add(index)
                break
    return suppressed


def _line_overlaps_bbox(line: Line, bbox: Any) -> bool:
    x0 = getattr(bbox, "x0", 0.0)
    y0 = getattr(bbox, "y0", 0.0)
    x1 = getattr(bbox, "x1", 0.0)
    y1 = getattr(bbox, "y1", 0.0)
    x_overlap = max(0.0, min(line.x_max, x1) - max(line.x_min, x0))
    y_overlap = max(0.0, min(line.y_max, y1) - max(line.y_min, y0))
    if x_overlap <= 0 or y_overlap <= 0:
        return False
    x_ratio = x_overlap / max(1.0, line.width)
    y_ratio = y_overlap / max(1.0, line.height)
    return x_ratio >= 0.5 and y_ratio >= 0.5


def _render_table_grid(grid: Any) -> str:
    """Render a reconstructed ``TableGrid`` as a Markdown pipe table.

    The leading ``header_rows`` rows form the header; the ``| --- |`` separator
    follows them. When ``header_rows`` is 0 an empty header row is synthesised so
    the output stays a valid pipe table. Ragged rows are padded to ``ncols`` and
    cell text is escaped with :func:`_escape_cell`.
    """
    matrix = [list(row) for row in getattr(grid, "matrix", ())]
    ncols = getattr(grid, "ncols", 0) or (max((len(row) for row in matrix), default=0))
    if ncols <= 0:
        return ""
    header_rows = getattr(grid, "header_rows", 0)

    def render_row(row: list[str]) -> str:
        padded = list(row) + [""] * (ncols - len(row))
        return "| " + " | ".join(_escape_cell(cell) for cell in padded[:ncols]) + " |"

    lines: list[str] = []
    if header_rows >= 1:
        for row in matrix[:header_rows]:
            lines.append(render_row(row))
        body = matrix[header_rows:]
    else:
        lines.append("| " + " | ".join("" for _ in range(ncols)) + " |")
        body = matrix
    lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for row in body:
        lines.append(render_row(row))
    return "\n".join(lines)


def _artifact_markdown(
    artifact: Any, asset_base_dir: Path | None, inline_images: bool = False
) -> str:
    kind = getattr(artifact, "kind", "artifact")
    grid = getattr(artifact, "table", None)
    if kind == "table" and grid is not None:
        return _render_table_grid(grid)

    caption = getattr(artifact, "caption", None)
    caption_label = getattr(caption, "label", "") if caption else ""
    page = getattr(artifact, "page", "?")
    alt_text = caption_label or (f"page {page}" if kind == "page" else f"{kind} page {page}")
    alt = _escape_alt_text(alt_text)

    if inline_images:
        png_bytes = getattr(artifact, "png_bytes", None)
        if png_bytes:
            encoded = base64.b64encode(png_bytes).decode("ascii")
            return f"![{alt}](data:image/png;base64,{encoded})"
        # Fallback: encode an already-written crop file if one exists.
        asset_path = getattr(artifact, "asset_path", None)
        data_uri = _image_data_uri(asset_path) if asset_path else ""
        return f"![{alt}]({data_uri})" if data_uri else ""

    asset_path = getattr(artifact, "asset_path", None)
    if not asset_path:
        return ""
    return f"![{alt}]({_markdown_link_path(asset_path, asset_base_dir)})"


def _image_data_uri(asset_path: str) -> str:
    """Return a base64 PNG data URI for the crop so it embeds in the Markdown
    itself, or "" if the file cannot be read (then the caller falls back to a
    normal file link)."""
    try:
        data = Path(asset_path).read_bytes()
    except OSError:
        return ""
    if not data:
        return ""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _artifact_identity(artifact: Any) -> str:
    artifact_id = getattr(artifact, "id", "")
    return artifact_id or str(id(artifact))


def _markdown_link_path(asset_path: str, asset_base_dir: Path | None) -> str:
    path = Path(asset_path)
    if asset_base_dir is not None and path.is_absolute():
        try:
            path_text = os.path.relpath(path, asset_base_dir)
        except ValueError:
            path_text = str(path)
    else:
        path_text = str(path)
    path_text = path_text.replace(os.sep, "/")
    if any(char.isspace() for char in path_text) or any(char in path_text for char in "()"):
        escaped = path_text.replace(">", "%3E")
        return f"<{escaped}>"
    return path_text


def _escape_alt_text(text: str) -> str:
    return text.replace("\n", " ").replace("[", "\\[").replace("]", "\\]").strip()

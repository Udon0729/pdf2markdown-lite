from __future__ import annotations

import os
import re
from pathlib import Path
from statistics import median
from typing import Any

from .layout import Line, PageLayout, normalize_line_text

LIST_RE = re.compile(
    r"^\s*((?:[-*+•◦▪●])|(?:\(?\d+[\).])|(?:\(?[A-Za-z][\).])|(?:[IVXLCM]+[\).]))\s+"
)


def render_markdown(
    pages: list[PageLayout],
    *,
    page_breaks: bool = False,
    detect_tables: bool = True,
    artifacts: list[Any] | None = None,
    embed_artifacts: bool = False,
    asset_base_dir: str | Path | None = None,
) -> str:
    parts: list[str] = []
    document_median_height = _document_median_word_height(pages)
    artifacts_by_page = _artifacts_by_page(artifacts or []) if embed_artifacts else {}
    base_dir = Path(asset_base_dir) if asset_base_dir is not None else None

    for page in pages:
        ordered = order_lines(page)
        page_markdown = _render_page(
            ordered,
            median_word_height=document_median_height or page.median_word_height or 10.0,
            detect_tables=detect_tables,
            artifacts=artifacts_by_page.get(page.number, []),
            asset_base_dir=base_dir,
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
) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    formula: list[str] = []
    paragraph_key: tuple[int, int, str] | None = None
    index = 0
    anchored_artifacts = _anchored_artifacts(artifacts)
    emitted_artifact_ids: set[str] = set()
    suppressed_line_indices = _suppressed_line_indices(lines, artifacts)

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_key
        if paragraph:
            blocks.append(_join_paragraph(paragraph))
            paragraph = []
            paragraph_key = None

    def flush_formula() -> None:
        nonlocal formula
        if formula:
            blocks.append("```math\n" + "\n".join(formula).strip() + "\n```")
            formula = []

    def emit_artifacts_for_line(line_index: int) -> None:
        for artifact in anchored_artifacts.get(line_index, []):
            artifact_id = _artifact_identity(artifact)
            if artifact_id in emitted_artifact_ids:
                continue
            snippet = _artifact_markdown(artifact, asset_base_dir)
            if snippet:
                flush_paragraph()
                flush_formula()
                blocks.append(snippet)
                emitted_artifact_ids.add(artifact_id)

    while index < len(lines):
        if index in suppressed_line_indices:
            flush_paragraph()
            flush_formula()
            emit_artifacts_for_line(index)
            index += 1
            continue

        if detect_tables:
            table, consumed = _try_render_table(lines, index)
            if table:
                flush_paragraph()
                flush_formula()
                blocks.append(table)
                for table_index in range(index, index + consumed):
                    emit_artifacts_for_line(table_index)
                index += consumed
                continue

        line = lines[index]
        text = line.text.strip()
        if not text:
            flush_paragraph()
            flush_formula()
            index += 1
            continue

        if _is_formula_line(text):
            flush_paragraph()
            formula.append(text)
        else:
            flush_formula()
            heading_level = _heading_level(line, median_word_height)
            if heading_level:
                flush_paragraph()
                blocks.append(f"{'#' * heading_level} {text}")
            elif _is_list_item(text):
                flush_paragraph()
                blocks.append(_normalize_list_item(text))
            else:
                if paragraph_key is not None and line.block_key != paragraph_key:
                    flush_paragraph()
                paragraph.append(text)
                paragraph_key = line.block_key
        emit_artifacts_for_line(index)
        index += 1

    flush_paragraph()
    flush_formula()
    for artifact in artifacts:
        artifact_id = _artifact_identity(artifact)
        if artifact_id in emitted_artifact_ids:
            continue
        snippet = _artifact_markdown(artifact, asset_base_dir)
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

    ratio = line.median_word_height / median_word_height if median_word_height else 1.0
    numbered_heading = re.match(r"^\d+(?:\.\d+)*\s+\S+", text) is not None
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


def _try_render_table(lines: list[Line], start: int) -> tuple[str | None, int]:
    rows: list[list[str]] = []
    consumed = 0
    expected_cells = 0

    for line in lines[start : start + 20]:
        cells = _split_table_cells(line)
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


def _split_table_cells(line: Line) -> list[str]:
    words = sorted(line.words, key=lambda word: word.x_min)
    if len(words) < 3:
        return []

    widths = [word.width / max(1, len(word.text)) for word in words if word.width > 0]
    char_width = median(widths) if widths else 5.0
    threshold = max(14.0, char_width * 4.0)

    cells: list[list[str]] = [[words[0].text]]
    large_gap_count = 0
    for previous, current in zip(words, words[1:]):
        gap = current.x_min - previous.x_max
        if gap >= threshold:
            large_gap_count += 1
            cells.append([current.text])
        else:
            cells[-1].append(current.text)

    if large_gap_count == 0:
        return []
    return [normalize_line_text(" ".join(cell)).strip() for cell in cells if cell]


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


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
        separator = "" if _touching_cjk(result, next_text) else " "
        result += separator + next_text
    return result


def _touching_cjk(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _is_cjk(left[-1]) or _is_cjk(right[0])


def _is_cjk(char: str) -> bool:
    return (
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


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


def _suppressed_line_indices(lines: list[Line], artifacts: list[Any]) -> set[int]:
    suppressed: set[int] = set()
    caption_indices: set[int] = set()
    for artifact in artifacts:
        caption = getattr(artifact, "caption", None)
        caption_indices.update(getattr(caption, "line_indices", ()) if caption else ())

    for index, line in enumerate(lines):
        if index in caption_indices:
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


def _artifact_markdown(artifact: Any, asset_base_dir: Path | None) -> str:
    asset_path = getattr(artifact, "asset_path", None)
    if not asset_path:
        return ""
    caption = getattr(artifact, "caption", None)
    caption_text = getattr(caption, "text", "") if caption else ""
    kind = getattr(artifact, "kind", "artifact")
    page = getattr(artifact, "page", "?")
    alt_text = caption_text or (f"page {page}" if kind == "page" else f"{kind} page {page}")
    return f"![{_escape_alt_text(alt_text)}]({_markdown_link_path(asset_path, asset_base_dir)})"


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

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .layout import Line, PageLayout
from .markdown import order_lines

CAPTION_RE = re.compile(
    r"^\s*(?P<label>(?P<kind>fig(?:ure)?|table|図|表)\.?\s*"
    r"(?P<number>(?:[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]*[0-9０-９]+"
    r"(?:[-.][0-9A-Za-z０-９Ａ-Ｚａ-ｚ]+)*)|(?:[IVXLCDM]+)))"
    r"\s*[:：.\-]?\s*(?P<body>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def expand(self, amount: float, page: PageLayout | None = None) -> BBox:
        x0 = self.x0 - amount
        y0 = self.y0 - amount
        x1 = self.x1 + amount
        y1 = self.y1 + amount
        if page is not None:
            x0 = max(0.0, x0)
            y0 = max(0.0, y0)
            x1 = min(page.width, x1)
            y1 = min(page.height, y1)
        return BBox(x0, y0, x1, y1)

    def intersects(self, other: BBox, *, margin: float = 0.0) -> bool:
        return not (
            self.x1 + margin < other.x0
            or other.x1 + margin < self.x0
            or self.y1 + margin < other.y0
            or other.y1 + margin < self.y0
        )

    def x_overlap_ratio(self, other: BBox) -> float:
        overlap = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        base = max(1.0, min(self.width, other.width))
        return overlap / base

    def to_json(self) -> list[float]:
        return [
            round(self.x0, 3),
            round(self.y0, 3),
            round(self.x1, 3),
            round(self.y1, 3),
        ]


@dataclass(frozen=True)
class Caption:
    kind: str
    label: str
    text: str
    page: int
    bbox: BBox
    line_indices: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "text": self.text,
            "page": self.page,
            "bbox": self.bbox.to_json(),
            "line_indices": list(self.line_indices),
        }


@dataclass(frozen=True)
class VisualRegion:
    kind: str
    source: str
    page: int
    bbox: BBox
    anchor_line_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class VisualArtifact:
    id: str
    kind: str
    source: str
    page: int
    bbox: BBox
    asset_path: str | None
    caption: Caption | None
    score: float
    anchor_line_indices: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox.to_json(),
            "asset_path": self.asset_path,
            "caption": self.caption.to_json() if self.caption else None,
            "score": round(self.score, 3),
            "anchor_line_indices": list(self.anchor_line_indices),
        }


def extract_artifacts(
    pdf_path: str | Path,
    pages: list[PageLayout],
    *,
    assets_dir: str | Path | None,
    dpi: int = 180,
    jobs: int = 0,
) -> list[VisualArtifact]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for --extract-artifacts. "
            "Install it with `python3 -m pip install 'pdfmdlite[figures]'`."
        ) from exc

    pdf_path = Path(pdf_path)
    output_dir = Path(assets_dir) if assets_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    worker_count = _effective_render_jobs(jobs, len(pages))
    if worker_count <= 1:
        artifacts: list[VisualArtifact] = []
        with fitz.open(pdf_path) as doc:
            for page_layout in pages:
                if page_layout.number < 1 or page_layout.number > len(doc):
                    continue
                artifacts.extend(
                    _extract_page_artifacts(
                        doc[page_layout.number - 1],
                        page_layout,
                        output_dir,
                        dpi,
                    )
                )
    else:
        tasks = [
            (str(pdf_path), page_layout, str(output_dir) if output_dir is not None else None, dpi)
            for page_layout in pages
        ]
        artifacts = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for page_artifacts in executor.map(_extract_page_artifacts_batch, tasks):
                artifacts.extend(page_artifacts)
        artifacts.sort(key=lambda artifact: (artifact.page, artifact.bbox.y0, artifact.bbox.x0))
    return artifacts


def _extract_page_artifacts_batch(
    task: tuple[str, PageLayout, str | None, int],
) -> list[VisualArtifact]:
    import fitz  # type: ignore[import-not-found]

    pdf_path, page_layout, output_dir, dpi = task
    with fitz.open(pdf_path) as doc:
        if page_layout.number < 1 or page_layout.number > len(doc):
            return []
        return _extract_page_artifacts(
            doc[page_layout.number - 1],
            page_layout,
            Path(output_dir) if output_dir is not None else None,
            dpi,
        )


def _extract_page_artifacts(
    pdf_page: Any,
    page_layout: PageLayout,
    output_dir: Path | None,
    dpi: int,
) -> list[VisualArtifact]:
    captions = detect_captions(page_layout)
    regions = _page_visual_regions(pdf_page, page_layout, captions)
    page_artifacts = _attach_captions(page_layout, regions, captions)

    artifacts: list[VisualArtifact] = []
    counters: dict[str, int] = {}
    for artifact in page_artifacts:
        counters[artifact.kind] = counters.get(artifact.kind, 0) + 1
        artifact_id = (
            f"page-{page_layout.number:04d}-{artifact.kind}-"
            f"{counters[artifact.kind]:02d}"
        )
        asset_path = None
        if output_dir is not None:
            asset_path = str(output_dir / f"{artifact_id}.png")
            _render_region(pdf_page, artifact.bbox.expand(4, page_layout), asset_path, dpi)
        artifacts.append(
            VisualArtifact(
                id=artifact_id,
                kind=artifact.kind,
                source=artifact.source,
                page=artifact.page,
                bbox=artifact.bbox,
                asset_path=asset_path,
                caption=artifact.caption,
                score=artifact.score,
                anchor_line_indices=artifact.anchor_line_indices,
            )
        )
    return artifacts


def _effective_render_jobs(requested_jobs: int, page_count: int) -> int:
    if page_count <= 0:
        return 1
    if requested_jobs < 0:
        raise RuntimeError("jobs must be 0 or greater")
    if requested_jobs == 1:
        return 1
    if requested_jobs == 0:
        if page_count < 16:
            return 1
        return max(1, min(os.cpu_count() or 1, 4, page_count))
    return max(1, min(requested_jobs, page_count))


def detect_captions(page: PageLayout) -> list[Caption]:
    lines = sorted([line for line in page.lines if line.text], key=lambda line: (line.y_min, line.x_min))
    captions: list[Caption] = []
    used_indices: set[int] = set()
    median_height = page.median_word_height or 10.0

    for index, line in enumerate(lines):
        if index in used_indices:
            continue
        match = CAPTION_RE.match(line.text)
        if not match:
            continue

        kind = _normalize_caption_kind(match.group("kind"))
        caption_lines = [line]
        line_indices = [index]
        used_indices.add(index)
        for next_index in range(index + 1, min(index + 7, len(lines))):
            next_line = lines[next_index]
            if CAPTION_RE.match(next_line.text):
                break
            if caption_lines[-1].text.rstrip().endswith((".", "。")):
                first_char = next_line.text[:1]
                if first_char and not first_char.islower():
                    break
            gap = next_line.y_min - caption_lines[-1].y_max
            similar_indent = abs(next_line.x_min - line.x_min) <= max(18.0, line.width * 0.12)
            if gap < 0 or gap > median_height * 2.4 or not similar_indent:
                break
            caption_lines.append(next_line)
            line_indices.append(next_index)
            used_indices.add(next_index)

        text = " ".join(item.text for item in caption_lines).strip()
        captions.append(
            Caption(
                kind=kind,
                label=match.group("label").strip(),
                text=text,
                page=page.number,
                bbox=_lines_bbox(caption_lines),
                line_indices=tuple(line_indices),
            )
        )
    return captions


def detect_text_table_regions(page: PageLayout) -> list[VisualRegion]:
    lines = order_lines(page)
    regions: list[VisualRegion] = []
    index = 0
    while index < len(lines):
        row_lines: list[Line] = []
        expected_cells = 0
        for line in lines[index : index + 40]:
            cells = _split_table_cells(line)
            if len(cells) < 2:
                break
            if expected_cells and abs(len(cells) - expected_cells) > 1:
                break
            expected_cells = expected_cells or len(cells)
            row_lines.append(line)
        if len(row_lines) >= 2:
            regions.append(
                VisualRegion(
                    kind="table",
                    source="text_table",
                    page=page.number,
                    bbox=_lines_bbox(row_lines),
                    anchor_line_indices=(index + len(row_lines) - 1,),
                )
            )
            index += len(row_lines)
        else:
            index += 1
    return regions


def _caption_table_regions(page: PageLayout, captions: list[Caption]) -> list[VisualRegion]:
    table_captions = [caption for caption in captions if caption.kind == "table"]
    if not table_captions:
        return []

    lines = order_lines(page)
    regions: list[VisualRegion] = []
    sorted_captions = sorted(table_captions, key=lambda caption: caption.bbox.y0)

    for caption_index, caption in enumerate(sorted_captions):
        next_caption_y = (
            sorted_captions[caption_index + 1].bbox.y0
            if caption_index + 1 < len(sorted_captions)
            else page.height * 0.92
        )
        caption_line_height = caption.bbox.height / max(1, len(caption.line_indices))
        max_table_y = min(next_caption_y - 6.0, caption.bbox.y1 + page.height * 0.32)
        table_lines: list[Line] = []
        previous: Line | None = None
        started = False

        for line in lines:
            if line.y_min <= caption.bbox.y1 + 2.0:
                continue
            if line.y_min >= max_table_y:
                break
            if CAPTION_RE.match(line.text):
                break
            if not _same_caption_column(page, caption, line):
                continue

            small_text = line.median_word_height <= caption_line_height * 0.88
            table_like = _looks_like_dense_table_line(line)
            if not started:
                if not (small_text or table_like):
                    continue
                started = True
            else:
                gap = line.y_min - (previous.y_max if previous else line.y_min)
                if gap > caption_line_height * 1.5 and not (small_text or table_like):
                    break
                if not small_text and not table_like and len(table_lines) >= 2:
                    break

            table_lines.append(line)
            previous = line

        if len(table_lines) >= 2:
            regions.append(
                VisualRegion(
                    kind="table",
                    source="caption_table",
                    page=page.number,
                    bbox=_lines_bbox(table_lines),
                    anchor_line_indices=(max(caption.line_indices),),
                )
            )
    return regions


def _equation_regions(
    page: PageLayout,
    *,
    excluded_regions: list[VisualRegion] | None = None,
) -> list[VisualRegion]:
    excluded_bboxes = [region.bbox for region in (excluded_regions or [])]
    lines = order_lines(page)
    candidates = [
        (index, line)
        for index, line in enumerate(lines)
        if _looks_like_display_equation_line(line, page)
        and not any(_line_overlaps_bbox(line, bbox) for bbox in excluded_bboxes)
    ]
    if not candidates:
        return []

    clusters: list[list[tuple[int, Line]]] = []
    current: list[tuple[int, Line]] = []
    previous_line: Line | None = None
    for item in candidates:
        _, line = item
        if previous_line is None or line.y_min - previous_line.y_max <= 18.0:
            current.append(item)
        else:
            if current:
                clusters.append(current)
            current = [item]
        previous_line = line
    if current:
        clusters.append(current)

    regions: list[VisualRegion] = []
    for cluster in clusters:
        cluster_lines = [line for _, line in cluster]
        bbox = _lines_bbox(cluster_lines)
        text = " ".join(line.text for line in cluster_lines)
        if len(cluster_lines) == 1 and not re.search(r"[=∑∏≤≥≈∈∀⊤]", text):
            continue
        if bbox.width < page.width * 0.12 or bbox.height < 6:
            continue
        if bbox.height > page.height * 0.18:
            continue
        bbox = BBox(
            max(0.0, bbox.x0 - 24.0),
            max(0.0, bbox.y0 - 3.0),
            min(page.width, bbox.x1 + 72.0),
            min(page.height, bbox.y1 + 3.0),
        )
        anchor = max(index for index, _ in cluster)
        regions.append(
            VisualRegion(
                kind="equation",
                source="equation",
                page=page.number,
                bbox=bbox,
                anchor_line_indices=(anchor,),
            )
        )
    return regions


def write_artifacts_manifest(
    manifest_path: str | Path,
    *,
    pdf_path: str | Path,
    artifacts: list[VisualArtifact],
) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_pdf": str(Path(pdf_path)),
        "artifact_count": len(artifacts),
        "artifacts": [artifact.to_json() for artifact in artifacts],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _page_visual_regions(
    pdf_page: Any,
    page: PageLayout,
    captions: list[Caption],
) -> list[VisualRegion]:
    regions: list[VisualRegion] = []

    for bbox in _image_rects(pdf_page):
        if _is_meaningful_region(bbox, page, recall=True):
            regions.append(VisualRegion(kind="figure", source="image", page=page.number, bbox=bbox))

    bboxlog = _bboxlog_entries(pdf_page)
    bboxlog_regions, ruled_rects = _bboxlog_visual_regions(bboxlog, page)
    regions.extend(bboxlog_regions)
    regions.extend(_ruled_table_regions(ruled_rects, page, captions))

    drawing_rects = [
        bbox
        for bbox in _drawing_rects(pdf_page)
        + [bbox for operation, bbox in bboxlog if "path" in operation]
        if _is_drawing_piece(bbox, page)
    ]
    for bbox in _cluster_rects(_dedupe_bboxes(drawing_rects), page):
        if _is_meaningful_region(bbox, page, recall=True):
            kind = "table" if _looks_like_table_rule_cluster(bbox, ruled_rects, page, captions) else "figure"
            regions.append(
                VisualRegion(kind=kind, source="drawing", page=page.number, bbox=bbox)
            )

    table_regions = _caption_table_regions(page, captions) + detect_text_table_regions(page)
    regions.extend(table_regions)
    regions.extend(_equation_regions(page, excluded_regions=regions))
    regions.extend(_caption_fallback_regions(page, captions, regions))
    regions.extend(_display_list_text_regions(bboxlog, page, regions))
    return _dedupe_cross_kind_regions(_merge_regions(regions, page), page)


def _bboxlog_entries(pdf_page: Any) -> list[tuple[str, BBox]]:
    try:
        entries = pdf_page.get_bboxlog()
    except Exception:
        return []

    result: list[tuple[str, BBox]] = []
    for item in entries:
        if not item or len(item) < 2:
            continue
        result.append((str(item[0]), _coerce_bbox(item[1])))
    return result


def _bboxlog_visual_regions(
    entries: list[tuple[str, BBox]],
    page: PageLayout,
) -> tuple[list[VisualRegion], list[BBox]]:
    regions: list[VisualRegion] = []
    drawing_rects: list[BBox] = []
    ruled_rects: list[BBox] = []

    for operation, bbox in entries:
        if not _bbox_on_page(bbox, page):
            continue
        if "image" in operation or "shade" in operation:
            if _is_meaningful_region(bbox, page, recall=True):
                regions.append(
                    VisualRegion(
                        kind="figure",
                        source=f"bboxlog:{operation}",
                        page=page.number,
                        bbox=bbox,
                    )
                )
            continue
        if "path" in operation:
            if _is_rule_piece(bbox, page):
                ruled_rects.append(bbox)
            if _is_drawing_piece(bbox, page):
                drawing_rects.append(bbox)

    for bbox in _cluster_rects(_dedupe_bboxes(drawing_rects), page):
        if _is_meaningful_region(bbox, page, recall=True):
            regions.append(
                VisualRegion(
                    kind="figure",
                    source="bboxlog:path",
                    page=page.number,
                    bbox=bbox,
                )
            )
    return regions, _dedupe_bboxes(ruled_rects)


def _ruled_table_regions(
    ruled_rects: list[BBox],
    page: PageLayout,
    captions: list[Caption],
) -> list[VisualRegion]:
    if not ruled_rects:
        return []

    regions: list[VisualRegion] = []
    for cluster in _cluster_rects(ruled_rects, page):
        if not _is_meaningful_region(cluster, page, recall=True):
            continue
        if not _looks_like_table_rule_cluster(cluster, ruled_rects, page, captions):
            continue
        regions.append(
            VisualRegion(
                kind="table",
                source="ruled_table",
                page=page.number,
                bbox=cluster,
            )
        )
    return regions


def _caption_fallback_regions(
    page: PageLayout,
    captions: list[Caption],
    regions: list[VisualRegion],
) -> list[VisualRegion]:
    fallback: list[VisualRegion] = []
    sorted_captions = sorted(captions, key=lambda caption: caption.bbox.y0)
    for index, caption in enumerate(sorted_captions):
        if _has_caption_attached_region(caption, regions, page):
            continue

        x0, x1 = _caption_column_bounds(page, caption)
        if caption.kind == "figure":
            y1 = max(0.0, caption.bbox.y0 - 4.0)
            previous_caption_y = (
                sorted_captions[index - 1].bbox.y1 + 6.0 if index > 0 else page.height * 0.08
            )
            y0 = max(previous_caption_y, y1 - page.height * 0.40)
        else:
            y0 = min(page.height, caption.bbox.y1 + 4.0)
            next_caption_y = (
                sorted_captions[index + 1].bbox.y0 - 6.0
                if index + 1 < len(sorted_captions)
                else page.height * 0.92
            )
            y1 = min(next_caption_y, y0 + page.height * 0.35)

        bbox = BBox(x0, y0, x1, y1)
        if _is_meaningful_region(bbox, page, recall=True):
            fallback.append(
                VisualRegion(
                    kind=caption.kind,
                    source="caption_fallback",
                    page=page.number,
                    bbox=bbox,
                    anchor_line_indices=(max(caption.line_indices),),
                )
            )
    return fallback


def _display_list_text_regions(
    entries: list[tuple[str, BBox]],
    page: PageLayout,
    regions: list[VisualRegion],
) -> list[VisualRegion]:
    text_bboxes = [
        bbox
        for operation, bbox in entries
        if "text" in operation and _bbox_on_page(bbox, page)
    ]
    if not text_bboxes:
        return []

    candidates = _equation_regions(page, excluded_regions=[])
    equation_bboxes = [candidate.bbox for candidate in candidates]
    result: list[VisualRegion] = []
    for bbox in _cluster_rects(
        [bbox for bbox in text_bboxes if _overlaps_any(bbox, equation_bboxes, min_ratio=0.15)],
        page,
    ):
        if _is_meaningful_region(bbox, page, recall=True):
            result.append(
                VisualRegion(
                    kind="equation",
                    source="bboxlog:text_equation",
                    page=page.number,
                    bbox=bbox,
                )
            )
    return result


def _has_caption_attached_region(
    caption: Caption,
    regions: list[VisualRegion],
    page: PageLayout,
) -> bool:
    for region in regions:
        if region.kind != caption.kind:
            continue
        if region.bbox.x_overlap_ratio(caption.bbox) < 0.08:
            continue
        if _vertical_gap(region.bbox, caption.bbox) <= max(96.0, page.height * 0.16):
            return True
    return False


def _caption_column_bounds(page: PageLayout, caption: Caption) -> tuple[float, float]:
    if caption.bbox.width >= page.width * 0.42:
        return page.width * 0.06, page.width * 0.94
    center = (caption.bbox.x0 + caption.bbox.x1) / 2.0
    if center <= page.width / 2.0:
        return page.width * 0.05, page.width * 0.53
    return page.width * 0.47, page.width * 0.95


def _looks_like_table_rule_cluster(
    cluster: BBox,
    ruled_rects: list[BBox],
    page: PageLayout,
    captions: list[Caption],
) -> bool:
    near_figure_caption = any(
        caption.kind == "figure"
        and caption.bbox.x_overlap_ratio(cluster) >= 0.20
        and _vertical_gap(cluster, caption.bbox) <= max(36.0, page.height * 0.06)
        for caption in captions
    )
    if near_figure_caption:
        return False

    members = [bbox for bbox in ruled_rects if bbox.intersects(cluster, margin=2.0)]
    horizontal = [
        bbox
        for bbox in members
        if bbox.width >= page.width * 0.08 and bbox.height <= max(2.5, page.height * 0.004)
    ]
    vertical = [
        bbox
        for bbox in members
        if bbox.height >= page.height * 0.025 and bbox.width <= max(2.5, page.width * 0.004)
    ]
    near_table_caption = any(
        caption.kind == "table"
        and caption.bbox.x_overlap_ratio(cluster) >= 0.08
        and _vertical_gap(cluster, caption.bbox) <= page.height * 0.20
        for caption in captions
    )
    if len(horizontal) >= 2 and len(vertical) >= 2:
        return True
    if near_table_caption and len(horizontal) >= 2:
        return True
    if near_table_caption and len(members) >= 4:
        return True
    return False


def _dedupe_cross_kind_regions(regions: list[VisualRegion], page: PageLayout) -> list[VisualRegion]:
    kept: list[VisualRegion] = []
    priority = {"table": 3, "equation": 2, "figure": 1}
    for region in sorted(
        regions,
        key=lambda item: (
            item.page,
            priority.get(item.kind, 0) * -1,
            item.bbox.y0,
            item.bbox.x0,
        ),
    ):
        duplicate = False
        for existing in kept:
            overlap = _overlap_area(region.bbox, existing.bbox)
            if overlap <= 0:
                continue
            smaller = max(1.0, min(region.bbox.area, existing.bbox.area))
            if overlap / smaller >= 0.88:
                duplicate = True
                break
        if not duplicate and _is_meaningful_region(region.bbox, page, recall=True):
            kept.append(region)
    return sorted(kept, key=lambda item: (item.page, item.bbox.y0, item.bbox.x0))


def _image_rects(pdf_page: Any) -> list[BBox]:
    rects: list[BBox] = []
    try:
        images = pdf_page.get_images(full=True)
    except Exception:
        return rects

    seen: set[tuple[float, float, float, float]] = set()
    for image in images:
        if not image:
            continue
        xref = image[0]
        try:
            image_rects = pdf_page.get_image_rects(xref)
        except Exception:
            continue
        for rect in image_rects:
            bbox = _coerce_bbox(rect)
            key = tuple(round(value, 1) for value in bbox.to_json())
            if key not in seen:
                rects.append(bbox)
                seen.add(key)
    return rects


def _drawing_rects(pdf_page: Any) -> list[BBox]:
    try:
        drawings = pdf_page.get_cdrawings()
    except Exception:
        try:
            drawings = pdf_page.get_drawings()
        except Exception:
            return []

    rects: list[BBox] = []
    for drawing in drawings:
        rect = drawing.get("rect") if isinstance(drawing, dict) else None
        if rect is None:
            continue
        rects.append(_coerce_bbox(rect))
    return rects


def _cluster_rects(rects: list[BBox], page: PageLayout) -> list[BBox]:
    clusters = rects[:]
    margin = max(6.0, min(page.width, page.height) * 0.015)
    changed = True
    while changed:
        changed = False
        merged: list[BBox] = []
        while clusters:
            current = clusters.pop()
            remaining: list[BBox] = []
            for other in clusters:
                if current.intersects(other, margin=margin):
                    current = current.union(other)
                    changed = True
                else:
                    remaining.append(other)
            clusters = remaining
            merged.append(current)
        clusters = merged
    return clusters


def _merge_regions(regions: list[VisualRegion], page: PageLayout) -> list[VisualRegion]:
    merged: list[VisualRegion] = []
    for region in sorted(regions, key=lambda item: (item.page, item.bbox.y0, item.bbox.x0)):
        if region.bbox.area <= 0:
            continue
        match_index = None
        for index, existing in enumerate(merged):
            if existing.kind != region.kind:
                continue
            if existing.bbox.intersects(region.bbox, margin=8.0):
                match_index = index
                break
        if match_index is None:
            merged.append(region)
        else:
            existing = merged[match_index]
            anchor_line_indices = existing.anchor_line_indices or region.anchor_line_indices
            merged[match_index] = VisualRegion(
                kind=existing.kind,
                source=existing.source
                if existing.source == region.source
                else f"{existing.source}+{region.source}",
                page=existing.page,
                bbox=existing.bbox.union(region.bbox).expand(0, page),
                anchor_line_indices=anchor_line_indices,
            )
    return merged


def _attach_captions(
    page: PageLayout,
    regions: list[VisualRegion],
    captions: list[Caption],
) -> list[VisualArtifact]:
    artifacts: list[VisualArtifact] = []
    for region in regions:
        if region.kind == "equation":
            caption, score = None, 0.0
        else:
            caption, score = _nearest_caption(page, region, captions)
        kind = region.kind
        if caption is not None:
            kind = caption.kind if caption.kind in {"figure", "table"} else kind
        artifacts.append(
            VisualArtifact(
                id="",
                kind=kind,
                source=region.source,
                page=region.page,
                bbox=region.bbox,
                asset_path=None,
                caption=caption,
                score=score,
                anchor_line_indices=region.anchor_line_indices,
            )
        )
    return artifacts


def _nearest_caption(
    page: PageLayout,
    region: VisualRegion,
    captions: list[Caption],
) -> tuple[Caption | None, float]:
    best_caption: Caption | None = None
    best_score = float("inf")
    for caption in captions:
        if caption.page != region.page:
            continue
        vertical_gap = _vertical_gap(region.bbox, caption.bbox)
        if vertical_gap > max(72.0, page.height * 0.18):
            continue
        overlap = region.bbox.x_overlap_ratio(caption.bbox)
        if overlap < 0.1 and vertical_gap > 18.0:
            continue

        score = vertical_gap - overlap * 24.0
        if caption.kind != region.kind:
            score += 40.0
        if caption.kind == "figure" and caption.bbox.y0 >= region.bbox.y1:
            score -= 10.0
        if caption.kind == "table" and caption.bbox.y1 <= region.bbox.y0:
            score -= 10.0
        if region.source == "text_table" and caption.kind == "table":
            score -= 20.0

        if score < best_score:
            best_score = score
            best_caption = caption

    return best_caption, best_score if best_caption is not None else 0.0


def _vertical_gap(first: BBox, second: BBox) -> float:
    if second.y0 >= first.y1:
        return second.y0 - first.y1
    if first.y0 >= second.y1:
        return first.y0 - second.y1
    return 0.0


def _render_region(pdf_page: Any, bbox: BBox, output_path: str, dpi: int) -> None:
    import fitz  # type: ignore[import-not-found]

    scale = dpi / 72.0
    rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    pixmap.save(output_path)


def _normalize_caption_kind(kind: str) -> str:
    normalized = kind.lower().rstrip(".")
    if normalized in {"table", "表"}:
        return "table"
    return "figure"


def _lines_bbox(lines: list[Line]) -> BBox:
    return BBox(
        min(line.x_min for line in lines),
        min(line.y_min for line in lines),
        max(line.x_max for line in lines),
        max(line.y_max for line in lines),
    )


def _coerce_bbox(rect: Any) -> BBox:
    if hasattr(rect, "x0"):
        return BBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    values = list(rect)
    return BBox(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _bbox_on_page(bbox: BBox, page: PageLayout) -> bool:
    if bbox.area <= 0:
        return False
    return not (
        bbox.x1 < 0
        or bbox.y1 < 0
        or bbox.x0 > page.width
        or bbox.y0 > page.height
    )


def _is_meaningful_region(bbox: BBox, page: PageLayout, *, recall: bool = False) -> bool:
    page_area = max(1.0, page.width * page.height)
    min_area_ratio = 0.00045 if recall else 0.004
    min_width_ratio = 0.035 if recall else 0.08
    min_height_ratio = 0.008 if recall else 0.025
    if bbox.area < page_area * min_area_ratio:
        return False
    if bbox.width < page.width * min_width_ratio or bbox.height < page.height * min_height_ratio:
        return False
    if bbox.width > page.width * 0.98 and bbox.height > page.height * 0.98:
        return False
    return True


def _is_rule_piece(bbox: BBox, page: PageLayout) -> bool:
    if bbox.area <= 0:
        return False
    thin_height = bbox.height <= max(2.5, page.height * 0.004)
    thin_width = bbox.width <= max(2.5, page.width * 0.004)
    long_enough = bbox.width >= page.width * 0.04 or bbox.height >= page.height * 0.015
    return long_enough and (thin_height or thin_width)


def _is_drawing_piece(bbox: BBox, page: PageLayout) -> bool:
    if bbox.area <= 0:
        return False
    if bbox.width > page.width * 0.98 and bbox.height > page.height * 0.98:
        return False
    if bbox.width < 2 and bbox.height < 2:
        return False
    return True


def _dedupe_bboxes(rects: list[BBox]) -> list[BBox]:
    result: list[BBox] = []
    seen: set[tuple[float, float, float, float]] = set()
    for bbox in rects:
        key = tuple(round(value, 1) for value in bbox.to_json())
        if key in seen:
            continue
        seen.add(key)
        result.append(bbox)
    return result


def _overlap_area(first: BBox, second: BBox) -> float:
    x_overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    y_overlap = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return x_overlap * y_overlap


def _overlaps_any(bbox: BBox, others: list[BBox], *, min_ratio: float) -> bool:
    for other in others:
        overlap = _overlap_area(bbox, other)
        if overlap / max(1.0, min(bbox.area, other.area)) >= min_ratio:
            return True
    return False


def _looks_like_dense_table_line(line: Line) -> bool:
    digit_count = sum(any(char.isdigit() for char in word.text) for word in line.words)
    small_text = line.median_word_height <= 8.5
    if small_text and len(line.words) >= 4:
        return True
    if digit_count >= 4:
        return True
    if small_text and digit_count >= 2:
        return True
    return False


def _line_overlaps_bbox(line: Line, bbox: BBox) -> bool:
    x_overlap = max(0.0, min(line.x_max, bbox.x1) - max(line.x_min, bbox.x0))
    y_overlap = max(0.0, min(line.y_max, bbox.y1) - max(line.y_min, bbox.y0))
    if x_overlap <= 0 or y_overlap <= 0:
        return False
    return (
        x_overlap / max(1.0, line.width) >= 0.5
        and y_overlap / max(1.0, line.height) >= 0.5
    )


def _same_caption_column(page: PageLayout, caption: Caption, line: Line) -> bool:
    if caption.bbox.width >= page.width * 0.45:
        return True
    x0, x1 = _caption_column_bounds(page, caption)
    return x0 <= line.x_center <= x1


def _looks_like_display_equation_line(line: Line, page: PageLayout) -> bool:
    text = line.text.strip()
    if not text or len(text) > 120:
        return False
    if CAPTION_RE.match(text):
        return False
    if text.lower().startswith(("where ", "therefore ", "assuming ")):
        return False
    if len(text) > 60 and len(text.split()) > 8:
        return False

    math_chars = set("=∑∏√≤≥≠≈∈∀∂∆△⊤⊥λγµσθΩωαβζ∞·±−∗")
    math_count = sum(1 for char in text if char in math_chars)
    compact = line.width <= page.width * 0.78
    centered = line.x_min >= page.width * 0.08 and line.x_max <= page.width * 0.92
    if math_count and compact and centered:
        return True
    if re.match(r"^(?:[A-Z]{1,3}|[ijkln]|[ijkln]=\d+|X|N[Bb])$", text):
        return compact and centered
    if re.match(r"^\(?\d+\)?$", text):
        return False
    return False


def _split_table_cells(line: Line) -> list[str]:
    words = sorted(line.words, key=lambda word: word.x_min)
    if len(words) < 3:
        return []

    widths = [word.width / max(1, len(word.text)) for word in words if word.width > 0]
    char_width = median(widths) if widths else 5.0
    threshold = max(14.0, char_width * 4.0)

    cells: list[list[str]] = [[words[0].text]]
    gap_count = 0
    for previous, current in zip(words, words[1:]):
        gap = current.x_min - previous.x_max
        if gap >= threshold:
            gap_count += 1
            cells.append([current.text])
        else:
            cells[-1].append(current.text)
    if gap_count == 0:
        return []
    return [" ".join(cell).strip() for cell in cells if cell]

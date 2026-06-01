from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .layout import Line, PageLayout, normalize_line_text
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
class TableGrid:
    """A reconstructed table as a row x column matrix of cell text.

    Built deterministically from the PDF glyph stream and drawn ruled lines
    (CPU-only, ML-free). ``header_rows`` counts the leading rows of ``matrix``
    that form the table header (the row(s) above the under-header rule).
    """

    matrix: tuple[tuple[str, ...], ...]
    header_rows: int
    ncols: int

    def to_json(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "header_rows": self.header_rows,
            "ncols": self.ncols,
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
    table: TableGrid | None = None
    # In-memory PNG bytes for inline (self-contained) embedding; never written
    # to disk and intentionally excluded from to_json (it is not JSON-friendly).
    png_bytes: bytes | None = None

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
            "table": self.table.to_json() if self.table else None,
        }


def extract_artifacts(
    pdf_path: str | Path,
    pages: list[PageLayout],
    *,
    assets_dir: str | Path | None,
    dpi: int = 180,
    jobs: int = 0,
    inline_images: bool = False,
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
    # Inline mode renders crops to in-memory PNG bytes and embeds them directly
    # in the Markdown, so no asset files (and no assets directory) are written.
    if output_dir is not None and not inline_images:
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
                        inline_images,
                    )
                )
    else:
        tasks = [
            (
                str(pdf_path),
                page_layout,
                str(output_dir) if output_dir is not None else None,
                dpi,
                inline_images,
            )
            for page_layout in pages
        ]
        artifacts = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for page_artifacts in executor.map(_extract_page_artifacts_batch, tasks):
                artifacts.extend(page_artifacts)
        artifacts.sort(key=lambda artifact: (artifact.page, artifact.bbox.y0, artifact.bbox.x0))

    if output_dir is not None:
        _prune_stale_crops(output_dir, artifacts)
    return artifacts


def _prune_stale_crops(output_dir: Path, artifacts: list[VisualArtifact]) -> None:
    """Remove crop PNGs in this run's assets dir that this run did not emit.

    A regeneration must not leave orphaned crops behind: a table region now
    renders as a Markdown table (no PNG), and a previous run's
    ``page-*-table-*.png`` would otherwise linger and confuse the reader.
    Scoped strictly to the run's own assets directory and only to files whose
    name matches the artifact crop scheme (``page-NNNN-<kind>-NN.png``), so no
    unrelated user file is ever touched.
    """
    keep = {
        Path(artifact.asset_path).name
        for artifact in artifacts
        if artifact.asset_path
    }
    crop_name = re.compile(r"^page-\d{4}-[a-z]+-\d{2}\.png$")
    for entry in output_dir.iterdir():
        if not entry.is_file():
            continue
        if not crop_name.match(entry.name):
            continue
        if entry.name in keep:
            continue
        entry.unlink()


def _extract_page_artifacts_batch(
    task: tuple[str, PageLayout, str | None, int, bool],
) -> list[VisualArtifact]:
    import fitz  # type: ignore[import-not-found]

    pdf_path, page_layout, output_dir, dpi, inline_images = task
    with fitz.open(pdf_path) as doc:
        if page_layout.number < 1 or page_layout.number > len(doc):
            return []
        return _extract_page_artifacts(
            doc[page_layout.number - 1],
            page_layout,
            Path(output_dir) if output_dir is not None else None,
            dpi,
            inline_images,
        )


def _extract_page_artifacts(
    pdf_page: Any,
    page_layout: PageLayout,
    output_dir: Path | None,
    dpi: int,
    inline_images: bool = False,
) -> list[VisualArtifact]:
    captions = detect_captions(page_layout)
    regions = _page_visual_regions(pdf_page, page_layout, captions)
    page_artifacts = _attach_captions(page_layout, regions, captions)
    page_artifacts = _merge_captioned_artifacts(page_artifacts, page_layout)

    artifacts: list[VisualArtifact] = []
    counters: dict[str, int] = {}
    for artifact in page_artifacts:
        counters[artifact.kind] = counters.get(artifact.kind, 0) + 1
        artifact_id = (
            f"page-{page_layout.number:04d}-{artifact.kind}-"
            f"{counters[artifact.kind]:02d}"
        )

        # A ruled table region is reconstructed into a Markdown pipe table from
        # the glyph stream and drawn rules; on success it carries that grid and
        # NO PNG crop is written (the table is emitted as text at its caption
        # anchor). Figures and unreconstructable regions still crop to a PNG.
        grid: TableGrid | None = None
        if artifact.kind == "table":
            grid = _reconstruct_table_grid(pdf_page, page_layout, artifact.bbox)

        asset_path = None
        png_bytes = None
        if grid is None:
            crop = artifact.bbox.expand(4, page_layout)
            if inline_images:
                # Render to in-memory PNG bytes; write no file.
                png_bytes = _render_region_bytes(pdf_page, crop, dpi)
            elif output_dir is not None:
                asset_path = str(output_dir / f"{artifact_id}.png")
                _render_region(pdf_page, crop, asset_path, dpi)
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
                table=grid,
                png_bytes=png_bytes,
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
    # Equations are reconstructed as LaTeX by mathreco, not cropped, so no
    # equation/caption-fallback/display-list-text regions are produced here.
    # Figures and tables remain image crops.
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
    priority = {"table": 3, "figure": 1}
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


def _merge_captioned_artifacts(
    artifacts: list[VisualArtifact],
    page: PageLayout,
) -> list[VisualArtifact]:
    groups: dict[tuple[int, str, Caption], list[VisualArtifact]] = {}
    merged: list[VisualArtifact] = []
    for artifact in artifacts:
        if artifact.caption is None:
            merged.append(artifact)
            continue
        key = (artifact.page, artifact.kind, artifact.caption)
        groups.setdefault(key, []).append(artifact)

    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        bbox = members[0].bbox
        for member in members[1:]:
            bbox = bbox.union(member.bbox)
        sources: list[str] = []
        for member in members:
            if member.source not in sources:
                sources.append(member.source)
        anchor_line_indices: tuple[int, ...] = ()
        for member in members:
            if member.anchor_line_indices:
                anchor_line_indices = member.anchor_line_indices
                break
        merged.append(
            VisualArtifact(
                id="",
                kind=members[0].kind,
                source="+".join(sources),
                page=members[0].page,
                bbox=bbox,
                asset_path=None,
                caption=members[0].caption,
                score=min(member.score for member in members),
                anchor_line_indices=anchor_line_indices,
            )
        )

    merged.sort(key=lambda artifact: (artifact.bbox.y0, artifact.bbox.x0))
    return merged


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


def _render_region_bytes(pdf_page: Any, bbox: BBox, dpi: int) -> bytes:
    """Render a region to in-memory PNG bytes (no file written)."""
    import fitz  # type: ignore[import-not-found]

    scale = dpi / 72.0
    rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    return pixmap.tobytes("png")


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


def _same_caption_column(page: PageLayout, caption: Caption, line: Line) -> bool:
    if caption.bbox.width >= page.width * 0.45:
        return True
    x0, x1 = _caption_column_bounds(page, caption)
    return x0 <= line.x_center <= x1


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


def _cluster_centers(values: list[float], tolerance: float = 2.0) -> list[float]:
    """Collapse near-coincident rule centers into one center each.

    A fully ruled grid often draws a rule as a doubled/over-stroked pair whose
    centers fall within a point or two of each other; clustering keeps one
    boundary per visual rule. Returns sorted cluster means.
    """
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _region_ruled_grid(
    pdf_page: Any,
    region: BBox,
) -> tuple[list[float], list[float]]:
    """Recover horizontal- and vertical-rule centers inside a table region.

    The decisive geometry of a born-digital ruled table lives in the page's
    ``get_bboxlog()`` ``path`` entries, which carry the actual stroke thickness
    (``get_drawings`` reports the same rules as zero-area rects). Rules are
    classified by REGION-RELATIVE length so a short cell-tall vertical rule is
    not rejected by a page-relative threshold (the reason ``_is_rule_piece``
    cannot be reused here). Returns ``(h_centers, v_centers)``: ``h_centers`` are
    horizontal-rule y-centers (row boundaries), ``v_centers`` are vertical-rule
    x-centers (column boundaries), each sorted and clustered within 2pt.
    """
    region_width = max(1.0, region.width)
    region_height = max(1.0, region.height)
    margin = 3.0
    h_centers: list[float] = []
    v_centers: list[float] = []
    for operation, bbox in _bboxlog_entries(pdf_page):
        if "path" not in operation:
            continue
        cx = (bbox.x0 + bbox.x1) / 2.0
        cy = (bbox.y0 + bbox.y1) / 2.0
        if not (
            region.x0 - margin <= cx <= region.x1 + margin
            and region.y0 - margin <= cy <= region.y1 + margin
        ):
            continue
        if bbox.height <= 2.5 and bbox.width >= 0.30 * region_width:
            h_centers.append(cy)
        elif bbox.width <= 2.5 and bbox.height >= 0.15 * region_height:
            v_centers.append(cx)
    return _cluster_centers(h_centers), _cluster_centers(v_centers)


def _gap_column_edges(words_by_row: list[list[Any]], region: BBox) -> list[float]:
    """Booktabs fallback: derive column edges from shared large inter-word gaps.

    For each row, find inter-word gaps wide enough to separate columns (the same
    ``max(14, 4*char_width)`` style ``_split_table_cells`` uses) and record each
    gap midpoint. Midpoints that recur (within a tolerance) across at least 60%
    of the rows are internal column edges; the region's left and right edges
    bound the outer columns. Used only when no vertical rules are drawn.
    """
    if not words_by_row:
        return []
    gap_midpoints: list[float] = []
    row_count = 0
    for words in words_by_row:
        ordered = sorted(words, key=lambda word: word.x_min)
        if len(ordered) < 2:
            continue
        row_count += 1
        widths = [
            word.width / max(1, len(word.text)) for word in ordered if word.width > 0
        ]
        char_width = median(widths) if widths else 5.0
        threshold = max(14.0, char_width * 4.0)
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.x_min - previous.x_max
            if gap >= threshold:
                gap_midpoints.append((previous.x_max + current.x_min) / 2.0)
    if row_count == 0:
        return []

    clusters = _cluster_centers(gap_midpoints, tolerance=8.0)
    tolerance = 8.0
    internal: list[float] = []
    for center in clusters:
        support = sum(
            1 for midpoint in gap_midpoints if abs(midpoint - center) <= tolerance
        )
        if support >= max(2, int(row_count * 0.6 + 0.999)):
            internal.append(center)
    if not internal:
        return []
    return [region.x0] + sorted(internal) + [region.x1]


def _join_cell_words(words: list[tuple[int, float, str]]) -> str:
    """Join a cell's words in reading order with hyphen-wrap repair.

    ``words`` are ``(text_line_y, x_min, text)`` tuples. Sorting by text-line y
    then x keeps a wrapped multi-line cell in reading order (ordering by pure x
    interleaves the lines). When a text line ends with a hyphen and the next
    word starts lowercase, the hyphen is dropped and the words are concatenated
    -- the same repair ``markdown._join_paragraph`` applies to broken prose
    (e.g. ``activa-`` + ``tion`` -> ``activation``).
    """
    ordered = sorted(words, key=lambda item: (item[0], item[1]))
    result = ""
    for _, _, text in ordered:
        token = text
        if not result:
            result = token
            continue
        if result.endswith("-") and token[:1].islower():
            result = result[:-1] + token
            continue
        separator = "" if _touching_cjk(result, token) else " "
        result += separator + token
    return normalize_line_text(result).strip()


def _touching_cjk(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _is_cjk(left[-1]) or _is_cjk(right[0])


def _is_cjk(char: str) -> bool:
    return (
        "぀" <= char <= "ヿ"
        or "㐀" <= char <= "䶿"
        or "一" <= char <= "鿿"
        or "豈" <= char <= "﫿"
    )


def _row_band_for(cy: float, h_centers: list[float]) -> int | None:
    for i in range(len(h_centers) - 1):
        if h_centers[i] - 1.0 <= cy < h_centers[i + 1] + 1.0:
            return i
    return None


def _column_for(cx: float, edges: list[float]) -> int | None:
    for j in range(len(edges) - 1):
        if edges[j] - 1.0 <= cx < edges[j + 1] + 1.0:
            return j
    return None


def _reconstruct_table_grid(
    pdf_page: Any,
    page_layout: PageLayout,
    region: BBox,
) -> TableGrid | None:
    """Reconstruct a detected table region into a row x column cell matrix.

    Deterministic and ML-free: row boundaries come from horizontal drawn rules
    (a tall band is a wrapped multi-line cell-row, kept as one logical row),
    column boundaries from vertical drawn rules (primary path) or, when none are
    drawn, shared inter-word gaps (booktabs fallback). Every word whose center
    lies in the region is assigned to exactly one (row, column) cell; cell words
    are joined in reading order with hyphen-wrap repair. The row(s) above the
    under-header rule (the 2nd horizontal rule) are the header.

    Returns ``None`` only when no horizontal rules are found (not a ruled table
    region); a region with rules but fewer than 2 recoverable columns raises a
    defect rather than silently degrading.
    """
    h_centers, v_centers = _region_ruled_grid(pdf_page, region)
    if len(h_centers) < 2:
        return None

    nrows = len(h_centers) - 1

    region_words: list[Any] = []
    for line in page_layout.lines:
        for word in line.words:
            if not word.text:
                continue
            cy = (word.y_min + word.y_max) / 2.0
            if _row_band_for(cy, h_centers) is None:
                continue
            cx = (word.x_min + word.x_max) / 2.0
            if not (region.x0 - 1.0 <= cx <= region.x1 + 1.0):
                continue
            region_words.append(word)

    if len(v_centers) >= 2:
        column_edges = v_centers
    else:
        words_by_row: list[list[Any]] = [[] for _ in range(nrows)]
        for word in region_words:
            row = _row_band_for((word.y_min + word.y_max) / 2.0, h_centers)
            if row is not None:
                words_by_row[row].append(word)
        column_edges = _gap_column_edges(words_by_row, region)
        if len(column_edges) < 2:
            raise RuntimeError(
                "table reconstruction defect: region "
                f"{region.to_json()} on page {page_layout.number} has horizontal "
                "rules but no recoverable column structure (neither vertical "
                "rules nor shared inter-word gaps)"
            )

    ncols = len(column_edges) - 1
    cells: dict[tuple[int, int], list[tuple[int, float, str]]] = {}
    for word in region_words:
        cy = (word.y_min + word.y_max) / 2.0
        cx = (word.x_min + word.x_max) / 2.0
        row = _row_band_for(cy, h_centers)
        col = _column_for(cx, column_edges)
        if row is None or col is None:
            continue
        cells.setdefault((row, col), []).append(
            (round(word.y_min), word.x_min, word.text)
        )

    matrix: list[tuple[str, ...]] = []
    for row in range(nrows):
        matrix.append(
            tuple(
                _join_cell_words(cells.get((row, col), []))
                for col in range(ncols)
            )
        )

    # The under-header rule is the 2nd clustered horizontal rule, so the single
    # band above it (row 0) is the header. A region with only one band has no
    # header.
    header_rows = 1 if nrows >= 1 else 0
    return TableGrid(matrix=tuple(matrix), header_rows=header_rows, ncols=ncols)

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pdfmdlite.artifacts import (
    BBox,
    Caption,
    TableGrid,
    VisualArtifact,
    VisualRegion,
    _merge_captioned_artifacts,
    _merge_regions,
    _prune_stale_crops,
    _reconstruct_table_grid,
    _region_ruled_grid,
    detect_captions,
    detect_text_table_regions,
    write_artifacts_manifest,
)
from pdfmdlite.layout import Line, PageLayout, Word


def make_line(
    text: str,
    *,
    y: float,
    x: float = 72.0,
    size: float = 10.0,
    page: int = 1,
    block: int = 0,
) -> Line:
    words = []
    cursor = x
    for index, token in enumerate(text.split()):
        width = max(4.0, len(token) * size * 0.5)
        words.append(
            Word(
                text=token,
                x_min=cursor,
                y_min=y,
                x_max=cursor + width,
                y_max=y + size,
                block_id=block,
                line_id=0,
                word_id=index,
            )
        )
        cursor += width + size * 0.6
    return Line(words=words, page_number=page, block_id=block)


def make_row(cells: list[str], *, y: float, page: int = 1) -> Line:
    words = []
    for index, cell in enumerate(cells):
        x = 72 + index * 150
        words.append(
            Word(
                text=cell,
                x_min=x,
                y_min=y,
                x_max=x + len(cell) * 5,
                y_max=y + 10,
                block_id=5,
                line_id=int(y),
                word_id=index,
            )
        )
    return Line(words=words, page_number=page, block_id=5)


class EquationCropPathRemovedTests(unittest.TestCase):
    def test_no_equation_region_detector_remains(self) -> None:
        # Equations are reconstructed as LaTeX by mathreco, never cropped, so
        # the character-based equation-region detectors must be gone from the
        # artifacts module.
        import pdfmdlite.artifacts as artifacts

        for name in (
            "_equation_regions",
            "_extend_equation_region_down",
            "_display_list_text_regions",
            "_predominantly_in_equation_column",
            "_caption_fallback_regions",
            "_looks_like_display_equation_line",
        ):
            self.assertFalse(
                hasattr(artifacts, name),
                f"{name} should have been removed from artifacts.py",
            )

    def test_merge_unions_intersecting_same_kind_regions(self) -> None:
        # With the equation special-case gone, _merge_regions simply unions two
        # intersecting regions of the same kind (used for figures/tables).
        page = PageLayout(number=1, width=612, height=792, lines=[])
        top = VisualRegion(kind="figure", source="image", page=1, bbox=BBox(214, 95, 441, 200))
        bottom = VisualRegion(kind="figure", source="image", page=1, bbox=BBox(214, 195, 441, 280))

        merged = _merge_regions([top, bottom], page)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, BBox(214, 95, 441, 280))


class ArtifactDetectionTests(unittest.TestCase):
    def test_detects_multiline_figure_caption(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("Figure S1. End-to-end architecture of the parser", y=500, block=1),
                make_line("with caption continuation on the next line.", y=512, block=1),
            ],
        )

        captions = detect_captions(page)

        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].kind, "figure")
        self.assertEqual(captions[0].label, "Figure S1")
        self.assertIn("caption continuation", captions[0].text)

    def test_detects_text_table_region(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("Table 1. Metrics", y=80, block=1),
                make_row(["Method", "Accuracy", "Latency"], y=110),
                make_row(["Base", "91", "44ms"], y=124),
                make_row(["Ours", "95", "18ms"], y=138),
            ],
        )

        regions = detect_text_table_regions(page)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "table")
        self.assertEqual(regions[0].source, "text_table")

    def test_manifest_writer_is_json(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[make_line("Figure 1. Demo", y=100, block=1)],
        )
        caption = detect_captions(page)[0]
        artifact = VisualArtifact(
            id="page-0001-figure-01",
            kind="figure",
            source="image",
            page=1,
            bbox=caption.bbox,
            asset_path="assets/page-0001-figure-01.png",
            caption=caption,
            score=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "artifacts.json"
            write_artifacts_manifest(manifest, pdf_path="paper.pdf", artifacts=[artifact])
            text = manifest.read_text(encoding="utf-8")

        self.assertIn('"artifact_count": 1', text)
        self.assertIn('"Figure 1. Demo"', text)


class MergeCaptionedArtifactsTests(unittest.TestCase):
    def _page(self) -> PageLayout:
        return PageLayout(number=7, width=612, height=792, lines=[])

    def _figure_caption(self) -> Caption:
        return Caption(
            kind="figure",
            label="Figure 4",
            text="Figure 4: An overview.",
            page=7,
            bbox=BBox(108, 210, 486, 222),
            line_indices=(0,),
        )

    def test_merge_captioned_artifacts_unions_same_caption(self) -> None:
        page = self._page()
        caption = self._figure_caption()
        left = VisualArtifact(
            id="",
            kind="figure",
            source="image",
            page=7,
            bbox=BBox(108, 82, 268, 202),
            asset_path=None,
            caption=caption,
            score=5.0,
            anchor_line_indices=(0,),
        )
        right = VisualArtifact(
            id="",
            kind="figure",
            source="vector",
            page=7,
            bbox=BBox(326, 82, 486, 202),
            asset_path=None,
            caption=caption,
            score=3.0,
            anchor_line_indices=(),
        )

        merged = _merge_captioned_artifacts([left, right], page)

        self.assertEqual(len(merged), 1)
        result = merged[0]
        self.assertEqual(result.bbox, BBox(108, 82, 486, 202))
        self.assertEqual(result.kind, "figure")
        self.assertIs(result.caption, caption)
        self.assertEqual(result.source, "image+vector")
        self.assertEqual(result.score, 3.0)
        self.assertEqual(result.anchor_line_indices, (0,))

    def test_none_caption_artifacts_not_merged(self) -> None:
        page = self._page()
        first = VisualArtifact(
            id="",
            kind="equation",
            source="equation",
            page=7,
            bbox=BBox(108, 82, 268, 120),
            asset_path=None,
            caption=None,
            score=0.0,
        )
        second = VisualArtifact(
            id="",
            kind="equation",
            source="equation",
            page=7,
            bbox=BBox(108, 200, 268, 240),
            asset_path=None,
            caption=None,
            score=0.0,
        )

        merged = _merge_captioned_artifacts([first, second], page)

        self.assertEqual(len(merged), 2)

    def test_same_label_different_page_not_merged(self) -> None:
        page = self._page()
        caption_a = Caption(
            kind="figure",
            label="Figure 2",
            text="Figure 2: First.",
            page=7,
            bbox=BBox(108, 210, 486, 222),
            line_indices=(0,),
        )
        caption_b = Caption(
            kind="figure",
            label="Figure 2",
            text="Figure 2: First.",
            page=8,
            bbox=BBox(108, 210, 486, 222),
            line_indices=(0,),
        )
        first = VisualArtifact(
            id="",
            kind="figure",
            source="image",
            page=7,
            bbox=BBox(108, 82, 268, 202),
            asset_path=None,
            caption=caption_a,
            score=1.0,
        )
        second = VisualArtifact(
            id="",
            kind="figure",
            source="image",
            page=8,
            bbox=BBox(108, 82, 268, 202),
            asset_path=None,
            caption=caption_b,
            score=1.0,
        )

        merged = _merge_captioned_artifacts([first, second], page)

        self.assertEqual(len(merged), 2)

    def test_distinct_captions_not_merged(self) -> None:
        page = self._page()
        caption_a = Caption(
            kind="figure",
            label="Figure 4",
            text="Figure 4: First.",
            page=7,
            bbox=BBox(108, 210, 300, 222),
            line_indices=(0,),
        )
        caption_b = Caption(
            kind="figure",
            label="Figure 5",
            text="Figure 5: Second.",
            page=7,
            bbox=BBox(108, 410, 300, 422),
            line_indices=(2,),
        )
        first = VisualArtifact(
            id="",
            kind="figure",
            source="image",
            page=7,
            bbox=BBox(108, 82, 268, 202),
            asset_path=None,
            caption=caption_a,
            score=1.0,
        )
        second = VisualArtifact(
            id="",
            kind="figure",
            source="image",
            page=7,
            bbox=BBox(108, 282, 268, 402),
            asset_path=None,
            caption=caption_b,
            score=1.0,
        )

        merged = _merge_captioned_artifacts([first, second], page)

        self.assertEqual(len(merged), 2)


class _FakeRect:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class _FakePage:
    """A stand-in for a fitz page exposing only ``get_bboxlog``.

    The real born-digital ruled table reports its rules through
    ``get_bboxlog()`` ``path`` entries that carry stroke thickness, while
    ``get_drawings`` reports the same rules as zero-area rects (rejected by
    ``_is_rule_piece``). This fake mirrors the bboxlog entries.
    """

    def __init__(self, entries: list[tuple[str, _FakeRect]]) -> None:
        self._entries = entries

    def get_bboxlog(self) -> list[tuple[str, _FakeRect]]:
        return self._entries


def _ruled_table_page() -> _FakePage:
    # Horizontal rules at the seven y boundaries and vertical rules at the four
    # x boundaries of a 6-row x 3-column grid, expressed with the real stroke
    # thickness so the region-relative classifier accepts them. The vertical
    # rules are deliberately short (cell-tall), which a page-relative threshold
    # would reject.
    h_y = [146.2, 157.6, 168.9, 180.3, 191.6, 213.9, 225.3]
    v_x = [116.7, 158.9, 398.0, 495.3]
    entries: list[tuple[str, _FakeRect]] = []
    for y in h_y:
        entries.append(("stroke-path", _FakeRect(116.5, y - 0.2, 495.5, y + 0.2)))
    for top, bottom in zip(h_y, h_y[1:]):
        for x in v_x:
            entries.append(("stroke-path", _FakeRect(x - 0.2, top, x + 0.2, bottom)))
    return _FakePage(entries)


def _word(text: str, x0: float, y0: float, x1: float, y1: float) -> Word:
    return Word(text=text, x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _table1_layout() -> PageLayout:
    # Three columns; the 5th data row (1-0003) wraps onto two text lines and the
    # wrapped middle cell carries an "activa-"/"tion" hyphen break to repair.
    rows = [
        (147.5, [("Feature", 116.7), ("Description", 159.0), ("Interpretability", 398.0)]),
        (147.5, [(None, None), ("(Generated by GPT-4)", 250.0), ("Score", 440.0)]),
        (159.5, [("1-0000", 116.7), ("parts of individual names.", 159.0), ("0.33", 440.0)]),
        (170.5, [("1-0001", 116.7), ("actions by a subject.", 159.0), ("-0.11", 440.0)]),
        (182.0, [("1-0002", 116.7), ("instances of the letter.", 159.0), ("0.55", 440.0)]),
        (193.0, [("1-0003", 116.7), ("low activa-", 159.0), ("0.57", 440.0)]),
        (204.5, [(None, None), ("tion for names.", 159.0), (None, None)]),
        (215.0, [("1-0004", 116.7), ("legal terms.", 159.0), ("0.19", 440.0)]),
    ]
    lines: list[Line] = []
    for y, cells in rows:
        words: list[Word] = []
        for text, x in cells:
            if text is None:
                continue
            words.append(_word(text, x, y, x + len(text) * 4.0, y + 9.0))
        if words:
            lines.append(Line(words=words, page_number=4, block_id=0))
    return PageLayout(number=4, width=612, height=792, lines=lines)


class RegionRuledGridTests(unittest.TestCase):
    def test_recovers_zero_area_style_rules_by_region_relative_thresholds(self) -> None:
        page = _ruled_table_page()
        region = BBox(116.0, 146.0, 496.0, 226.0)

        h_centers, v_centers = _region_ruled_grid(page, region)

        self.assertEqual([round(c, 1) for c in h_centers],
                         [146.2, 157.6, 168.9, 180.3, 191.6, 213.9, 225.3])
        # The short vertical rules are recovered even though they are well under
        # any page-relative minimum length.
        self.assertEqual([round(c, 1) for c in v_centers],
                         [116.7, 158.9, 398.0, 495.3])


class ReconstructTableGridTests(unittest.TestCase):
    def test_reconstructs_matrix_merges_wrapped_row_and_repairs_hyphen(self) -> None:
        grid = _reconstruct_table_grid(
            _ruled_table_page(),
            _table1_layout(),
            BBox(116.0, 146.0, 496.0, 226.0),
        )

        self.assertIsInstance(grid, TableGrid)
        self.assertEqual(grid.ncols, 3)
        self.assertEqual(grid.header_rows, 1)
        # 6 logical rows: the tall band auto-merges the two header text lines and
        # the wrapped "1-0003" body cell.
        self.assertEqual(len(grid.matrix), 6)
        self.assertEqual(grid.matrix[0][0], "Feature")
        self.assertEqual(grid.matrix[0][1], "Description (Generated by GPT-4)")
        self.assertEqual(grid.matrix[0][2], "Interpretability Score")
        self.assertEqual(grid.matrix[1], ("1-0000", "parts of individual names.", "0.33"))
        # The "activa-"/"tion" wrap is repaired to "activation".
        self.assertEqual(grid.matrix[4][1], "low activation for names.")
        self.assertEqual(grid.matrix[5], ("1-0004", "legal terms.", "0.19"))

    def test_returns_none_without_horizontal_rules(self) -> None:
        page = _FakePage([])
        grid = _reconstruct_table_grid(
            page, _table1_layout(), BBox(116.0, 146.0, 496.0, 226.0)
        )
        self.assertIsNone(grid)

    def test_column_consistency_every_body_row_has_header_width(self) -> None:
        grid = _reconstruct_table_grid(
            _ruled_table_page(),
            _table1_layout(),
            BBox(116.0, 146.0, 496.0, 226.0),
        )
        assert grid is not None
        ncols = grid.ncols
        self.assertEqual(len(grid.matrix[0]), ncols)
        for row in grid.matrix[grid.header_rows :]:
            self.assertEqual(len(row), ncols)


class PruneStaleCropsTests(unittest.TestCase):
    def _crop(self, directory: Path, name: str) -> Path:
        path = directory / name
        path.write_bytes(b"\x89PNG\r\n")
        return path

    def test_removes_orphaned_table_crop_but_keeps_referenced_and_unrelated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp)
            # A stale table crop from a prior run (the table is now Markdown, so
            # no current artifact references it), a figure crop this run emits,
            # and a user file that does not match the crop scheme.
            stale_table = self._crop(assets, "page-0004-table-01.png")
            kept_figure = self._crop(assets, "page-0002-figure-01.png")
            user_file = self._crop(assets, "my-notes.png")

            artifacts = [
                VisualArtifact(
                    id="page-0004-table-01",
                    kind="table",
                    source="ruled_table",
                    page=4,
                    bbox=BBox(116, 146, 496, 226),
                    asset_path=None,  # reconstructed as Markdown: no crop
                    caption=None,
                    score=0.0,
                    table=TableGrid(matrix=(("a", "b"),), header_rows=1, ncols=2),
                ),
                VisualArtifact(
                    id="page-0002-figure-01",
                    kind="figure",
                    source="image",
                    page=2,
                    bbox=BBox(72, 100, 300, 300),
                    asset_path=str(kept_figure),
                    caption=None,
                    score=0.0,
                ),
            ]

            _prune_stale_crops(assets, artifacts)

            self.assertFalse(stale_table.exists())
            self.assertTrue(kept_figure.exists())
            self.assertTrue(user_file.exists())


class RealTable1ReconstructionTests(unittest.TestCase):
    SAMPLE_PDF = Path(__file__).resolve().parent.parent / "2309.08600v3.pdf"

    def test_reconstructs_real_table1_header_and_body_row(self) -> None:
        if not self.SAMPLE_PDF.exists():
            self.skipTest("sample PDF not present")
        try:
            import fitz  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        import fitz

        from pdfmdlite.pymupdf_text import extract_layout

        document = extract_layout(self.SAMPLE_PDF)
        page_layout = next(page for page in document.pages if page.number == 4)
        with fitz.open(self.SAMPLE_PDF) as doc:
            grid = _reconstruct_table_grid(
                doc[3], page_layout, BBox(116.276, 145.994, 495.723, 225.496)
            )

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(grid.ncols, 3)
        self.assertEqual(grid.header_rows, 1)
        self.assertEqual(
            grid.matrix[0],
            ("Feature", "Description (Generated by GPT-4)", "Interpretability Score"),
        )
        self.assertEqual(
            grid.matrix[1],
            ("1-0000", "parts of individual names, especially last names.", "0.33"),
        )
        # The wrapped 1-0003 row repairs "activa-"/"tion".
        wrapped = next(row for row in grid.matrix if row[0] == "1-0003")
        self.assertIn("activation", wrapped[1])
        self.assertEqual(wrapped[2], "0.57")

    def test_cell_conservation_no_content_word_dropped(self) -> None:
        if not self.SAMPLE_PDF.exists():
            self.skipTest("sample PDF not present")
        try:
            import fitz  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        import fitz
        from collections import Counter

        from pdfmdlite.layout import normalize_line_text
        from pdfmdlite.pymupdf_text import extract_layout

        region = BBox(116.276, 145.994, 495.723, 225.496)
        h_centers = [146.2, 157.6, 168.9, 180.3, 191.6, 213.9, 225.3]
        document = extract_layout(self.SAMPLE_PDF)
        page_layout = next(page for page in document.pages if page.number == 4)

        # The multiset of content words whose center lies in a row band of the
        # region (each is a real single token in the PDF text layer) must equal
        # the multiset of tokens across the reconstructed cells: nothing dropped,
        # nothing duplicated. The one deterministic transform is the hyphen
        # repair "activa-"/"tion" -> "activation", reconciled before comparing.
        in_region: Counter[str] = Counter()
        for line in page_layout.lines:
            for word in line.words:
                if not word.text:
                    continue
                cy = (word.y_min + word.y_max) / 2.0
                cx = (word.x_min + word.x_max) / 2.0
                if not (region.x0 - 1.0 <= cx <= region.x1 + 1.0):
                    continue
                if not any(
                    lo - 1.0 <= cy < hi + 1.0
                    for lo, hi in zip(h_centers, h_centers[1:])
                ):
                    continue
                normalized = normalize_line_text(word.text).strip()
                if normalized:
                    in_region[normalized] += 1

        with fitz.open(self.SAMPLE_PDF) as doc:
            grid = _reconstruct_table_grid(doc[3], page_layout, region)
        assert grid is not None

        emitted: Counter[str] = Counter()
        for row in grid.matrix:
            for cell in row:
                for token in cell.split():
                    emitted[token] += 1

        if in_region.get("activa-") and in_region.get("tion"):
            in_region["activa-"] -= 1
            in_region["tion"] -= 1
            in_region["activation"] += 1
            in_region = +in_region  # drop zero/negative counts

        self.assertEqual(emitted, in_region)


if __name__ == "__main__":
    unittest.main()

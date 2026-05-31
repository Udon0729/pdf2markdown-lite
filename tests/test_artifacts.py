from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pdfmdlite.artifacts import (
    VisualArtifact,
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


if __name__ == "__main__":
    unittest.main()

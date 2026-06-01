from __future__ import annotations

import unittest

from pdfmdlite.artifacts import BBox, Caption, TableGrid, VisualArtifact
from pdfmdlite.layout import DocumentLayout, Line, PageLayout, Word, remove_repeating_marginalia
from pdfmdlite.markdown import _render_table_grid, render_markdown


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


class MarkdownRenderingTests(unittest.TestCase):
    def test_heading_and_paragraph_rendering(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("Document Title", y=80, size=20, block=1),
                make_line("This is a para-", y=130, size=10, block=2),
                make_line("graph with a repaired break.", y=143, size=10, block=2),
                make_line("• First item", y=180, size=10, block=3),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn("# Document Title", markdown)
        self.assertIn("This is a paragraph with a repaired break.", markdown)
        self.assertIn("- First item", markdown)

    def test_multiline_heading_with_linebreak_hyphen_is_merged(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line(
                    "SPARSE AUTOENCODERS FIND HIGHLY INTER-", y=80, x=108, size=20, block=1
                ),
                make_line(
                    "PRETABLE FEATURES IN LANGUAGE MODELS", y=100, x=108, size=20, block=1
                ),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn(
            "# SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE FEATURES IN LANGUAGE MODELS",
            markdown,
        )
        self.assertNotIn("# PRETABLE", markdown)
        self.assertNotIn("INTER-", markdown)

    def test_orphan_section_number_prepended_to_heading(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("2", y=476, x=108, size=12, block=1),
                make_line(
                    "TAKING FEATURES OUT OF SUPERPOSITION WITH SPARSE DICTIONARY",
                    y=476,
                    x=127,
                    size=12,
                    block=1,
                ),
                make_line("LEARNING", y=490, x=127, size=12, block=1),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn(
            "## 2 TAKING FEATURES OUT OF SUPERPOSITION WITH SPARSE DICTIONARY LEARNING",
            markdown,
        )
        self.assertNotIn("\n\n2\n\n", markdown)

    def test_subsection_number_merged_preserves_internal_hyphen(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("4", y=325, x=108, size=12, block=1),
                make_line(
                    "IDENTIFYING CAUSALLY-IMPORTANT DICTIONARY FEATURES FOR",
                    y=325,
                    x=127,
                    size=12,
                    block=1,
                ),
                make_line(
                    "INDIRECT OBJECT IDENTIFICATION", y=339, x=127, size=12, block=1
                ),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn(
            "## 4 IDENTIFYING CAUSALLY-IMPORTANT DICTIONARY FEATURES FOR "
            "INDIRECT OBJECT IDENTIFICATION",
            markdown,
        )
        self.assertIn("CAUSALLY-IMPORTANT", markdown)
        self.assertNotIn("\n\n4\n\n", markdown)

    def test_appendix_letter_label_merged_into_heading(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("A", y=524, x=108, size=12, block=1),
                make_line(
                    "AUTOINTERPRETATION PROTOCOL", y=524, x=130, size=12, block=1
                ),
                make_line("C.1", y=560, x=108, size=10, block=2),
                make_line(
                    "INTERPRETABILITY IS CONSISTENT", y=560, x=134, size=10, block=2
                ),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn("## A AUTOINTERPRETATION PROTOCOL", markdown)
        self.assertIn("## C.1 INTERPRETABILITY IS CONSISTENT", markdown)
        self.assertNotRegex(markdown, r"(?m)^A$")
        self.assertNotRegex(markdown, r"(?m)^C\.1$")

    def test_math_fragment_with_braces_not_heading(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                # Tall (size 13) so the height ratio would otherwise promote it.
                make_line("{xi}nvec", y=100, x=108, size=13, block=1),
                make_line(
                    "this is ordinary body prose long enough to read as a paragraph",
                    y=120,
                    x=108,
                    size=10,
                    block=2,
                ),
            ],
        )

        markdown = render_markdown([page])

        self.assertNotIn("# {xi}nvec", markdown)
        self.assertNotIn("## {xi}nvec", markdown)

    def test_rotated_arxiv_margin_stamp_is_dropped(self) -> None:
        # A 90-degree rotated stamp keeps glyphs in the extreme left margin
        # (narrow x extent, abnormally tall bboxes). Build it from explicit
        # words so the line x-center stays near the page's left edge.
        stamp_words = []
        for index, token in enumerate("arXiv:2309.08600v3 [cs.LG] 4 Oct 2023".split()):
            stamp_words.append(
                Word(
                    text=token,
                    x_min=10.9,
                    y_min=218 + index * 45,
                    x_max=37.6,
                    y_max=218 + index * 45 + 40,
                    block_id=2,
                    line_id=0,
                    word_id=index,
                )
            )
        lines = [
            make_line("ABSTRACT", y=199, x=278, size=12, block=1),
            Line(words=stamp_words, page_number=1, block_id=2),
        ]
        # plenty of normal-height body lines so the document median stays ~10
        for offset in range(12):
            lines.append(
                make_line(
                    "One of the roadblocks to a better understanding of neural networks.",
                    y=226 + offset * 11,
                    x=144,
                    size=10,
                    block=3,
                )
            )
        page = PageLayout(number=1, width=612, height=792, lines=lines)

        markdown = render_markdown([page])

        self.assertNotIn("arXiv:2309.08600v3", markdown)
        self.assertIn("One of the roadblocks", markdown)
        self.assertNotIn("# arXiv", markdown)

    def test_legitimate_large_heading_not_dropped_as_stamp(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("A Large Centered Title", y=80, x=108, size=20, block=1),
                make_line("Body paragraph follows here.", y=130, x=108, size=10, block=2),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn("# A Large Centered Title", markdown)

    def test_cjk_lines_join_without_extra_space(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("これは日本語の", y=100, size=10, block=1),
                make_line("段落です", y=113, size=10, block=1),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn("これは日本語の段落です", markdown)

    def test_repeated_headers_and_page_numbers_are_removed(self) -> None:
        pages = [
            PageLayout(
                number=1,
                width=612,
                height=792,
                lines=[
                    make_line("Shared Header", y=20, page=1, block=0),
                    make_line("Body one", y=100, page=1, block=1),
                    make_line("1", y=760, page=1, block=2),
                ],
            ),
            PageLayout(
                number=2,
                width=612,
                height=792,
                lines=[
                    make_line("Shared Header", y=20, page=2, block=0),
                    make_line("Body two", y=100, page=2, block=1),
                    make_line("2", y=760, page=2, block=2),
                ],
            ),
        ]

        remove_repeating_marginalia(pages)
        markdown = render_markdown(pages)

        self.assertNotIn("Shared Header", markdown)
        self.assertNotIn("\n1\n", markdown)
        self.assertIn("Body one", markdown)
        self.assertIn("Body two", markdown)

    def test_simple_table_detection(self) -> None:
        def row(cells: list[str], y: float) -> Line:
            words = []
            for index, cell in enumerate(cells):
                x = 72 + index * 130
                words.append(
                    Word(
                        text=cell,
                        x_min=x,
                        y_min=y,
                        x_max=x + len(cell) * 5,
                        y_max=y + 10,
                        block_id=1,
                        line_id=int(y),
                        word_id=index,
                    )
                )
            return Line(words=words, page_number=1, block_id=1)

        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                row(["Name", "Score", "Rank"], 100),
                row(["Ada", "98", "1"], 114),
                row(["Linus", "91", "2"], 128),
            ],
        )

        markdown = render_markdown([page])

        self.assertIn("| Name | Score | Rank |", markdown)
        self.assertIn("| --- | --- | --- |", markdown)
        self.assertIn("| Ada | 98 | 1 |", markdown)

    def test_inline_images_embeds_png_as_base64_data_uri(self) -> None:
        import base64
        import tempfile
        from pathlib import Path

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[make_line("Figure 1. Demo", y=100, size=10, block=1)],
        )
        caption = Caption(
            kind="figure",
            label="Figure 1",
            text="Figure 1. Demo",
            page=1,
            bbox=BBox(72, 100, 220, 110),
            line_indices=(0,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "page-0001-figure-01.png"
            asset.write_bytes(png_bytes)
            artifact = VisualArtifact(
                id="page-0001-figure-01",
                kind="figure",
                source="image",
                page=1,
                bbox=BBox(72, 40, 300, 90),
                asset_path=str(asset),
                caption=caption,
                score=0.0,
            )
            inline_md = render_markdown(
                [page], artifacts=[artifact], embed_artifacts=True, inline_images=True
            )
            link_md = render_markdown(
                [page], artifacts=[artifact], embed_artifacts=True, asset_base_dir=tmp
            )

        # inline mode embeds the PNG as a base64 data URI (self-contained .md)
        self.assertIn("![Figure 1](data:image/png;base64,", inline_md)
        self.assertNotIn("page-0001-figure-01.png", inline_md)
        # default mode keeps an external file link and no data URI
        self.assertIn("page-0001-figure-01.png", link_md)
        self.assertNotIn("data:image", link_md)

    def test_inline_images_uses_in_memory_png_bytes_without_file(self) -> None:
        import base64

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[make_line("Figure 1. Demo", y=100, size=10, block=1)],
        )
        caption = Caption(
            kind="figure",
            label="Figure 1",
            text="Figure 1. Demo",
            page=1,
            bbox=BBox(72, 100, 220, 110),
            line_indices=(0,),
        )
        # asset_path is None: the crop lives only in memory (png_bytes).
        artifact = VisualArtifact(
            id="page-0001-figure-01",
            kind="figure",
            source="image",
            page=1,
            bbox=BBox(72, 40, 300, 90),
            asset_path=None,
            caption=caption,
            score=0.0,
            png_bytes=png_bytes,
        )

        markdown = render_markdown(
            [page], artifacts=[artifact], embed_artifacts=True, inline_images=True
        )

        self.assertIn("![Figure 1](data:image/png;base64,", markdown)
        self.assertNotIn(".png)", markdown)  # no external file reference at all

    def test_render_table_grid_emits_pipe_table(self) -> None:
        grid = TableGrid(
            matrix=(
                ("Feature", "Description", "Score"),
                ("1-0000", "names", "0.33"),
                ("1-0001", "actions a|b", "-0.11"),
            ),
            header_rows=1,
            ncols=3,
        )

        table = _render_table_grid(grid)
        lines = table.split("\n")

        self.assertEqual(lines[0], "| Feature | Description | Score |")
        self.assertEqual(lines[1], "| --- | --- | --- |")
        self.assertEqual(lines[2], "| 1-0000 | names | 0.33 |")
        # A pipe inside a cell is escaped so it does not split the column.
        self.assertEqual(lines[3], "| 1-0001 | actions a\\|b | -0.11 |")
        # One separator row plus a body row per data row, header excluded.
        self.assertEqual(len(lines), 4)

    def test_render_table_grid_synthesizes_header_when_none(self) -> None:
        grid = TableGrid(
            matrix=(("a", "b"), ("c", "d")),
            header_rows=0,
            ncols=2,
        )

        table = _render_table_grid(grid)
        lines = table.split("\n")

        # An empty header row keeps the pipe table valid Markdown.
        self.assertEqual(lines[0], "|  |  |")
        self.assertEqual(lines[1], "| --- | --- |")
        self.assertEqual(lines[2], "| a | b |")
        self.assertEqual(lines[3], "| c | d |")

    def test_table_artifact_renders_as_pipe_table_at_caption_anchor(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("Table 1. Results", y=100, size=10, block=1),
                make_line("Next paragraph", y=180, size=10, block=2),
            ],
        )
        caption = Caption(
            kind="table",
            label="Table 1",
            text="Table 1. Results",
            page=1,
            bbox=BBox(72, 100, 220, 110),
            line_indices=(0,),
        )
        grid = TableGrid(
            matrix=(("Method", "Score"), ("Ours", "95")),
            header_rows=1,
            ncols=2,
        )
        artifact = VisualArtifact(
            id="page-0001-table-01",
            kind="table",
            source="ruled_table",
            page=1,
            bbox=BBox(72, 120, 300, 160),
            asset_path=None,
            caption=caption,
            score=0.0,
            anchor_line_indices=(0,),
            table=grid,
        )

        markdown = render_markdown(
            [page],
            artifacts=[artifact],
            embed_artifacts=True,
            asset_base_dir="/tmp",
        )

        self.assertIn("| Method | Score |", markdown)
        self.assertIn("| --- | --- |", markdown)
        self.assertIn("| Ours | 95 |", markdown)
        # No image link for a reconstructed table.
        self.assertNotIn("![", markdown)
        self.assertNotIn(".png", markdown)
        # The table lands at the caption anchor (after the caption text).
        self.assertLess(markdown.index("Table 1. Results"), markdown.index("| Method"))

    def test_figure_artifact_still_renders_as_image(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[make_line("Figure 1. Demo", y=100, size=10, block=1)],
        )
        caption = Caption(
            kind="figure",
            label="Figure 1",
            text="Figure 1. Demo",
            page=1,
            bbox=BBox(72, 100, 220, 110),
            line_indices=(0,),
        )
        artifact = VisualArtifact(
            id="page-0001-figure-01",
            kind="figure",
            source="image",
            page=1,
            bbox=BBox(72, 40, 300, 90),
            asset_path="/tmp/output_assets/page-0001-figure-01.png",
            caption=caption,
            score=0.0,
            table=None,
        )

        markdown = render_markdown(
            [page],
            artifacts=[artifact],
            embed_artifacts=True,
            asset_base_dir="/tmp",
        )

        self.assertIn("![Figure 1](output_assets/page-0001-figure-01.png)", markdown)

    def test_embeds_artifact_after_caption(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("Figure 1. Pipeline overview", y=100, size=10, block=1),
                make_line("Next paragraph", y=140, size=10, block=2),
            ],
        )
        caption = Caption(
            kind="figure",
            label="Figure 1",
            text="Figure 1. Pipeline overview",
            page=1,
            bbox=BBox(72, 100, 220, 110),
            line_indices=(0,),
        )
        artifact = VisualArtifact(
            id="page-0001-figure-01",
            kind="figure",
            source="image",
            page=1,
            bbox=BBox(72, 40, 300, 90),
            asset_path="/tmp/output_assets/page-0001-figure-01.png",
            caption=caption,
            score=0.0,
        )

        markdown = render_markdown(
            [page],
            artifacts=[artifact],
            embed_artifacts=True,
            asset_base_dir="/tmp",
        )

        self.assertIn("Figure 1. Pipeline overview", markdown)
        self.assertEqual(markdown.count("Figure 1. Pipeline overview"), 1)
        self.assertIn("![Figure 1](output_assets/page-0001-figure-01.png)", markdown)
        self.assertNotIn("![Figure 1. Pipeline overview]", markdown)
        self.assertLess(
            markdown.index("Figure 1. Pipeline overview"),
            markdown.index("output_assets/page-0001-figure-01.png"),
        )

    def test_lone_equals_not_emitted_as_math_fence(self) -> None:
        # Right-column display-equation fragments scattered by PDF extraction:
        # a lone "=" must never become its own ```math``` block, and the
        # right-margin equation number "(1)" must not leak as a body line.
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line(
                    "input vector x, our network produces the output, given by",
                    y=80,
                    x=108,
                    size=10,
                    block=1,
                ),
                make_line("c", y=98, x=256, size=10, block=2),
                make_line("=", y=98, x=271, size=10, block=2),
                make_line("ReLU(Mx + b)", y=98, x=289, size=10, block=2),
                make_line("(1)", y=98, x=492, size=10, block=2),
                make_line(
                    "where M is a learned parameter matrix used downstream.",
                    y=130,
                    x=108,
                    size=10,
                    block=3,
                ),
            ],
        )

        markdown = render_markdown([page])

        self.assertNotIn("```math\n=\n```", markdown)
        # No standalone "(1)" line survives (it sits in the right margin).
        self.assertNotIn("\n(1)\n", markdown)
        self.assertNotIn("\n(1)", markdown.rstrip())
        # Surrounding body text is preserved.
        self.assertIn("input vector x", markdown)
        self.assertIn("where M is a learned parameter", markdown)

    def test_equation_fragments_consolidated_or_dropped(self) -> None:
        # A right-column cluster sharing a vertical band collapses into at most
        # one math block whose content is not a single symbol/number.
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("=", y=98, x=271, size=10, block=2),
                make_line("ReLU(Mx + b)", y=98, x=289, size=10, block=2),
                make_line("M T c", y=112, x=289, size=10, block=2),
                make_line("=", y=112, x=271, size=10, block=2),
                make_line("cifi", y=126, x=289, size=10, block=2),
                make_line("(2)", y=126, x=492, size=10, block=2),
            ],
        )

        markdown = render_markdown([page])

        self.assertEqual(markdown.count("```math"), 1)
        import re as _re

        fences = _re.findall(r"```math\n(.*?)\n```", markdown, _re.S)
        self.assertEqual(len(fences), 1)
        content = fences[0].strip()
        self.assertNotRegex(content, r"^\(?\d+\)?$")
        self.assertGreater(len(content.split("\n")), 1)

    def test_equation_region_does_not_suppress_body(self) -> None:
        # An equation artifact with a TIGHT bbox over only the equation rows
        # must not suppress the surrounding body paragraph or heading.
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line(
                    "The autoencoder is trained to minimise the loss function below.",
                    y=80,
                    x=108,
                    size=10,
                    block=1,
                ),
                make_line("L(x) = stuff", y=110, x=240, size=10, block=2),
                make_line(
                    "where the term controls the sparsity of the reconstruction.",
                    y=160,
                    x=108,
                    size=10,
                    block=3,
                ),
                make_line("INTERPRETING DICTIONARY FEATURES", y=200, x=108, size=10, block=4),
            ],
        )
        artifact = VisualArtifact(
            id="page-0001-equation-01",
            kind="equation",
            source="equation",
            page=1,
            bbox=BBox(238, 108, 442, 122),
            asset_path="/tmp/output_assets/page-0001-equation-01.png",
            caption=None,
            score=0.0,
            anchor_line_indices=(1,),
        )

        markdown = render_markdown(
            [page],
            artifacts=[artifact],
            embed_artifacts=True,
            asset_base_dir="/tmp",
        )

        self.assertIn("The autoencoder is trained", markdown)
        self.assertIn("where the term controls the sparsity", markdown)
        self.assertIn("INTERPRETING DICTIONARY FEATURES", markdown)
        self.assertIn("output_assets/page-0001-equation-01.png", markdown)

    def test_artifact_alt_falls_back_when_no_caption(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("E = m c^2", y=100, size=10, block=1),
                make_line("Next paragraph", y=140, size=10, block=2),
            ],
        )
        artifact = VisualArtifact(
            id="page-0001-equation-01",
            kind="equation",
            source="equation",
            page=1,
            bbox=BBox(72, 90, 300, 115),
            asset_path="/tmp/output_assets/page-0001-equation-01.png",
            caption=None,
            score=0.0,
            anchor_line_indices=(0,),
        )

        markdown = render_markdown(
            [page],
            artifacts=[artifact],
            embed_artifacts=True,
            asset_base_dir="/tmp",
        )

        self.assertIn("![equation page 1](output_assets/page-0001-equation-01.png)", markdown)
        self.assertNotIn("![]", markdown)


class RealPdfTableEndToEndTests(unittest.TestCase):
    """Full-pipeline checks on the bundled sample PDF.

    Exercises the integration end to end: the two ruled tables must surface as
    Markdown pipe tables at their caption anchors (no table PNG crop), figures
    must stay image crops, equations must stay LaTeX, and the table cell text
    must not leak into prose.
    """

    import os as _os
    from pathlib import Path as _Path

    SAMPLE_PDF = _Path(__file__).resolve().parent.parent / "2309.08600v3.pdf"

    def _convert(self):
        import tempfile

        from pdfmdlite.converter import ConversionOptions, convert_pdf_to_result

        if not self.SAMPLE_PDF.exists():
            self.skipTest("sample PDF not present")
        try:
            import fitz  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        tmp = tempfile.mkdtemp()
        assets = self._Path(tmp) / "out_assets"
        options = ConversionOptions(
            ocr="off",
            artifact_mode="both",
            assets_dir=assets,
            asset_base_dir=self._Path(tmp),
            jobs=0,
            math="on",
        )
        result = convert_pdf_to_result(self.SAMPLE_PDF, options)
        return result, assets

    def test_tables_render_as_pipe_tables_without_crop(self) -> None:
        result, assets = self._convert()
        md = result.markdown

        # Table 1 (page 4) and Table 2 (page 16) header rows.
        self.assertIn(
            "| Feature | Description (Generated by GPT-4) | Interpretability Score |",
            md,
        )
        self.assertIn(
            "| Moment | Correlation with top-random interpretability score |", md
        )
        # A known Table 1 body row and the hyphen-repaired wrapped cell.
        self.assertIn(
            "| 1-0000 | parts of individual names, especially last names. | 0.33 |",
            md,
        )
        self.assertRegex(md, r"\| 1-0003 \|[^|]*activation[^|]*\| 0\.57 \|")
        # Table 2 body rows.
        self.assertIn("| Kurtosis | 0.15 |", md)

        # No table PNG is referenced in the Markdown nor written to disk.
        self.assertNotRegex(md, r"page-\d{4}-table-\d{2}\.png")
        table_crops = list(assets.glob("*-table-*.png")) if assets.exists() else []
        self.assertEqual(table_crops, [])
        table_artifacts = [a for a in result.artifacts if a.kind == "table"]
        self.assertEqual(len(table_artifacts), 2)
        for artifact in table_artifacts:
            self.assertIsNone(artifact.asset_path)
            self.assertIsNotNone(artifact.table)

    def test_figures_keep_image_crops(self) -> None:
        result, assets = self._convert()
        md = result.markdown
        figure_artifacts = [a for a in result.artifacts if a.kind == "figure"]
        self.assertTrue(figure_artifacts)
        for artifact in figure_artifacts:
            self.assertIsNotNone(artifact.asset_path)
        # At least one figure crop is embedded in the Markdown.
        self.assertRegex(md, r"!\[[^\]]*\]\([^)]*page-\d{4}-figure-\d{2}\.png\)")

    def test_equations_stay_latex(self) -> None:
        result, _ = self._convert()
        md = result.markdown
        # Display math survives as fenced LaTeX and inline math as $...$.
        self.assertIn("```math", md)
        self.assertIn("\\mathrm{ReLU}", md)
        self.assertRegex(md, r"\$[^$]*\\mathbb\{R\}[^$]*\$")

    def test_table_cell_text_does_not_leak_to_prose(self) -> None:
        result, _ = self._convert()
        md = result.markdown
        # Each distinctive cell value occurs exactly once: inside its pipe table,
        # never duplicated as stray paragraph prose.
        self.assertEqual(md.count("1-0003"), 1)
        self.assertEqual(md.count("court case references"), 1)
        self.assertEqual(md.count("Kurtosis"), 1)


class _FakeRegion:
    def __init__(self, bbox, latex, kind, source_line_indices, number=None):
        self.bbox = bbox
        self.latex = latex
        self.kind = kind
        self.source_line_indices = source_line_indices
        self.number = number


class _FakeMathResult:
    def __init__(self, display=(), inline=()):
        self.display = display
        self.inline = inline


class MathResultsRenderingTests(unittest.TestCase):
    def test_display_equation_emitted_and_source_suppressed(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("input vector x, the network produces the output, given by",
                          y=80, x=108, size=10, block=1),
                make_line("c", y=98, x=256, size=10, block=2),
                make_line("=", y=98, x=271, size=10, block=2),
                make_line("ReLU(Mx + b)", y=98, x=289, size=10, block=2),
                make_line("(1)", y=98, x=492, size=10, block=2),
                make_line("where M is a learned parameter matrix used downstream.",
                          y=130, x=108, size=10, block=3),
            ],
        )
        # Display region covers the three equation-fragment lines (indices 1..3).
        region = _FakeRegion(
            BBox(255, 93, 360, 108),
            r"\mathbf{c} = \mathrm{ReLU}(M\mathbf{x} + \mathbf{b})",
            "display",
            (1, 2, 3),
            number="1",
        )
        math_results = {1: _FakeMathResult(display=(region,))}

        markdown = render_markdown([page], math_results=math_results)

        self.assertIn(
            "```math\n\\mathbf{c} = \\mathrm{ReLU}(M\\mathbf{x} + \\mathbf{b})\n```",
            markdown,
        )
        # The glyph-text fragments are suppressed: no stray "ReLU(Mx + b)" body
        # line and no lone "=" fence.
        self.assertNotIn("ReLU(Mx + b)", markdown)
        self.assertNotIn("```math\n=\n```", markdown)
        # Surrounding prose is preserved.
        self.assertIn("input vector x", markdown)
        self.assertIn("where M is a learned parameter", markdown)
        # The equation appears exactly once.
        self.assertEqual(markdown.count("```math"), 1)

    def test_inline_math_spliced_into_paragraph(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("the hidden layer has size dhid where", y=80, x=108, size=10, block=1),
            ],
        )
        line = page.lines[0]
        # The inline run covers the word "dhid" (find its x-range).
        word = next(w for w in line.words if w.text == "dhid")
        region = _FakeRegion(
            BBox(word.x_min - 0.5, line.y_min, word.x_max + 0.5, line.y_max),
            r"d_{\mathrm{hid}}",
            "inline",
            (0,),
        )
        math_results = {1: _FakeMathResult(inline=(region,))}

        markdown = render_markdown([page], math_results=math_results)

        self.assertIn(r"$d_{\mathrm{hid}}$", markdown)
        self.assertNotIn(" dhid ", markdown)
        self.assertIn("the hidden layer has size", markdown)
        self.assertIn("where", markdown)

    def test_math_off_uses_legacy_fallback(self) -> None:
        # With math_results=None (math disabled), the legacy character path is
        # used; a lone "=" still must not become its own math fence.
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                make_line("=", y=98, x=271, size=10, block=2),
                make_line("ReLU(Mx + b)", y=98, x=289, size=10, block=2),
                make_line("(1)", y=98, x=492, size=10, block=2),
            ],
        )

        markdown = render_markdown([page], math_results=None)

        self.assertNotIn("```math\n=\n```", markdown)


if __name__ == "__main__":
    unittest.main()

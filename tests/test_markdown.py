from __future__ import annotations

import unittest

from pdfmdlite.artifacts import BBox, Caption, VisualArtifact
from pdfmdlite.layout import DocumentLayout, Line, PageLayout, Word, remove_repeating_marginalia
from pdfmdlite.markdown import render_markdown


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
        self.assertIn("![Figure 1. Pipeline overview](output_assets/page-0001-figure-01.png)", markdown)
        self.assertLess(
            markdown.index("Figure 1. Pipeline overview"),
            markdown.index("output_assets/page-0001-figure-01.png"),
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdfmdlite import ConversionOptions, convert_pdf, convert_pdf_to_result
from pdfmdlite.layout import DocumentLayout, Line, PageLayout, Word
from pdfmdlite.poppler import PopplerError


@unittest.skipIf(shutil.which("pdftotext") is None, "pdftotext is not installed")
class PopplerIntegrationTests(unittest.TestCase):
    def test_minimal_pdf_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "sample.pdf"
            _write_minimal_pdf(pdf_path)

            markdown = convert_pdf(pdf_path, ConversionOptions())

        self.assertIn("# Sample Title", markdown)
        self.assertIn("This is a short paragraph.", markdown)


class TextEngineFallbackTests(unittest.TestCase):
    def test_auto_falls_back_to_pymupdf_when_poppler_fails(self) -> None:
        page = PageLayout(
            number=1,
            width=612,
            height=792,
            lines=[
                Line(
                    words=[
                        Word("Fallback", 72, 72, 120, 84),
                        Word("worked", 126, 72, 170, 84),
                    ],
                    page_number=1,
                    block_id=1,
                    source="pymupdf",
                )
            ],
            source="pymupdf",
        )
        document = DocumentLayout(pages=[page])

        with patch(
            "pdfmdlite.converter.extract_poppler_layout",
            side_effect=PopplerError("poppler crashed"),
        ), patch("pdfmdlite.converter.extract_pymupdf_layout", return_value=document):
            result = convert_pdf_to_result("dummy.pdf", ConversionOptions(text_engine="auto"))

        self.assertIn("Fallback worked", result.markdown)


@unittest.skipIf(
    shutil.which("pdftoppm") is None or shutil.which("tesseract") is None,
    "pdftoppm or tesseract is not installed",
)
class OcrCoordinateTests(unittest.TestCase):
    def test_ocr_layout_uses_pdf_point_coordinates(self) -> None:
        from pdfmdlite.ocr import ocr_page_to_layout

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "sample.pdf"
            _write_minimal_pdf(pdf_path)
            page = ocr_page_to_layout(pdf_path, 1, "eng")

        # The MediaBox is 612 x 792 points. OCR rasterizes at OCR_RENDER_DPI, so
        # it must scale Tesseract's pixel coordinates back to points; otherwise
        # geometry is in pixels and artifact crop rendering (which clips in
        # points) produces invalid rectangles and crashes.
        self.assertAlmostEqual(page.width, 612.0, delta=2.0)
        self.assertAlmostEqual(page.height, 792.0, delta=2.0)
        for word in page.words:
            self.assertLessEqual(word.x_max, page.width + 1.0)
            self.assertLessEqual(word.y_max, page.height + 1.0)


def _write_minimal_pdf(path: Path) -> None:
    stream = (
        b"BT\n"
        b"/F1 20 Tf\n"
        b"72 720 Td\n"
        b"(Sample Title) Tj\n"
        b"/F1 12 Tf\n"
        b"0 -36 Td\n"
        b"(This is a short paragraph.) Tj\n"
        b"ET\n"
    )
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        b"5 0 obj\n<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream\nendobj\n",
    ]

    content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content))
        content += obj
    xref_offset = len(content)
    content += b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    path.write_bytes(content)


if __name__ == "__main__":
    unittest.main()

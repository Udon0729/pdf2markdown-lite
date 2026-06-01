from __future__ import annotations

import csv
import shutil
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .layout import Line, PageLayout, Word

# DPI used to rasterize a page before Tesseract reads it. Tesseract reports
# pixel coordinates at this resolution, so OCR geometry must be divided by
# OCR_RENDER_DPI / 72 to match the PDF-point coordinate space the rest of the
# pipeline (layout heuristics, artifact crop rendering) assumes.
OCR_RENDER_DPI = 200


def ocr_page_to_layout(pdf_path: str | Path, page_number: int, lang: str) -> PageLayout:
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm was not found. Install Poppler first.")
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract was not found. Install Tesseract or use --ocr off.")

    pdf_path = Path(pdf_path)
    with tempfile.TemporaryDirectory(prefix="pdfmdlite-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        render = subprocess.run(
            [
                "pdftoppm",
                "-q",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                str(OCR_RENDER_DPI),
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if render.returncode != 0:
            detail = render.stderr.strip() or "pdftoppm failed"
            raise RuntimeError(detail)

        images = sorted(Path(tmp).glob("page-*.png"))
        if not images:
            return PageLayout(number=page_number, width=0, height=0, source="ocr")

        image_path = images[0]
        width, height = _png_size(image_path)
        tsv = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                lang,
                "--psm",
                "6",
                "tsv",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if tsv.returncode != 0:
            detail = tsv.stderr.strip() or "tesseract failed"
            raise RuntimeError(detail)

    scale = OCR_RENDER_DPI / 72.0
    page = PageLayout(
        number=page_number,
        width=width / scale,
        height=height / scale,
        source="ocr",
    )
    grouped: dict[tuple[int, int, int], list[Word]] = defaultdict(list)
    reader = csv.DictReader(tsv.stdout.splitlines(), delimiter="\t")
    for row in reader:
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or "-1")
        except ValueError:
            confidence = -1
        if confidence < 0:
            continue
        left = _to_float(row.get("left")) / scale
        top = _to_float(row.get("top")) / scale
        word_width = _to_float(row.get("width")) / scale
        word_height = _to_float(row.get("height")) / scale
        block_num = int(row.get("block_num") or 0)
        par_num = int(row.get("par_num") or 0)
        line_num = int(row.get("line_num") or 0)
        word_num = int(row.get("word_num") or 0)
        key = (block_num, par_num, line_num)
        grouped[key].append(
            Word(
                text=text,
                x_min=left,
                y_min=top,
                x_max=left + word_width,
                y_max=top + word_height,
                block_id=block_num,
                line_id=line_num,
                word_id=word_num,
            )
        )

    for block_id, key in enumerate(sorted(grouped)):
        words = sorted(grouped[key], key=lambda word: (word.x_min, word.word_id))
        page.lines.append(
            Line(words=words, page_number=page_number, block_id=block_id, source="ocr")
        )
    page.lines.sort(key=lambda line: (line.y_min, line.x_min))
    return page


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", header[16:24])
    return (0, 0)


def _to_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0

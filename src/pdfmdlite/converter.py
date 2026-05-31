from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    VisualArtifact,
    extract_artifacts,
)
from .layout import remove_repeating_marginalia
from .markdown import render_markdown
from .ocr import ocr_page_to_layout
from .poppler import PopplerError, extract_layout as extract_poppler_layout
from .pymupdf_text import extract_layout as extract_pymupdf_layout


@dataclass(frozen=True)
class ConversionOptions:
    ocr: str = "off"
    ocr_lang: str = "eng"
    text_engine: str = "pymupdf"
    strip_headers: bool = True
    page_breaks: bool = False
    detect_tables: bool = True
    min_text_words: int = 8
    first_page: int | None = None
    last_page: int | None = None
    artifact_mode: str = "off"
    extract_artifacts: bool = False
    assets_dir: Path | None = None
    asset_base_dir: Path | None = None
    artifact_dpi: int = 180
    jobs: int = 0


@dataclass(frozen=True)
class ConversionResult:
    markdown: str
    artifacts: list[VisualArtifact]


def convert_pdf(pdf_path: str | Path, options: ConversionOptions | None = None) -> str:
    return convert_pdf_to_result(pdf_path, options).markdown


def convert_pdf_to_result(
    pdf_path: str | Path,
    options: ConversionOptions | None = None,
) -> ConversionResult:
    options = options or ConversionOptions()
    pdf_path = Path(pdf_path)

    if options.ocr not in {"off", "auto", "force"}:
        raise ValueError("ocr must be one of: off, auto, force")
    if options.text_engine not in {"auto", "poppler", "pymupdf"}:
        raise ValueError("text_engine must be one of: auto, poppler, pymupdf")
    if options.artifact_mode not in {
        "off",
        "manifest",
        "embed",
        "both",
    }:
        raise ValueError(
            "artifact_mode must be one of: off, manifest, embed, both"
        )

    document = _extract_document_layout(pdf_path, options)

    if options.ocr != "off":
        for index, page in enumerate(document.pages):
            should_ocr = options.ocr == "force" or page.word_count < options.min_text_words
            if should_ocr:
                ocr_page = ocr_page_to_layout(pdf_path, page.number, options.ocr_lang)
                if ocr_page.word_count:
                    document.pages[index] = ocr_page

    if options.strip_headers:
        remove_repeating_marginalia(document.pages)

    artifacts: list[VisualArtifact] = []
    if options.extract_artifacts or options.artifact_mode != "off":
        artifacts = extract_artifacts(
            pdf_path,
            document.pages,
            assets_dir=options.assets_dir,
            dpi=options.artifact_dpi,
            jobs=options.jobs,
        )

    markdown = render_markdown(
        document.pages,
        page_breaks=options.page_breaks,
        detect_tables=options.detect_tables,
        artifacts=artifacts,
        embed_artifacts=options.artifact_mode in {"embed", "both"},
        asset_base_dir=options.asset_base_dir,
    )
    return ConversionResult(markdown=markdown, artifacts=artifacts)


def _extract_document_layout(pdf_path: Path, options: ConversionOptions):
    kwargs = {
        "first_page": options.first_page,
        "last_page": options.last_page,
    }
    if options.text_engine == "pymupdf":
        return extract_pymupdf_layout(pdf_path, **kwargs)
    if options.text_engine == "poppler":
        return extract_poppler_layout(pdf_path, **kwargs)

    try:
        return extract_poppler_layout(pdf_path, **kwargs)
    except PopplerError:
        return extract_pymupdf_layout(pdf_path, **kwargs)

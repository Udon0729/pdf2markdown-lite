from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .artifacts import write_artifacts_manifest
from .converter import ConversionOptions, convert_pdf_to_result
from .poppler import PopplerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfmdlite",
        description="Convert a PDF to Markdown using CPU-only layout heuristics.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("-o", "--output", type=Path, help="Output Markdown path.")
    parser.add_argument(
        "--ocr",
        choices=("off", "auto", "force"),
        default="off",
        help="OCR mode. 'auto' OCRs only pages with too little embedded text.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract language code, for example 'eng' or 'jpn'.",
    )
    parser.add_argument(
        "--text-engine",
        choices=("auto", "poppler", "pymupdf"),
        default="pymupdf",
        help=(
            "Text extraction backend. Defaults to PyMuPDF for speed. 'auto' "
            "tries Poppler first and falls back to PyMuPDF if Poppler fails."
        ),
    )
    parser.add_argument(
        "--keep-headers",
        action="store_true",
        help="Do not strip repeated headers, footers, or marginal page numbers.",
    )
    parser.add_argument(
        "--page-breaks",
        action="store_true",
        help="Insert HTML comments between pages.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable simple Markdown table reconstruction.",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        help="First 1-based page to process. Useful for chunking large PDFs.",
    )
    parser.add_argument(
        "--last-page",
        type=int,
        help="Last 1-based page to process. Useful for chunking large PDFs.",
    )
    parser.add_argument(
        "--extract-artifacts",
        action="store_true",
        help="Deprecated alias for --artifact-mode manifest.",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("off", "manifest", "embed", "both"),
        default="off",
        help=(
            "Figure/table/equation handling: off, manifest sidecar only, embed images in "
            "Markdown, or both embed and manifest. Non-off modes use high-recall "
            "PDF object extraction and require PyMuPDF."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Directory for extracted figure/table/equation crop files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON path for extracted figure/table metadata.",
    )
    parser.add_argument(
        "--artifact-dpi",
        type=int,
        default=180,
        help="DPI for extracted figure/table/equation crops.",
    )
    parser.add_argument(
        "--math",
        choices=("on", "off"),
        default="on",
        help=(
            "Equation handling: 'on' reconstructs deterministic LaTeX from the "
            "PDF glyph stream (display math as fenced math blocks, inline math as "
            "$...$); 'off' uses the legacy character-heuristic path."
        ),
    )
    parser.add_argument(
        "--inline-images",
        action="store_true",
        help=(
            "Embed figure images directly in the Markdown as base64 data URIs so "
            "the .md is a self-contained single file, instead of linking to "
            "external files in the assets directory."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "CPU worker processes for artifact extraction. 0 chooses automatically "
            "for long PDFs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        parser.error(f"input PDF does not exist: {args.pdf}")
    if not args.pdf.is_file():
        parser.error(f"input path is not a file: {args.pdf}")
    if args.first_page is not None and args.first_page < 1:
        parser.error("--first-page must be >= 1")
    if args.last_page is not None and args.last_page < 1:
        parser.error("--last-page must be >= 1")
    if (
        args.first_page is not None
        and args.last_page is not None
        and args.last_page < args.first_page
    ):
        parser.error("--last-page must be greater than or equal to --first-page")
    if args.artifact_dpi <= 0:
        parser.error("--artifact-dpi must be greater than 0")
    if args.jobs < 0:
        parser.error("--jobs must be 0 or greater")

    artifact_mode = args.artifact_mode
    if (
        artifact_mode == "off"
        and (args.extract_artifacts or args.manifest is not None or args.assets_dir is not None)
    ):
        artifact_mode = "manifest"

    assets_dir = args.assets_dir
    manifest_path = args.manifest
    asset_base_dir = args.output.parent if args.output else Path.cwd()
    if artifact_mode != "off":
        # In inline mode figures embed as base64 in the .md; no assets dir is
        # created or needed, so do not auto-derive one.
        if assets_dir is None and not args.inline_images:
            assets_dir = (
                args.output.parent / f"{args.output.stem}_assets"
                if args.output
                else Path("assets")
            )
        if artifact_mode in {"manifest", "both"} and manifest_path is None:
            manifest_path = (
                args.output.with_suffix(".artifacts.json")
                if args.output
                else Path("artifacts.json")
            )

    options = ConversionOptions(
        ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        text_engine=args.text_engine,
        strip_headers=not args.keep_headers,
        page_breaks=args.page_breaks,
        detect_tables=not args.no_tables,
        first_page=args.first_page,
        last_page=args.last_page,
        artifact_mode=artifact_mode,
        extract_artifacts=args.extract_artifacts,
        assets_dir=assets_dir,
        asset_base_dir=asset_base_dir,
        artifact_dpi=args.artifact_dpi,
        jobs=args.jobs,
        math=args.math,
        inline_images=args.inline_images,
    )

    try:
        result = convert_pdf_to_result(args.pdf, options)
    except PopplerError as exc:
        print(f"pdfmdlite: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"pdfmdlite: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.markdown, encoding="utf-8")
    else:
        print(result.markdown)

    should_write_manifest = (
        artifact_mode in {"manifest", "both"} or args.manifest is not None
    )
    if should_write_manifest and manifest_path is not None:
        write_artifacts_manifest(
            manifest_path,
            pdf_path=args.pdf,
            artifacts=result.artifacts,
        )
    return 0

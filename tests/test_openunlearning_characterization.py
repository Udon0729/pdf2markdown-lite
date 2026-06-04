"""Characterization tests that pin the converter's output on a real paper.

The input PDF (an OpenUnlearning paper) is **not** committed: per the repo
convention, local test PDFs are git-ignored. Instead the *current* converter
output is committed as a golden fixture, and these tests re-run the converter on
the local PDF and assert the output is byte-for-byte unchanged. They self-skip
when the PDF is absent (a fresh clone or CI), mirroring the Poppler integration
test that skips without ``pdftotext``.

Why this exists: a regression safety net for refactors and performance work.

* ``test_text_markdown_matches_golden`` pins text extraction, reading order,
  heading detection, paragraph joins and the deterministic math -> LaTeX
  reconstruction (equations whose glyphs are outside the symbol tables stay as
  source text, so a future table-extension shows up as a diff here).
* ``test_artifact_metadata_matches_golden`` pins region detection (figure/table
  bounding boxes -- the output of the drawing-rect clustering) and table-grid
  reconstruction (cell matrices). An optimization of the region clustering can
  therefore be validated as output-preserving against this golden.

Determinism: the artifact run forces ``jobs=1`` (serial) and the goldens record
rounded bboxes and the table cell text only -- never PNG bytes or asset paths --
so the comparison is stable across runs and machines on a fixed PyMuPDF build.

Regenerate the goldens after an *intended* behavior change (review the diff):

    uv run --no-editable python tests/test_openunlearning_characterization.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

from pdfmdlite.converter import ConversionOptions, convert_pdf_to_result

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / "2026_OpenUnlearning_Accelerati.pdf"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEXT_GOLDEN = FIXTURES / "openunlearning_text.golden.md"
ARTIFACTS_GOLDEN = FIXTURES / "openunlearning_artifacts.golden.json"


def _run_text() -> str:
    """Markdown from the text + LaTeX path (no figure rendering; fast)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = convert_pdf_to_result(
            PDF_PATH, ConversionOptions(math="on", artifact_mode="off")
        )
    return result.markdown


def _run_artifacts() -> list[dict[str, Any]]:
    """Artifact metadata from the full extraction path (serial == deterministic)."""
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = convert_pdf_to_result(
                PDF_PATH,
                ConversionOptions(
                    math="on",
                    artifact_mode="both",
                    jobs=1,
                    assets_dir=Path(tmp) / "assets",
                    asset_base_dir=Path(tmp),
                ),
            )
        return _artifact_summary(result.artifacts)


def _artifact_summary(artifacts: list[Any]) -> list[dict[str, Any]]:
    """A deterministic, reviewable digest of the extracted artifacts.

    Excludes PNG bytes and asset paths (non-deterministic / environment-specific)
    and rounds bbox coordinates to 0.1pt to absorb float jitter.
    """
    summary: list[dict[str, Any]] = []
    for artifact in artifacts:
        caption = getattr(artifact, "caption", None)
        grid = getattr(artifact, "table", None)
        table = None
        if grid is not None:
            table = {
                "nrows": len(grid.matrix),
                "ncols": grid.ncols,
                "header_rows": grid.header_rows,
                "matrix": [list(row) for row in grid.matrix],
            }
        bbox = artifact.bbox
        summary.append(
            {
                "id": artifact.id,
                "page": artifact.page,
                "kind": artifact.kind,
                "bbox": [round(float(v), 1) for v in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)],
                "caption": getattr(caption, "text", None),
                "table": table,
            }
        )
    return summary


_REGEN_HINT = (
    "If this change is intended, regenerate the golden:\n"
    "    uv run --no-editable python tests/test_openunlearning_characterization.py\n"
    "then review the diff before committing."
)


@unittest.skipUnless(PDF_PATH.exists(), f"input PDF not present: {PDF_PATH.name}")
class OpenUnlearningCharacterizationTests(unittest.TestCase):
    maxDiff = 8000

    def test_text_markdown_matches_golden(self) -> None:
        self.assertTrue(
            TEXT_GOLDEN.exists(),
            f"golden missing: {TEXT_GOLDEN.name}\n{_REGEN_HINT}",
        )
        expected = TEXT_GOLDEN.read_text(encoding="utf-8")
        actual = _run_text()
        self.assertMultiLineEqual(expected, actual, msg=_REGEN_HINT)

    def test_artifact_metadata_matches_golden(self) -> None:
        self.assertTrue(
            ARTIFACTS_GOLDEN.exists(),
            f"golden missing: {ARTIFACTS_GOLDEN.name}\n{_REGEN_HINT}",
        )
        expected = json.loads(ARTIFACTS_GOLDEN.read_text(encoding="utf-8"))
        actual = _run_artifacts()
        self.assertEqual(expected, actual, msg=_REGEN_HINT)


def _regenerate() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"cannot regenerate: input PDF absent at {PDF_PATH}")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    TEXT_GOLDEN.write_text(_run_text(), encoding="utf-8")
    ARTIFACTS_GOLDEN.write_text(
        json.dumps(_run_artifacts(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {TEXT_GOLDEN.relative_to(REPO_ROOT)}")
    print(f"wrote {ARTIFACTS_GOLDEN.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    _regenerate()

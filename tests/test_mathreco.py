from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from pdfmdlite.mathreco import (
    LARGEOP_PLACEHOLDER,
    MathGlyph,
    MathRecoUnhandled,
    Rule,
    check_region_conservation,
    check_symbol_conservation,
    classify,
    font_family,
    is_math_font,
    latex_token_multiset,
    recognize_latex,
    reconstruct_math,
    source_token_multiset,
)
from pdfmdlite.mathreco import _RawLine, _detect_inline_regions

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "2309.08600v3.pdf"


def glyph(
    char: str,
    font: str,
    *,
    size: float = 10.0,
    flags: int = 0,
    gid: int = -1,
    ox: float = 100.0,
    oy: float = 100.0,
) -> MathGlyph:
    return MathGlyph(
        char=char,
        font=font,
        size=size,
        flags=flags,
        ox=ox,
        oy=oy,
        x0=ox,
        y0=oy - size,
        x1=ox + size * 0.5,
        y1=oy,
        gid=gid,
    )


def real_glyph(
    char: str,
    font: str,
    *,
    size: float,
    ox: float,
    oy: float,
    x0: float,
    x1: float,
    gid: int = -1,
    flags: int = 0,
) -> MathGlyph:
    """Build a glyph from the real page-3 dump (x0/x1 measured, y from size)."""
    return MathGlyph(
        char=char,
        font=font,
        size=size,
        flags=flags,
        ox=ox,
        oy=oy,
        x0=x0,
        y0=oy - size,
        x1=x1,
        y1=oy,
        gid=gid,
    )


class FontFamilyTests(unittest.TestCase):
    def test_strips_subset_prefix(self) -> None:
        self.assertEqual(font_family("ABCDEF+CMR10"), "CMR")
        self.assertEqual(font_family("CMBX10"), "CMBX")

    def test_is_math_font(self) -> None:
        self.assertTrue(is_math_font("CMMI10"))
        self.assertTrue(is_math_font("MSBM10"))
        self.assertTrue(is_math_font("CMEX10"))
        self.assertFalse(is_math_font("NimbusRomNo9L-Regu"))


class ClassifyTests(unittest.TestCase):
    def test_cmex_large_operator_maps_to_sum(self) -> None:
        token, style = classify(glyph("X", "CMEX10", gid=65533))
        self.assertEqual(token, LARGEOP_PLACEHOLDER)
        self.assertEqual(token, r"\sum")
        self.assertEqual(style, "largeop")

    def test_cmbx_maps_to_mathbf(self) -> None:
        for char in ("c", "x", "b", "f"):
            token, style = classify(glyph(char, "CMBX10", flags=20, gid=ord(char)))
            self.assertEqual(token, char)
            self.assertEqual(style, "mathbf")

    def test_cmmi_letter_is_bare(self) -> None:
        for char in ("M", "c"):
            token, style = classify(glyph(char, "CMMI10", flags=6, gid=ord(char)))
            self.assertEqual(token, char)
            self.assertEqual(style, "italic_bare")

    def test_cmr_multi_letter_run_is_upright(self) -> None:
        for char in ("R", "e", "L", "U"):
            token, style = classify(glyph(char, "CMR10", flags=4, gid=ord(char)))
            self.assertEqual(token, char)
            self.assertEqual(style, "upright")

    def test_cmsy_minus_and_symbols(self) -> None:
        cases = {
            "−": "-",
            "∈": r"\in",
            "≈": r"\approx",
            "⊂": r"\subset",
        }
        for char, expected in cases.items():
            token, style = classify(glyph(char, "CMSY10", flags=4))
            self.assertEqual(token, expected)
            self.assertEqual(style, "symbol")

    def test_cmsy7_prime_times_ast(self) -> None:
        self.assertEqual(classify(glyph("′", "CMSY7"))[0], r"\prime")
        self.assertEqual(classify(glyph("×", "CMSY7"))[0], r"\times")
        self.assertEqual(classify(glyph("∗", "CMSY7"))[0], r"\ast")

    def test_cmsy_slot_overrides(self) -> None:
        self.assertEqual(classify(glyph("?", "CMSY10", gid=123))[0], r"\{")
        self.assertEqual(classify(glyph("?", "CMSY10", gid=125))[0], r"\}")
        self.assertEqual(classify(glyph("?", "CMSY10", gid=124))[0], "|")
        self.assertEqual(classify(glyph("?", "CMSY10", gid=76))[0], r"\mathcal{L}")

    def test_msbm_blackboard_r(self) -> None:
        token, style = classify(glyph("R", "MSBM10", gid=82))
        self.assertEqual(token, "R")
        self.assertEqual(style, "mathbb")

    def test_accent_glyph_maps_to_hat(self) -> None:
        token, _ = classify(glyph("ˆ", "CMBX10", flags=20, gid=710))
        self.assertEqual(token, r"\hat")

    def test_bar_tilde_accents(self) -> None:
        self.assertEqual(classify(glyph("¯", "CMR10"))[0], r"\bar")
        self.assertEqual(classify(glyph("˜", "CMR10"))[0], r"\tilde")

    def test_unhandled_glyph_raises(self) -> None:
        with self.assertRaises(MathRecoUnhandled):
            classify(glyph("∰", "MysteryFont12", gid=9999))


# ---------------------------------------------------------------------------
# Structure recognition: hand-built fixtures from the page-3 glyph dump.
# ---------------------------------------------------------------------------


def eq1_glyphs() -> list[MathGlyph]:
    # c = ReLU(Mx + b)   (the (1) equation number is excluded.)
    return [
        real_glyph("c", "CMBX10", size=9.963, ox=255.82, oy=105.80, x0=255.82, x1=260.91),
        real_glyph("=", "CMR10", size=9.963, ox=270.87, oy=105.80, x0=270.87, x1=278.63),
        real_glyph("R", "CMR10", size=9.963, ox=288.59, oy=105.80, x0=288.59, x1=295.92),
        real_glyph("e", "CMR10", size=9.963, ox=295.92, oy=105.80, x0=295.92, x1=300.34),
        real_glyph("L", "CMR10", size=9.963, ox=300.34, oy=105.80, x0=300.34, x1=306.57),
        real_glyph("U", "CMR10", size=9.963, ox=306.57, oy=105.80, x0=306.57, x1=314.04),
        real_glyph("(", "CMR10", size=9.963, ox=314.04, oy=105.80, x0=314.04, x1=317.92),
        real_glyph("M", "CMMI10", size=9.963, ox=317.92, oy=105.80, x0=317.92, x1=327.58),
        real_glyph("x", "CMBX10", size=9.963, ox=328.67, oy=105.80, x0=328.67, x1=334.72),
        real_glyph("+", "CMR10", size=9.963, ox=336.93, oy=105.80, x0=336.93, x1=344.68),
        real_glyph("b", "CMBX10", size=9.963, ox=346.89, oy=105.80, x0=346.89, x1=353.26),
        real_glyph(")", "CMR10", size=9.963, ox=353.26, oy=105.80, x0=353.26, x1=357.14),
    ]


def eq2_glyphs() -> list[MathGlyph]:
    # x-hat = M^T c
    return [
        real_glyph("ˆ", "CMBX10", size=9.963, ox=255.02, oy=121.15, x0=255.02, x1=260.75, gid=710),
        real_glyph("x", "CMBX10", size=9.963, ox=254.87, oy=121.15, x0=254.87, x1=260.91),
        real_glyph("=", "CMR10", size=9.963, ox=270.87, oy=121.15, x0=270.87, x1=278.63),
        real_glyph("M", "CMMI10", size=9.963, ox=288.59, oy=121.15, x0=288.59, x1=298.25),
        real_glyph("T", "CMMI7", size=6.974, ox=299.34, oy=117.04, x0=299.34, x1=304.05),
        real_glyph("c", "CMBX10", size=9.963, ox=305.61, oy=121.15, x0=305.61, x1=310.70),
    ]


def eq3_glyphs() -> list[MathGlyph]:
    # = sum_{i=0}^{d_{hid}-1} c_i f_i
    return [
        real_glyph("=", "CMR10", size=9.963, ox=270.87, oy=145.37, x0=270.87, x1=278.63),
        # upper limit: d_{hid} - 1
        real_glyph("d", "CMMI7", size=6.974, ox=288.59, oy=132.75, x0=288.59, x1=292.74),
        real_glyph("h", "NimbusRomNo9L-Regu", size=4.981, ox=292.73, oy=133.74, x0=292.73, x1=295.22),
        real_glyph("i", "NimbusRomNo9L-Regu", size=4.981, ox=295.22, oy=133.74, x0=295.22, x1=296.61),
        real_glyph("d", "NimbusRomNo9L-Regu", size=4.981, ox=296.61, oy=133.74, x0=296.61, x1=299.10),
        real_glyph("−", "CMSY7", size=6.974, ox=299.60, oy=132.75, x0=299.60, x1=305.83, gid=8722),
        real_glyph("1", "CMR7", size=6.974, ox=305.82, oy=132.75, x0=305.82, x1=309.80),
        # the large operator
        real_glyph("X", "CMEX10", size=9.963, ox=292.00, oy=135.90, x0=292.00, x1=306.39, gid=88),
        # lower limit: i = 0
        real_glyph("i", "CMMI7", size=6.974, ox=292.74, oy=157.12, x0=292.74, x1=295.56),
        real_glyph("=", "CMR7", size=6.974, ox=295.56, oy=157.12, x0=295.56, x1=301.67),
        real_glyph("0", "CMR7", size=6.974, ox=301.67, oy=157.12, x0=301.67, x1=305.65),
        # body: c_i (bold) f_i
        real_glyph("c", "CMMI10", size=9.963, ox=311.46, oy=145.37, x0=311.46, x1=315.77),
        real_glyph("i", "CMMI7", size=6.974, ox=315.77, oy=146.86, x0=315.77, x1=318.58),
        real_glyph("f", "CMBX10", size=9.963, ox=319.08, oy=145.37, x0=319.08, x1=322.58),
        real_glyph("i", "CMMI7", size=6.974, ox=322.58, oy=146.86, x0=322.58, x1=325.40),
    ]


def _normalise(latex: str) -> str:
    """Collapse spacing so trivial space differences do not fail an oracle."""
    return " ".join(latex.split())


class OracleTests(unittest.TestCase):
    def test_oracle_eq1(self) -> None:
        latex = recognize_latex(eq1_glyphs())
        self.assertEqual(
            _normalise(latex),
            r"\mathbf{c} = \mathrm{ReLU}(M\mathbf{x} + \mathbf{b})",
        )

    def test_oracle_eq2(self) -> None:
        latex = recognize_latex(eq2_glyphs())
        self.assertEqual(_normalise(latex), r"\hat{\mathbf{x}} = M^{T}\mathbf{c}")

    def test_oracle_eq3(self) -> None:
        latex = recognize_latex(eq3_glyphs())
        self.assertEqual(
            _normalise(latex),
            r"= \sum_{i=0}^{d_{\mathrm{hid}}-1} c_{i}\mathbf{f}_{i}",
        )


class StructureUnitTests(unittest.TestCase):
    def test_superscript_by_raised_smaller_glyph(self) -> None:
        glyphs = [
            glyph("M", "CMMI10", ox=100.0, oy=120.0, flags=6),
            glyph("T", "CMMI7", size=6.974, ox=110.0, oy=116.0),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"M^{T}")

    def test_subscript_by_lowered_smaller_glyph(self) -> None:
        glyphs = [
            glyph("c", "CMMI10", ox=100.0, oy=120.0, flags=6),
            glyph("i", "CMMI7", size=6.974, ox=106.0, oy=122.0),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"c_{i}")

    def test_sub_and_sup_at_same_x_discriminated_by_sign(self) -> None:
        # The L2-norm exponent: two '2' glyphs at the SAME x, one raised
        # (superscript), one lowered (subscript); only the sign of the baseline
        # shift tells them apart.
        glyphs = [
            real_glyph("|", "CMSY10", size=9.963, ox=279.81, oy=254.27, x0=279.81, x1=282.0, gid=124),
            real_glyph("|", "CMSY10", size=9.963, ox=282.58, oy=254.27, x0=282.58, x1=284.8, gid=124),
            real_glyph("x", "CMBX10", size=9.963, ox=285.34, oy=254.27, x0=285.34, x1=291.0),
            real_glyph("−", "CMSY10", size=9.963, ox=293.60, oy=254.27, x0=293.60, x1=301.0, gid=8722),
            real_glyph("ˆ", "CMBX10", size=9.963, ox=303.73, oy=254.27, x0=303.73, x1=309.0, gid=710),
            real_glyph("x", "CMBX10", size=9.963, ox=303.57, oy=254.27, x0=303.57, x1=309.6),
            real_glyph("|", "CMSY10", size=9.963, ox=309.61, oy=254.27, x0=309.61, x1=311.8, gid=124),
            real_glyph("|", "CMSY10", size=9.963, ox=312.38, oy=254.27, x0=312.38, x1=314.6, gid=124),
            real_glyph("2", "CMR7", size=6.974, ox=315.15, oy=250.16, x0=315.15, x1=319.0),
            real_glyph("2", "CMR7", size=6.974, ox=315.15, oy=256.74, x0=315.15, x1=319.0),
        ]
        self.assertEqual(
            _normalise(recognize_latex(glyphs)),
            r"||\mathbf{x} - \hat{\mathbf{x}}||_{2}^{2}",
        )

    def test_hat_accent_over_base(self) -> None:
        glyphs = [
            glyph("ˆ", "CMBX10", flags=20, gid=710, ox=255.02, oy=121.15),
            glyph("x", "CMBX10", flags=20, ox=254.87, oy=121.15),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"\hat{\mathbf{x}}")

    def test_large_operator_prod_with_limits(self) -> None:
        # \prod_{i=1}^{n}: CMEX gid 89 = \prod, identified by font+gid not char.
        glyphs = [
            real_glyph("Y", "CMEX10", size=9.963, ox=100.0, oy=120.0, x0=100.0, x1=112.0, gid=89),
            real_glyph("n", "CMMI7", size=6.974, ox=103.0, oy=108.0, x0=103.0, x1=108.0),
            real_glyph("i", "CMMI7", size=6.974, ox=101.0, oy=132.0, x0=101.0, x1=103.8),
            real_glyph("=", "CMR7", size=6.974, ox=103.8, oy=132.0, x0=103.8, x1=109.0),
            real_glyph("1", "CMR7", size=6.974, ox=109.0, oy=132.0, x0=109.0, x1=113.0),
        ]
        self.assertEqual(
            _normalise(recognize_latex(glyphs)),
            r"\prod_{i=1}^{n}",
        )


class InlineStructureTests(unittest.TestCase):
    """Inline-math defect fixes (D1/D2/D3), exercised on hand-built glyph runs.

    Inline runs flow through the SAME recognizer as display math (recognize_latex
    -> _row_size/_dominant_baseline/_emit_row), so these fixtures pin the shared
    geometry that the inline path depends on.
    """

    def test_inline_subscript_group_a_ij(self) -> None:
        # D2: 'a' with an i,j subscript group must group as a_{i,j}, not flatten
        # to 'ai,j'. Geometry taken from the real page-1 inline run: body 'a' at
        # oy 558.38 (size 9.96), subscript i/,/j at oy 559.88 (size 6.97). The
        # cluster average baseline (~559.38) mis-classed the subscripts as base;
        # the full-size-median baseline (558.38) makes the +1.50pt shift a sub.
        glyphs = [
            real_glyph("a", "CMMI10", size=9.963, ox=100.00, oy=558.38, x0=100.00, x1=105.30),
            real_glyph("i", "CMMI7", size=6.974, ox=105.30, oy=559.88, x0=105.30, x1=108.10),
            real_glyph(",", "CMMI7", size=6.974, ox=108.10, oy=559.88, x0=108.10, x1=110.00),
            real_glyph("j", "CMMI7", size=6.974, ox=110.00, oy=559.88, x0=110.00, x1=112.80),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"a_{i,j}")

    def test_inline_subscript_group_d_in(self) -> None:
        # D2 (+ D3): 'd' with a text-font 'in' subscript -> d_{\mathrm{in}}; the
        # multi-letter upright subscript run is wrapped in \mathrm.
        glyphs = [
            real_glyph("d", "CMMI10", size=9.963, ox=100.00, oy=633.00, x0=100.00, x1=105.20),
            real_glyph("i", "NimbusRomNo9L-Regu", size=6.974, ox=105.20, oy=634.50, x0=105.20, x1=107.30),
            real_glyph("n", "NimbusRomNo9L-Regu", size=6.974, ox=107.30, oy=634.50, x0=107.30, x1=111.80),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"d_{\mathrm{in}}")

    def test_inline_subscript_group_d_hid(self) -> None:
        # D2: a three-letter 'hid' subscript still groups under one base 'd'.
        glyphs = [
            real_glyph("d", "CMMI10", size=9.963, ox=100.00, oy=633.00, x0=100.00, x1=105.20),
            real_glyph("h", "NimbusRomNo9L-Regu", size=6.974, ox=105.20, oy=634.50, x0=105.20, x1=108.70),
            real_glyph("i", "NimbusRomNo9L-Regu", size=6.974, ox=108.70, oy=634.50, x0=108.70, x1=110.80),
            real_glyph("d", "NimbusRomNo9L-Regu", size=6.974, ox=110.80, oy=634.50, x0=110.80, x1=114.30),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"d_{\mathrm{hid}}")

    def test_inline_subscript_group_n_vec(self) -> None:
        # D2: 'n' (math italic) with a text-font 'vec' subscript -> n_{\mathrm{vec}}.
        glyphs = [
            real_glyph("n", "CMMI7", size=6.974, ox=100.00, oy=560.00, x0=100.00, x1=104.50),
            real_glyph("v", "NimbusRomNo9L-Regu", size=4.981, ox=104.50, oy=561.00, x0=104.50, x1=106.50),
            real_glyph("e", "NimbusRomNo9L-Regu", size=4.981, ox=106.50, oy=561.00, x0=106.50, x1=108.50),
            real_glyph("c", "NimbusRomNo9L-Regu", size=4.981, ox=108.50, oy=561.00, x0=108.50, x1=110.50),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"n_{\mathrm{vec}}")

    def test_cmmi_operator_name_run_is_mathrm(self) -> None:
        # D3: four ADJACENT same-size CMMI (math italic) letters spell an operator
        # name and must be set upright with \mathrm, matching the CMR spelling of
        # ReLU elsewhere in the document.
        glyphs = [
            real_glyph("R", "CMMI10", size=9.963, ox=100.00, oy=120.00, x0=100.00, x1=107.30),
            real_glyph("e", "CMMI10", size=9.963, ox=107.30, oy=120.00, x0=107.30, x1=111.72),
            real_glyph("L", "CMMI10", size=9.963, ox=111.72, oy=120.00, x0=111.72, x1=117.95),
            real_glyph("U", "CMMI10", size=9.963, ox=117.95, oy=120.00, x0=117.95, x1=125.42),
        ]
        self.assertEqual(_normalise(recognize_latex(glyphs)), r"\mathrm{ReLU}")

    def test_single_cmmi_letter_stays_bare_italic(self) -> None:
        # D3 guard: a LONE math-italic letter is a variable, not an operator name,
        # so it stays bare (no \mathrm). This pins the run-length>=2 condition.
        self.assertEqual(_normalise(recognize_latex([glyph("M", "CMMI10")])), "M")

    def test_lone_cmsy_brace_is_not_inline_math(self) -> None:
        # D1: a lone CMSY delimiter brace embedded in a text line (the author
        # '{...}@gmail.com' line) must NOT be promoted to inline math.
        text_font = "NimbusRomNo9L-Regu"
        glyphs = [
            real_glyph("x", text_font, size=9.963, ox=100.0, oy=560.0, x0=100.0, x1=105.0),
            real_glyph("{", "CMSY10", size=9.963, ox=110.0, oy=560.0, x0=110.0, x1=115.0, gid=123),
            real_glyph("y", text_font, size=9.963, ox=120.0, oy=560.0, x0=120.0, x1=125.0),
            real_glyph("}", "CMSY10", size=9.963, ox=130.0, oy=560.0, x0=130.0, x1=135.0, gid=125),
        ]
        line = _RawLine(tuple(glyphs), 100.0, 550.0, 135.0, 560.0)
        regions = _detect_inline_regions([line], body_size=9.963)
        self.assertEqual(regions, [])

    def test_cmsy_brace_with_math_content_still_inline(self) -> None:
        # D1 must not over-reject: a CMSY brace flanking real math content (CMBX)
        # is a genuine inline expression and still yields a region.
        glyphs = [
            real_glyph("{", "CMSY10", size=9.963, ox=100.0, oy=560.0, x0=100.0, x1=104.0, gid=123),
            real_glyph("g", "CMBX10", size=9.963, ox=104.0, oy=560.0, x0=104.0, x1=110.0),
            real_glyph("}", "CMSY10", size=9.963, ox=110.0, oy=560.0, x0=110.0, x1=114.0, gid=125),
        ]
        line = _RawLine(tuple(glyphs), 100.0, 550.0, 114.0, 560.0)
        regions = _detect_inline_regions([line], body_size=9.963)
        self.assertEqual(len(regions), 1)
        self.assertEqual(_normalise(regions[0][1]), r"\{\mathbf{g}\}")


class ConservationTests(unittest.TestCase):
    def test_eq1_conserves_glyphs(self) -> None:
        glyphs = eq1_glyphs()
        latex = recognize_latex(glyphs)
        self.assertTrue(check_symbol_conservation(glyphs, latex))

    def test_eq2_conserves_glyphs(self) -> None:
        # The accent glyph is layout (it becomes \hat scaffolding), so it is
        # excluded from both multisets and conservation still holds.
        glyphs = eq2_glyphs()
        latex = recognize_latex(glyphs)
        self.assertTrue(check_symbol_conservation(glyphs, latex))

    def test_eq3_conserves_glyphs(self) -> None:
        glyphs = eq3_glyphs()
        latex = recognize_latex(glyphs)
        self.assertTrue(check_symbol_conservation(glyphs, latex))

    def test_conservation_detects_a_dropped_glyph(self) -> None:
        glyphs = eq1_glyphs()
        # Recognise the full equation, then check it against a SHORTER source
        # (the conservation invariant must fail when a glyph is missing).
        latex = recognize_latex(glyphs)
        self.assertFalse(check_symbol_conservation(glyphs[:-2], latex))

    def test_multiset_splits_upright_run_into_letters(self) -> None:
        glyphs = eq1_glyphs()
        source = source_token_multiset(glyphs)
        out = latex_token_multiset(recognize_latex(glyphs))
        # ReLU contributes one of each letter on both sides.
        for letter in ("R", "e", "L", "U"):
            self.assertEqual(source.get(letter, 0), out.get(letter, 0))


def eq4_underbrace_glyphs() -> list[MathGlyph]:
    # L(x) = underbrace{||x - x_hat||_2^2}_{Reconstruction loss}
    #        + underbrace{alpha ||c||_1}_{Sparsity loss}
    # built from the real page-3 dump; the CMEX '|{z}|' pieces draw the braces
    # and the small NimbusRom glyphs below them are the labels.
    main = [
        real_glyph("L", "CMSY10", size=9.963, ox=238.4, oy=254.27, x0=238.4, x1=245.2, gid=76),
        real_glyph("(", "CMR10", size=9.963, ox=245.2, oy=254.27, x0=245.2, x1=249.1),
        real_glyph("x", "CMBX10", size=9.963, ox=249.1, oy=254.27, x0=249.1, x1=255.1),
        real_glyph(")", "CMR10", size=9.963, ox=255.1, oy=254.27, x0=255.1, x1=259.0),
        real_glyph("=", "CMR10", size=9.963, ox=261.8, oy=254.27, x0=261.8, x1=269.5),
        real_glyph("|", "CMSY10", size=9.963, ox=279.8, oy=254.27, x0=279.8, x1=282.6, gid=124),
        real_glyph("|", "CMSY10", size=9.963, ox=282.6, oy=254.27, x0=282.6, x1=285.3, gid=124),
        real_glyph("x", "CMBX10", size=9.963, ox=285.3, oy=254.27, x0=285.3, x1=291.4),
        real_glyph("−", "CMSY10", size=9.963, ox=293.6, oy=254.27, x0=293.6, x1=301.4, gid=8722),
        real_glyph("x", "CMBX10", size=9.963, ox=303.6, oy=254.27, x0=303.6, x1=309.6),
        real_glyph("ˆ", "CMBX10", size=9.963, ox=303.7, oy=254.27, x0=303.7, x1=309.5, gid=710),
        real_glyph("|", "CMSY10", size=9.963, ox=309.6, oy=254.27, x0=309.6, x1=312.4, gid=124),
        real_glyph("|", "CMSY10", size=9.963, ox=312.4, oy=254.27, x0=312.4, x1=315.2, gid=124),
        real_glyph("+", "CMR10", size=9.963, ox=328.8, oy=254.27, x0=328.8, x1=336.5),
        real_glyph("α", "CMMI10", size=9.963, ox=342.4, oy=254.27, x0=342.4, x1=348.8),
        real_glyph("|", "CMSY10", size=9.963, ox=348.8, oy=254.27, x0=348.8, x1=351.6, gid=124),
        real_glyph("|", "CMSY10", size=9.963, ox=351.6, oy=254.27, x0=351.6, x1=354.3, gid=124),
        real_glyph("c", "CMBX10", size=9.963, ox=354.3, oy=254.27, x0=354.3, x1=359.4),
        real_glyph("|", "CMSY10", size=9.963, ox=359.4, oy=254.27, x0=359.4, x1=362.2, gid=124),
        real_glyph("|", "CMSY10", size=9.963, ox=362.2, oy=254.27, x0=362.2, x1=365.0, gid=124),
        # ^2 (raised) and _2 (lowered) of the first norm, at the same x
        real_glyph("2", "CMR7", size=6.974, ox=315.1, oy=250.16, x0=315.1, x1=319.1),
        real_glyph("2", "CMR7", size=6.974, ox=315.1, oy=256.74, x0=315.1, x1=319.1),
        # _1 of the second norm
        real_glyph("1", "CMR7", size=6.974, ox=365.0, oy=255.77, x0=365.0, x1=368.9),
    ]
    def brace(ch: str, x0: float, x1: float) -> MathGlyph:
        # The CMEX brace bbox sits below the main baseline (origin y 260.9,
        # bbox 260.3..270.3 from the real dump), unlike a normal glyph whose
        # bbox is above its origin -- so build it explicitly.
        return MathGlyph(ch, "CMEX10", 9.963, 0, x0, 260.9, x0, 260.3, x1, 270.3, -1)

    braces = [
        brace("|", 279.8, 284.3),
        brace("{", 295.2, 299.7),
        brace("z", 299.7, 304.2),
        brace("}", 315.1, 319.6),
        brace("|", 342.4, 346.9),
        brace("{", 351.4, 355.9),
        brace("z", 355.9, 360.4),
        brace("}", 365.0, 369.4),
    ]

    def label(text: str, start: float) -> list[MathGlyph]:
        glyphs = []
        cursor = start
        for word in text.split():
            for ch in word:
                glyphs.append(
                    MathGlyph(ch, "NimbusRomNo9L-Regu", 6.974, 0, cursor, 270.4,
                              cursor, 264.3, cursor + 3.0, 270.4, -1)
                )
                cursor += 3.1
            cursor += 2.0
        return glyphs

    return main + braces + label("Reconstruction loss", 272.3) + label("Sparsity loss", 338.2)


class UnderbraceStructureTests(unittest.TestCase):
    def test_synthetic_underbrace_recognised(self) -> None:
        glyphs = eq4_underbrace_glyphs()
        latex = recognize_latex(glyphs, [])
        # Two underbraces, each with a \text label and the norm body inside.
        self.assertEqual(latex.count(r"\underbrace{"), 2)
        self.assertIn(r"\underbrace{||\mathbf{x} - \hat{\mathbf{x}}||_{2}^{2}}", latex)
        self.assertIn(r"\underbrace{\alpha||\mathbf{c}||_{1}}", latex)
        self.assertIn(r"\mathcal{L}", latex)
        self.assertIn(r"\text{", latex)

    def test_underbrace_conserves_content(self) -> None:
        glyphs = eq4_underbrace_glyphs()
        latex = recognize_latex(glyphs, [])
        # Brace pieces and labels are decoration; math content is conserved.
        self.assertTrue(check_region_conservation(glyphs, latex))

    def test_cmex_brace_not_misread_as_sum(self) -> None:
        # A bare CMEX brace piece outside a recognised underbrace must still not
        # be left as a stray \sum; inside recognize_latex the underbrace path
        # consumes it, so no \sum leaks for an underbrace-only expression.
        glyphs = eq4_underbrace_glyphs()
        latex = recognize_latex(glyphs, [])
        self.assertNotIn(r"\sum", latex)


@unittest.skipUnless(SAMPLE_PDF.exists(), "sample arXiv PDF not present")
class RealPdfRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pdfmdlite.pymupdf_text import extract_layout

        cls.layout = extract_layout(SAMPLE_PDF)
        cls.results = reconstruct_math(SAMPLE_PDF, cls.layout.pages)

    def _display_latex(self, page: int) -> dict[str, str]:
        result = self.results.get(page)
        self.assertIsNotNone(result, f"no math reconstructed on page {page}")
        return {region.number: _normalise(region.latex) for region in result.display}

    def test_page3_display_oracle(self) -> None:
        latex = self._display_latex(3)
        self.assertEqual(latex["1"], r"\mathbf{c} = \mathrm{ReLU}(M\mathbf{x} + \mathbf{b})")
        self.assertEqual(latex["2"], r"\hat{\mathbf{x}} = M^{T}\mathbf{c}")
        self.assertEqual(latex["3"], r"= \sum_{i=0}^{d_{\mathrm{hid}}-1} c_{i}\mathbf{f}_{i}")

    def test_page3_loss_equation_is_underbrace(self) -> None:
        latex = self._display_latex(3)
        self.assertIn(r"\underbrace{", latex["4"])
        self.assertIn(r"\mathcal{L}", latex["4"])
        self.assertNotIn(r"\sum", latex["4"])

    def test_page15_equations_present(self) -> None:
        latex = self._display_latex(15)
        # Minor equivalent variants are acceptable; the load-bearing structure
        # (the bold c / ReLU / M^T c forms) must be present.
        self.assertIn("5", latex)
        self.assertIn("6", latex)
        self.assertIn(r"\mathbf{c}", latex["5"])
        self.assertIn("ReLU", latex["5"])
        self.assertIn(r"\hat{\mathbf{x}}", latex["6"])
        self.assertIn("M", latex["6"])

    def test_every_display_region_conserves(self) -> None:
        import fitz
        from pdfmdlite.mathreco import (
            _body_size,
            _detect_display_regions,
            _raw_lines,
            rules_from_page,
        )

        with fitz.open(SAMPLE_PDF) as doc:
            for page in self.layout.pages:
                pdf_page = doc[page.number - 1]
                lines = _raw_lines(pdf_page)
                if not lines:
                    continue
                rules = rules_from_page(pdf_page)
                body = _body_size(lines)
                width, height = pdf_page.rect.width, pdf_page.rect.height
                for _bbox, latex, number, glyphs in _detect_display_regions(
                    page.number, lines, rules, width, height, body
                ):
                    self.assertTrue(
                        check_region_conservation(glyphs, latex),
                        f"page {page.number} eq ({number}) did not conserve: {latex}",
                    )

    def test_no_prose_or_caption_becomes_a_display_region(self) -> None:
        # Every display region's LaTeX is short and math-dense; a body sentence
        # or a "Figure N" / "Table N" caption must never be reconstructed as a
        # display equation.
        for result in self.results.values():
            for region in result.display:
                self.assertNotRegex(region.latex, r"(?i)\b(figure|table)\b")
                # A prose paragraph would carry many \mathrm words; a real
                # equation does not run on for hundreds of characters.
                self.assertLess(len(region.latex), 260)

    def test_inline_guard_rejects_bare_text_superscript(self) -> None:
        # The first page author block carries text superscripts (affiliation
        # markers); none must be turned into inline math.
        page1 = self.results.get(1)
        if page1 is None:
            return
        for region in page1.inline:
            self.assertTrue(
                any(ch.isalpha() or ch == "\\" for ch in region.latex),
                f"degenerate inline math: {region.latex!r}",
            )


@unittest.skipUnless(SAMPLE_PDF.exists(), "sample arXiv PDF not present")
@unittest.skipUnless(
    shutil.which("latexmk") and shutil.which("pdflatex"),
    "latexmk/pdflatex not installed",
)
class LatexCompileTests(unittest.TestCase):
    def test_all_sample_latex_compiles(self) -> None:
        from pdfmdlite.pymupdf_text import extract_layout

        layout = extract_layout(SAMPLE_PDF)
        results = reconstruct_math(SAMPLE_PDF, layout.pages)
        display: list[str] = []
        inline: list[str] = []
        for result in results.values():
            display.extend(region.latex for region in result.display)
            inline.extend(region.latex for region in result.inline)

        body = ["\\documentclass{article}", "\\usepackage{amsmath,amssymb,bm}", "\\begin{document}"]
        for latex in display:
            body.append("\\[" + latex + "\\]")
        for latex in inline:
            body.append("$" + latex + "$\\par")
        body.append("\\end{document}")
        document = "\n".join(body)

        with tempfile.TemporaryDirectory() as tmp:
            tex = Path(tmp) / "check.tex"
            tex.write_text(document, encoding="utf-8")
            proc = subprocess.run(
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    tex.name,
                ],
                cwd=tmp,
                capture_output=True,
                timeout=180,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=proc.stdout.decode("utf-8", "replace")[-3000:],
            )
            self.assertTrue((Path(tmp) / "check.pdf").exists())


if __name__ == "__main__":
    unittest.main()

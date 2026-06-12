"""Unit tests for the math structure recogniser's geometry and tables.

Every case is built from hand-made :class:`MathGlyph` lists and
:class:`Rule` rectangles (the module contract: the recogniser is fully
exercisable without a PDF), so these tests pin the *geometry* logic --
fractions, radicals, vertical bands, CMEX slot resolution, inline splicing --
independent of any input document. The characterization test pins the
end-to-end behaviour on a real paper.
"""
from __future__ import annotations

import unittest

from pdfmdlite.layout import Line, Word
from pdfmdlite.markdown import _splice_inline_math
from pdfmdlite.mathreco import (
    BBox,
    MathGlyph,
    MathRegion,
    Rule,
    _group_unnumbered_lines,
    _is_largeop,
    _largeop_command,
    _math_alphanumeric,
    _RawLine,
    check_symbol_conservation,
    classify,
    is_math_font,
    recognize_latex,
)


def glyph(
    char: str,
    font: str,
    ox: float,
    oy: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    size: float = 10.0,
) -> MathGlyph:
    return MathGlyph(
        char=char, font=font, size=size, flags=0,
        ox=ox, oy=oy, x0=x0, y0=y0, x1=x1, y1=y1,
    )


class SymbolTableTests(unittest.TestCase):
    def test_greek_letters(self) -> None:
        self.assertEqual(
            classify(glyph("β", "CMMI10", 0, 0, 0, 0, 5, 10)),
            (r"\beta", "italic_bare"),
        )
        self.assertEqual(
            classify(glyph("Ω", "CMR10", 0, 0, 0, 0, 5, 10))[0], r"\Omega"
        )

    def test_relations_and_operators(self) -> None:
        for char, token in [
            ("≤", r"\leq"), ("≥", r"\geq"), ("≠", r"\neq"),
            ("∂", r"\partial"), ("∇", r"\nabla"), ("∞", r"\infty"),
            ("→", r"\rightarrow"), ("⊗", r"\otimes"),
        ]:
            self.assertEqual(
                classify(glyph(char, "CMSY10", 0, 0, 0, 0, 5, 10))[0], token
            )

    def test_letterlike_blackboard(self) -> None:
        self.assertEqual(
            classify(glyph("ℝ", "XITSMath-Regular", 0, 0, 0, 0, 5, 10))[0],
            r"\mathbb{R}",
        )

    def test_math_alphanumeric_runs(self) -> None:
        self.assertEqual(_math_alphanumeric(0x1D400), ("A", "mathbf"))
        self.assertEqual(_math_alphanumeric(0x1D434), ("A", "italic_bare"))
        self.assertEqual(_math_alphanumeric(0x1D49C), (r"\mathcal{A}", "symbol"))
        self.assertEqual(_math_alphanumeric(0x1D538 + 17), ("R", "mathbb"))
        self.assertEqual(
            _math_alphanumeric(0x1D6E2 + 26), (r"\alpha", "italic_bare")
        )
        self.assertEqual(_math_alphanumeric(0x1D7CE), ("0", "upright"))
        self.assertIsNone(_math_alphanumeric(0x0041))

    def test_math_font_families(self) -> None:
        for font in [
            "ABCDEF+LMMathItalic10-Regular",
            "XITSMath-Regular",
            "STIXTwoMath-Regular",
            "TeXGyrePagellaMath-Regular",
            "CMMI10",
        ]:
            self.assertTrue(is_math_font(font), font)
        for font in ["LMRoman10-Regular", "NimbusRomNo9L-Regu", "Helvetica"]:
            self.assertFalse(is_math_font(font), font)


class CmexSlotTests(unittest.TestCase):
    def test_big_delimiters_resolve_by_slot(self) -> None:
        # Slot bytes proven on the sample papers: 0x00/0x01 big parens,
        # 'h'/'i' Big brackets, 'n'/'o' Big braces.
        cases = [
            ("\x00", "("), ("\x01", ")"),
            ("h", "["), ("i", "]"),
            ("n", r"\{"), ("o", r"\}"),
        ]
        for char, token in cases:
            g = glyph(char, "CMEX10", 0, 0, 0, 0, 5, 20)
            self.assertEqual(classify(g)[0], token)
            self.assertFalse(_is_largeop(g))

    def test_large_operator_resolves_by_slot(self) -> None:
        for char, command in [("P", r"\sum"), ("X", r"\sum")]:
            g = glyph(char, "CMEX10", 0, 0, 0, 0, 10, 20)
            self.assertTrue(_is_largeop(g))
            self.assertEqual(_largeop_command(g), command)

    def test_radical_slot_is_sqrt_marker(self) -> None:
        g = glyph("\x70", "CMEX10", 0, 0, 0, 0, 8, 30)
        self.assertEqual(classify(g)[0], r"\sqrt")
        self.assertFalse(_is_largeop(g))


class FractionTests(unittest.TestCase):
    def test_simple_fraction_with_left_context(self) -> None:
        glyphs = [
            glyph("x", "CMMI10", 10, 50, 10, 42, 16, 52),
            glyph("=", "CMR10", 20, 50, 20, 44, 26, 48),
            glyph("a", "CMMI10", 35, 44, 35, 36, 40, 45),
            glyph("+", "CMR10", 43, 44, 43, 37, 49, 44),
            glyph("b", "CMMI10", 52, 44, 52, 36, 57, 45),
            glyph("2", "CMR10", 43, 58, 43, 50, 49, 59),
        ]
        bar = Rule(30, 47.0, 60, 47.5, horizontal=True)
        latex = recognize_latex(glyphs, [bar])
        self.assertEqual(latex, r"x = \frac{a + b}{2}")
        self.assertTrue(check_symbol_conservation(glyphs, latex))

    def test_nested_fraction(self) -> None:
        glyphs = [
            glyph("a", "CMMI10", 30, 30, 30, 22, 35, 31, size=7.0),
            glyph("b", "CMMI10", 30, 42, 30, 34, 35, 43, size=7.0),
            glyph("c", "CMMI10", 30, 60, 30, 52, 35, 62),
        ]
        outer = Rule(25, 47.0, 42, 47.5, horizontal=True)
        inner = Rule(28, 32.5, 38, 33.0, horizontal=True)
        self.assertEqual(
            recognize_latex(glyphs, [outer, inner]), r"\frac{\frac{a}{b}}{c}"
        )


class RadicalTests(unittest.TestCase):
    def test_sqrt_with_vinculum(self) -> None:
        glyphs = [
            glyph("√", "CMSY10", 10, 48, 10, 30, 18, 50),
            glyph("x", "CMMI10", 20, 48, 20, 40, 26, 49),
            glyph("+", "CMR10", 28, 48, 28, 41, 33, 46),
            glyph("1", "CMR10", 35, 48, 35, 40, 39, 49),
        ]
        vinculum = Rule(17, 30.5, 40, 31.0, horizontal=True)
        self.assertEqual(recognize_latex(glyphs, [vinculum]), r"\sqrt{x + 1}")

    def test_bare_radical_falls_back_to_surd(self) -> None:
        glyphs = [glyph("√", "CMSY10", 10, 48, 10, 30, 18, 50)]
        self.assertEqual(recognize_latex(glyphs, []), r"\surd")


class VerticalBandTests(unittest.TestCase):
    def test_stacked_rows_join_with_linebreak(self) -> None:
        glyphs = [
            glyph("a", "CMMI10", 10, 50, 10, 42, 15, 52),
            glyph("=", "CMR10", 20, 50, 20, 44, 26, 48),
            glyph("b", "CMMI10", 30, 50, 30, 42, 35, 52),
            glyph("c", "CMMI10", 10, 70, 10, 62, 15, 72),
            glyph("=", "CMR10", 20, 70, 20, 64, 26, 68),
            glyph("d", "CMMI10", 30, 70, 30, 62, 35, 72),
        ]
        self.assertEqual(recognize_latex(glyphs, []), r"a = b \\ c = d")


class InlineSpliceTests(unittest.TestCase):
    def test_partial_word_coverage_splices_in_place(self) -> None:
        # The word "ℓ4-norm" is only partially covered by the math region
        # (the "ℓ4"); the LaTeX must replace that substring in place, never
        # duplicate the source text nor get appended at the end.
        words = [
            Word(text="the", x_min=0, y_min=0, x_max=20, y_max=10),
            Word(text="ℓ4-norm", x_min=25, y_min=0, x_max=70, y_max=10),
            Word(text="metric", x_min=75, y_min=0, x_max=110, y_max=10),
        ]
        line = Line(words=words, page_number=1)
        region = MathRegion(
            page=1,
            bbox=BBox(25, 0, 36, 10),  # only the "ℓ4" of "ℓ4-norm"
            latex=r"\ell^{4}",
            kind="inline",
            source_text="ℓ4",
        )
        self.assertEqual(
            _splice_inline_math(line, [region]),
            "the $\\ell^{4}$-norm metric",
        )


class UnnumberedGroupingTests(unittest.TestCase):
    def _eq_line(self, chars: str, y0: float, with_relation: bool) -> _RawLine:
        glyphs = []
        x = 100.0
        for i, char in enumerate(chars):
            font = "CMR10" if char in "=()0123456789" else "CMMI10"
            glyphs.append(
                glyph(char, font, x, y0 + 8, x, y0, x + 5, y0 + 10)
            )
            x += 6.0
        if with_relation:
            glyphs.append(
                glyph("=", "CMR10", x, y0 + 8, x, y0, x + 5, y0 + 10)
            )
        return _RawLine(tuple(glyphs), 100.0, y0, x + 5, y0 + 10)

    def test_short_fragment_without_relation_is_rejected(self) -> None:
        fragment = self._eq_line("abc", 100.0, with_relation=False)
        groups = _group_unnumbered_lines([fragment], 10.0, [fragment], [])
        self.assertEqual(groups, [])

    def test_equation_with_relation_is_accepted(self) -> None:
        equation = self._eq_line("abcdefgh", 100.0, with_relation=True)
        groups = _group_unnumbered_lines([equation], 10.0, [equation], [])
        self.assertEqual(len(groups), 1)

    def test_intervening_prose_breaks_the_group(self) -> None:
        first = self._eq_line("abcdefgh", 100.0, with_relation=True)
        second = self._eq_line("ijklmnop", 120.0, with_relation=True)
        prose = _RawLine(
            tuple(
                glyph(c, "NimbusRomNo9L-Regu", 50 + i * 6, 118, 50 + i * 6, 112, 55 + i * 6, 119)
                for i, c in enumerate("between")
            ),
            50.0, 112.0, 95.0, 119.0,
        )
        groups = _group_unnumbered_lines(
            [first, second], 10.0, [first, prose, second], []
        )
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()

"""Additional tests for ggplot2_py.guide."""

import pytest

from ggplot2_py.guide import (
    Guide, GuideNone, GuideLegend, GuideColourbar, GuideBins,
    GuideColoursteps, GuideAxis, GuideAxisLogticks, Guides,
    guide_none, guide_legend, guide_colourbar, guide_colorbar,
    guide_bins, guide_coloursteps, guide_colorsteps, guide_axis,
    guide_axis_logticks, guides, is_guide,
)


class TestGuideClasses:
    def test_none(self):
        assert is_guide(GuideNone())

    def test_legend(self):
        g = guide_legend(title="Test")
        assert is_guide(g)

    def test_colourbar(self):
        assert is_guide(GuideColourbar())

    def test_bins(self):
        assert is_guide(GuideBins())

    def test_coloursteps(self):
        assert is_guide(GuideColoursteps())

    def test_axis(self):
        assert is_guide(GuideAxis())

    def test_axis_logticks(self):
        assert is_guide(GuideAxisLogticks())


class TestGuides:
    def test_construction(self):
        assert Guides() is not None

    def test_guides_function(self):
        g = guides(colour="legend")
        assert g is not None

    def test_guides_with_kwargs(self):
        assert guides(colour="legend", fill="none") is not None


class TestConstructors:
    def test_none(self):
        assert is_guide(guide_none())

    def test_legend(self):
        assert is_guide(guide_legend())

    def test_colourbar(self):
        assert is_guide(guide_colourbar())

    def test_colorbar(self):
        assert is_guide(guide_colorbar())

    def test_bins(self):
        assert is_guide(guide_bins())

    def test_coloursteps(self):
        assert is_guide(guide_coloursteps())

    def test_colorsteps(self):
        assert is_guide(guide_colorsteps())

    def test_axis(self):
        assert is_guide(guide_axis())

    def test_axis_logticks(self):
        assert is_guide(guide_axis_logticks())

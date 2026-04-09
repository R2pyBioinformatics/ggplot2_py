"""Targeted coverage tests for ggplot2_py.plot – missing lines."""

import pytest
import numpy as np
import pandas as pd
import warnings

from ggplot2_py.plot import GGPlot, ggplot, ggplot_build
from ggplot2_py.aes import aes, Mapping


# ===========================================================================
# GGPlot: __radd__, __iadd__ (lines 202, 232-234)
# ===========================================================================

class TestGGPlotOperators:
    def test_radd_with_none(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        result = p.__radd__(None)
        assert isinstance(result, GGPlot)

    def test_radd_with_zero(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        result = p.__radd__(0)
        assert isinstance(result, GGPlot)

    def test_iadd(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        # iadd returns new plot
        from ggplot2_py.labels import labs
        p2 = p.__iadd__(labs(title="test"))
        assert isinstance(p2, GGPlot)


# ===========================================================================
# GGPlot: __getattr__, __setattr__ (lines 232-234, 273-275)
# ===========================================================================

class TestGGPlotAttrAccess:
    def test_getattr_meta(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        # Set a custom attribute
        p.custom_val = 42
        assert p.custom_val == 42

    def test_getattr_missing(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        with pytest.raises(AttributeError):
            _ = p.nonexistent_attr_xyz

    def test_getattr_private(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        with pytest.raises(AttributeError):
            _ = p._private_nonexistent


# ===========================================================================
# GGPlot: summary (lines 293)
# ===========================================================================

class TestGGPlotSummary:
    def test_summary_basic(self):
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        result = p.summary()
        assert isinstance(result, str)

    def test_summary_empty(self):
        p = ggplot()
        result = p.summary()
        assert isinstance(result, str)


# ===========================================================================
# ggplot() constructor: swapped data/mapping (lines 362-365)
# ===========================================================================

class TestGGPlotConstructor:
    def test_swapped_data_mapping(self):
        """When mapping is a DataFrame, swap data and mapping."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p = ggplot(mapping=df, data=aes(x="x"))
        assert isinstance(p, GGPlot)

    def test_none_mapping(self):
        p = ggplot(pd.DataFrame({"x": [1]}))
        assert isinstance(p, GGPlot)


# ===========================================================================
# ggplot_build: various branches (lines 464-465, 475-484, 565-578, 581)
# ===========================================================================

class TestGGPlotBuild:
    def test_build_basic(self):
        """Build a minimal plot."""
        p = ggplot(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}), aes(x="x", y="y"))
        result = ggplot_build(p)
        assert result is not None

    def test_build_with_layer(self):
        from ggplot2_py.geom import geom_point
        p = ggplot(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}), aes(x="x", y="y")) + geom_point()
        result = ggplot_build(p)
        assert result is not None

    def test_build_empty_data(self):
        p = ggplot()
        result = ggplot_build(p)

    def test_build_with_colour(self):
        """Build with colour aesthetic to trigger non-position scales (lines 564-578)."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "g": ["a", "b", "a"],
        })
        p = ggplot(df, aes(x="x", y="y", colour="g")) + geom_point()
        result = ggplot_build(p)
        assert result is not None

    def test_build_with_fill(self):
        """Build with fill to trigger more non-position branches."""
        from ggplot2_py.geom import geom_bar
        df = pd.DataFrame({
            "x": ["a", "b", "a", "b"],
            "fill": ["red", "blue", "red", "blue"],
        })
        p = ggplot(df, aes(x="x", fill="fill")) + geom_bar()
        result = ggplot_build(p)
        assert result is not None

    def test_build_add_coord(self):
        """Add coord to plot (line 837)."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.coord import coord_flip
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + geom_point()
        p = p + coord_flip()
        result = ggplot_build(p)

    def test_build_add_facet(self):
        """Add facet to plot."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.facet import facet_wrap
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "g": ["a", "a", "b", "b"]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point() + facet_wrap("~g")
        result = ggplot_build(p)

    def test_build_add_theme(self):
        """Add theme to plot (line 856-860)."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.theme import theme
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + geom_point()
        p = p + theme()
        result = ggplot_build(p)

    def test_build_add_labels(self):
        """Add labels to plot."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.labels import labs
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + geom_point() + labs(title="Test", x="X Axis", y="Y Axis")
        result = ggplot_build(p)


# ===========================================================================
# GGPlot: _repr_png_ (lines 273-275)
# ===========================================================================

class TestGGPlotReprPng:
    def test_repr_png(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        result = p._repr_png_()
        # May return None or bytes


# ===========================================================================
# print_plot / ggplot_gtable branches (lines 622, 624, 630-631, 682, 689-692)
# ===========================================================================

class TestPrintPlot:
    def test_repr(self):
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        result = repr(p)
        assert "GGPlot" in result or "ggplot" in result.lower()


# ===========================================================================
# Additional ggplot_build tests for deeper pipeline coverage
# ===========================================================================

class TestGGPlotBuildDeep:
    def test_build_with_stat_bin(self):
        """Exercise stat computation (lines 524-526)."""
        from ggplot2_py.geom import geom_histogram
        df = pd.DataFrame({"x": np.random.randn(20)})
        p = ggplot(df, aes(x="x")) + geom_histogram()
        result = ggplot_build(p)

    def test_build_with_stat_density(self):
        from ggplot2_py.geom import geom_density
        df = pd.DataFrame({"x": np.random.randn(20)})
        p = ggplot(df, aes(x="x")) + geom_density()
        result = ggplot_build(p)

    def test_build_with_geom_line(self):
        from ggplot2_py.geom import geom_line
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 2]})
        p = ggplot(df, aes(x="x", y="y")) + geom_line()
        result = ggplot_build(p)

    def test_build_with_multiple_layers(self):
        from ggplot2_py.geom import geom_point, geom_line
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 2]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point() + geom_line()
        result = ggplot_build(p)

    def test_build_with_geom_bar(self):
        from ggplot2_py.geom import geom_bar
        df = pd.DataFrame({"x": ["a", "b", "a", "c", "b", "a"]})
        p = ggplot(df, aes(x="x")) + geom_bar()
        result = ggplot_build(p)

    def test_build_with_geom_boxplot(self):
        from ggplot2_py.geom import geom_boxplot
        df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})
        p = ggplot(df, aes(x="x", y="y")) + geom_boxplot()
        result = ggplot_build(p)

    def test_ggplot_add_dataframe(self):
        """Adding a DataFrame replaces data (line 864)."""
        from ggplot2_py.geom import geom_point
        p = ggplot(pd.DataFrame({"x": [1], "y": [1]}), aes(x="x", y="y")) + geom_point()
        new_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p2 = p + new_data
        assert len(p2.data) == 3

    def test_ggplot_add_coord_override(self):
        """Adding coord when one already exists (line 837)."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.coord import coord_cartesian, coord_flip
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + geom_point() + coord_cartesian()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p2 = p + coord_flip()
        assert p2 is not None

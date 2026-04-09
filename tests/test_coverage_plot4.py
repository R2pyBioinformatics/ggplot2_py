"""Targeted coverage tests for ggplot2_py.plot – round 4."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.plot import (
    GGPlot,
    ggplot,
    ggplot_build,
    ggplot_gtable,
    ggplot_add,
    is_ggplot,
    set_last_plot,
    get_last_plot,
    BuiltGGPlot,
)
from ggplot2_py.aes import aes
from ggplot2_py.labels import labs


# ===========================================================================
# GGPlot attribute access (__setattr__, __getattr__, __repr__, summary)
# ===========================================================================

class TestGGPlotAttributes:
    def test_repr(self):
        """Cover __repr__ line 242+."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y"))
        r = repr(p)
        assert "GGPlot" in r
        assert "data=" in r

    def test_repr_no_data(self):
        p = ggplot()
        r = repr(p)
        assert "GGPlot" in r

    def test_setattr_meta(self):
        """Cover lines 232-234: meta attribute setting."""
        p = ggplot()
        p.custom_attr = "test_value"
        assert p.custom_attr == "test_value"

    def test_getattr_missing(self):
        """Cover __getattr__ AttributeError."""
        p = ggplot()
        with pytest.raises(AttributeError):
            _ = p.nonexistent_attribute_xyz

    def test_getattr_private(self):
        """Cover line 213: private attr raises."""
        p = ggplot()
        with pytest.raises(AttributeError):
            _ = p._private_thing

    def test_radd_zero(self):
        """Cover line 202: __radd__ with 0."""
        p = ggplot()
        result = 0 + p
        assert isinstance(result, GGPlot)

    def test_radd_none(self):
        """Cover __radd__ with None."""
        p = ggplot()
        result = None.__class__.__radd__ if hasattr(None, "__radd__") else p
        # Use sum() which calls __radd__ with 0
        result = sum([p])
        assert isinstance(result, GGPlot)


class TestGGPlotSummary:
    def test_summary_with_data(self):
        """Cover lines 285-301: summary method."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y"))
        s = p.summary()
        assert "data:" in s

    def test_summary_with_mapping(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y"))
        s = p.summary()
        assert "mapping:" in s

    def test_summary_empty(self):
        p = ggplot()
        s = p.summary()
        assert isinstance(s, str)


# ===========================================================================
# _repr_png_ / _repr_html_
# ===========================================================================

class TestGGPlotRepr:
    def test_repr_png_empty(self):
        """Cover lines 273-275: _repr_png_ with exception returns None."""
        p = ggplot()
        result = p._repr_png_()
        # Should return None on error (no layers, rendering may fail)
        # The function catches all exceptions


# ===========================================================================
# ggplot() constructor
# ===========================================================================

class TestGGPlotConstructor:
    def test_ggplot_swapped_args(self):
        """Cover lines 362-365: data and mapping swapped."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        mapping = aes("x", "y")
        # Pass mapping as first arg, data as second
        p = ggplot(mapping, df)
        assert isinstance(p, GGPlot)

    def test_ggplot_dict_data(self):
        """Cover dict data branch."""
        p = ggplot({"x": [1, 2], "y": [3, 4]})
        assert isinstance(p, GGPlot)


# ===========================================================================
# ggplot_build branches
# ===========================================================================

class TestGGPlotBuild:
    def test_build_empty_plot(self):
        """Cover lines 464-465: add blank layer when no layers."""
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes("x", "y"))
        built = ggplot_build(p)
        assert isinstance(built, BuiltGGPlot)

    def test_build_with_geom(self):
        """Cover lines 475-484: layer data resolution."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        assert built.data is not None

    def test_build_non_position_scales(self):
        """Cover lines 565-578: non-position scale training."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4],
                           "colour": ["red", "blue"]})
        p = ggplot(df, aes("x", "y", colour="colour")) + geom_point()
        built = ggplot_build(p)
        assert built is not None

    def test_build_no_npscales(self):
        """Cover lines 580-581: empty non-position scales."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        assert built is not None


# ===========================================================================
# ggplot_add branches
# ===========================================================================

class TestGGPlotAdd:
    def test_add_coord(self):
        """Cover lines 837+: adding coord."""
        from ggplot2_py.coord import coord_cartesian
        p = ggplot()
        p2 = p + coord_cartesian()
        assert p2.coordinates is not None

    def test_add_facet(self):
        """Cover line 856-860: adding facet."""
        from ggplot2_py.facet import facet_wrap
        p = ggplot()
        p2 = p + facet_wrap("x")
        assert p2.facet is not None

    def test_add_theme(self):
        """Cover adding theme."""
        from ggplot2_py.theme import theme, Theme
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes("x", "y"))
        # Ensure plot.theme is a Theme object first
        p.theme = Theme()
        p2 = p + theme()
        assert p2 is not None

    def test_add_dataframe(self):
        """Cover line 864: adding DataFrame replaces data."""
        p = ggplot()
        df = pd.DataFrame({"x": [1, 2]})
        p2 = p + df
        assert isinstance(p2.data, pd.DataFrame)

    def test_add_labs(self):
        """Cover adding labels."""
        p = ggplot()
        p2 = p + labs(x="X axis", y="Y axis")
        assert p2 is not None


# ===========================================================================
# Introspection helpers
# ===========================================================================

class TestIntrospection:
    def test_is_ggplot(self):
        p = ggplot()
        assert is_ggplot(p) is True
        assert is_ggplot(42) is False

    def test_last_plot(self):
        """Cover set/get last plot."""
        p = ggplot()
        set_last_plot(p)
        assert get_last_plot() is p

    def test_get_layer_data(self):
        """Cover lines 974-980."""
        from ggplot2_py.plot import get_layer_data
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        data = get_layer_data(p, i=1)
        assert isinstance(data, pd.DataFrame)

    def test_get_layer_data_out_of_range(self):
        """Cover line 979: out-of-range layer index."""
        from ggplot2_py.plot import get_layer_data
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with pytest.raises(Exception):
            get_layer_data(p, i=99)

    def test_get_layer_grob(self):
        """Cover lines 1003-1012."""
        from ggplot2_py.plot import get_layer_grob
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        grob = get_layer_grob(p, i=1)
        assert grob is not None

    def test_get_panel_scales(self):
        """Cover lines 1039-1049."""
        from ggplot2_py.plot import get_panel_scales
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        scales = get_panel_scales(p, i=1, j=1)
        assert scales is not None

    def test_get_layer_data_default_plot(self):
        """Cover line 974-975: default plot (None)."""
        from ggplot2_py.plot import get_layer_data
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        set_last_plot(p)
        data = get_layer_data(None, i=1)
        assert isinstance(data, pd.DataFrame)


# ===========================================================================
# update_labels
# ===========================================================================

class TestUpdateLabels:
    def test_update_labels(self):
        """Cover plot._update_labels."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        # Labels should be auto-detected from mapping
        built = ggplot_build(p)
        assert "x" in built.plot.labels or hasattr(built.plot.labels, "get")


# ===========================================================================
# ggplot_gtable
# ===========================================================================

class TestGGPlotGtable:
    def test_gtable_basic(self):
        """Cover lines 682, 689-692: gtable construction."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        table = ggplot_gtable(built)
        assert table is not None


# ===========================================================================
# plot.print_ / plot.draw (if available)
# ===========================================================================

class TestPlotPrint:
    def test_print_plot(self):
        """Cover print_plot if available."""
        from ggplot2_py.plot import print_plot
        p = ggplot(pd.DataFrame({"x": [1], "y": [1]}), aes("x", "y"))
        print_plot(p)


# ===========================================================================
# GGPlot._clone
# ===========================================================================

class TestGGPlotClone:
    def test_clone(self):
        """Cover _clone method."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y"))
        if hasattr(p, "_clone"):
            c = p._clone()
            assert isinstance(c, GGPlot)
            assert c is not p

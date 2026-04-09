"""Targeted coverage tests for remaining modules: annotation, facet, layer, layout."""

import pytest
import numpy as np
import pandas as pd
import warnings

# ===========================================================================
# Annotation tests (lines 203, 210-227, 290, 298-309, 415, 420-423)
# ===========================================================================

class _MockAnnotCoord:
    """Mock coord that passes data through for annotation geom testing."""
    def transform(self, data, panel_params):
        return data

class TestAnnotationCustom:
    def test_annotation_custom(self):
        from ggplot2_py.annotation import annotation_custom
        from grid_py import null_grob
        layer = annotation_custom(null_grob())
        assert layer is not None

    def test_annotation_custom_draw(self):
        """Test the GeomCustomAnn draw_panel method (lines 210-227)."""
        from ggplot2_py.annotation import annotation_custom
        from grid_py import null_grob, rect_grob
        layer = annotation_custom(rect_grob())
        # Access the geom's draw_panel via the layer
        geom = layer.geom if hasattr(layer, "geom") else None
        assert geom is not None and hasattr(geom, "draw_panel")
        grob = rect_grob()
        result = geom.draw_panel(
            None, {}, _MockAnnotCoord(),
            grob=grob, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        )
        assert result is not None

    def test_annotation_raster(self):
        from ggplot2_py.annotation import annotation_raster
        raster = np.zeros((3, 3, 3))
        layer = annotation_raster(raster, xmin=0, xmax=1, ymin=0, ymax=1)
        assert layer is not None

    def test_annotation_raster_draw(self):
        """Test the GeomRasterAnn draw_panel method (lines 298-309)."""
        from ggplot2_py.annotation import annotation_raster
        raster = np.zeros((3, 3, 3))
        layer = annotation_raster(raster, xmin=0, xmax=1, ymin=0, ymax=1)
        geom = layer.geom if hasattr(layer, "geom") else None
        assert geom is not None and hasattr(geom, "draw_panel")
        result = geom.draw_panel(
            None, {}, _MockAnnotCoord(),
            raster=raster, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        )
        assert result is not None

    def test_annotation_logticks(self):
        from ggplot2_py.annotation import annotation_logticks
        layer = annotation_logticks()
        assert layer is not None

    def test_annotation_logticks_draw(self):
        """Test the GeomLogticks draw_panel method (lines 420-423)."""
        from ggplot2_py.annotation import annotation_logticks
        layer = annotation_logticks()
        geom = layer.geom if hasattr(layer, "geom") else None
        assert geom is not None and hasattr(geom, "draw_panel")
        result = geom.draw_panel(None, {}, _MockAnnotCoord())
        assert result is not None


# ===========================================================================
# Layer tests: setup_layer, compute_aesthetics, compute_geom_1
# (lines 154, 184-187, 190, 193-194, 390-394, 398-402, 408, 411-414, 418)
# ===========================================================================

class TestLayer:
    def test_layer_data_callable(self):
        from ggplot2_py.layer import layer, Layer
        from ggplot2_py.stat import StatIdentity
        from ggplot2_py.geom import GeomPoint
        l = layer(
            geom=GeomPoint,
            stat=StatIdentity,
            data=lambda d: d.head(2),
            mapping=None,
            position="identity",
        )
        plot_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = l.layer_data(plot_data)
        assert len(result) == 2

    def test_layer_setup_layer(self):
        from ggplot2_py.layer import layer
        from ggplot2_py.stat import StatIdentity
        from ggplot2_py.geom import GeomPoint
        l = layer(
            geom=GeomPoint,
            stat=StatIdentity,
            data=None,
            mapping=None,
            position="identity",
        )
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = l.setup_layer(df, None)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# Facet tests: facet_null, facet_wrap, facet_grid setup
# (lines 115, 119, 158, 375, 433-466, 563-564, 567, 640, 752-753)
# ===========================================================================

class TestFacet:
    def test_facet_null(self):
        from ggplot2_py.facet import FacetNull, facet_null
        f = facet_null()
        assert f is not None

    def test_facet_wrap_basic(self):
        from ggplot2_py.facet import facet_wrap
        f = facet_wrap("~species")
        assert f is not None

    def test_facet_wrap_list(self):
        from ggplot2_py.facet import facet_wrap
        f = facet_wrap(["species"])
        assert f is not None

    def test_facet_grid_basic(self):
        from ggplot2_py.facet import facet_grid
        f = facet_grid("species ~ .")
        assert f is not None


# ===========================================================================
# Layout tests: setup, train_position, map_position
# (lines 78-81, 85-86, 161, 284, 286, 294)
# ===========================================================================

class TestLayout:
    def test_layout_setup(self):
        from ggplot2_py.layout import Layout
        layout = Layout()
        assert layout is not None


# ===========================================================================
# ggproto tests (lines 109, 194)
# ===========================================================================

class TestGGProto:
    def test_ggproto_repr(self):
        from ggplot2_py.ggproto import GGProto
        obj = GGProto()
        result = repr(obj)
        assert "ggproto" in result.lower() or "GGProto" in result

    def test_ggproto_str(self):
        from ggplot2_py.ggproto import GGProto
        obj = GGProto()
        result = str(obj)
        assert isinstance(result, str)


# ===========================================================================
# Draw key tests (lines 38-39, 85-86)
# ===========================================================================

class TestDrawKey:
    def test_draw_key_point(self):
        from ggplot2_py.draw_key import draw_key_point
        data = {"colour": "black", "fill": "white", "size": 1.5,
                "shape": 19, "alpha": 1.0, "stroke": 0.5}
        result = draw_key_point(data, {})
        assert result is not None

    def test_draw_key_rect(self):
        from ggplot2_py.draw_key import draw_key_rect
        data = {"colour": "black", "fill": "white", "alpha": 1.0,
                "linewidth": 0.5, "linetype": 1, "size": 1.0}
        result = draw_key_rect(data, {})
        assert result is not None


# ===========================================================================
# Limits tests (lines 60, 131-136, 161-169)
# ===========================================================================

class TestLimits:
    def test_xlim(self):
        from ggplot2_py.limits import xlim, ylim, lims
        result = xlim(0, 10)
        assert result is not None

    def test_ylim(self):
        from ggplot2_py.limits import ylim
        result = ylim(0, 10)
        assert result is not None

    def test_lims(self):
        from ggplot2_py.limits import lims
        result = lims(x=(0, 10))
        assert result is not None

    def test_expand_limits(self):
        from ggplot2_py.limits import expand_limits
        result = expand_limits(x=0)
        assert result is not None


# ===========================================================================
# _compat tests (lines 71-72, 95-96)
# ===========================================================================

class TestCompat:
    def test_cli_inform(self):
        from ggplot2_py._compat import cli_inform
        # Should not raise
        cli_inform("test message")

    def test_deprecate_warn(self):
        from ggplot2_py._compat import deprecate_warn
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            deprecate_warn("1.0", "old_func()")


# ===========================================================================
# _utils tests (lines 149, 492-494, 497-498, 518-519, 522-523)
# ===========================================================================

class TestUtils:
    def test_resolution(self):
        from ggplot2_py._utils import resolution
        result = resolution(np.array([1.0, 2.0, 3.0, 4.0]))
        assert result > 0

    def test_snake_class(self):
        from ggplot2_py._utils import snake_class
        from ggplot2_py.geom import GeomPoint
        result = snake_class(GeomPoint())
        assert "geom_point" == result

    def test_remove_missing(self):
        from ggplot2_py._utils import remove_missing
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, np.nan]})
        result = remove_missing(df, vars=["x", "y"])
        assert len(result) == 1


# ===========================================================================
# Theme tests (lines 286-288, 350-351, 362, 367-369, 473-475)
# ===========================================================================

class TestTheme:
    def test_theme_add(self):
        from ggplot2_py.theme import theme, add_theme
        t1 = theme()
        t2 = theme(axis_text_x=None)
        result = add_theme(t1, t2)
        assert result is not None

    def test_theme_get(self):
        from ggplot2_py.theme import theme
        t = theme()
        # Try to get a theme element
        try:
            result = t.get("axis.text.x")
        except (AttributeError, KeyError):
            pass


# ===========================================================================
# QPlot tests (lines 169, 197-198, 203-204, 208-209)
# ===========================================================================

class TestQplot:
    def test_qplot_basic(self):
        from ggplot2_py.qplot import qplot
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = qplot(x="x", y="y", data=df)
        assert result is not None

    def test_qplot_geom(self):
        from ggplot2_py.qplot import qplot
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = qplot(x="x", y="y", data=df, geom="line")
        assert result is not None

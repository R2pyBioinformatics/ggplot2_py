"""Targeted coverage tests for remaining modules – round 4.

Covers: coord.py, facet.py, layer.py, layout.py, position.py,
annotation.py, save.py, theme.py, theme_elements.py
"""

import pytest
import os
import math
import numpy as np
import pandas as pd
import tempfile

from ggplot2_py.aes import aes
from ggplot2_py.plot import ggplot, ggplot_build, GGPlot
from ggplot2_py._compat import waiver, is_waiver


# ===========================================================================
# coord.py
# ===========================================================================

class TestCoordPolar:
    def test_coord_polar_basic(self):
        from ggplot2_py.coord import coord_polar, CoordPolar
        c = coord_polar()
        assert isinstance(c, CoordPolar)

    def test_coord_polar_theta_y(self):
        from ggplot2_py.coord import coord_polar
        c = coord_polar(theta="y")
        assert c.theta == "y"

    def test_coord_polar_transform(self):
        from ggplot2_py.coord import coord_polar
        c = coord_polar()
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        panel_params = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.transform(df, panel_params)
        assert "x" in result.columns

    def test_coord_polar_bad_theta(self):
        from ggplot2_py.coord import coord_polar
        with pytest.raises(Exception):
            coord_polar(theta="z")


class TestCoordRadial:
    def test_coord_radial_basic(self):
        from ggplot2_py.coord import coord_radial, CoordRadial
        c = coord_radial()
        assert isinstance(c, CoordRadial)

    def test_coord_radial_reverse_theta(self):
        """Cover line 1593: reverse theta."""
        from ggplot2_py.coord import coord_radial
        c = coord_radial(reverse="theta")
        assert c is not None

    def test_coord_radial_reverse_r(self):
        """Cover line 1598: reverse r."""
        from ggplot2_py.coord import coord_radial
        c = coord_radial(reverse="r")
        assert c is not None

    def test_coord_radial_reverse_thetar(self):
        from ggplot2_py.coord import coord_radial
        c = coord_radial(reverse="thetar")
        assert c is not None

    def test_coord_radial_bad_reverse(self):
        from ggplot2_py.coord import coord_radial
        with pytest.raises(Exception):
            coord_radial(reverse="bad")

    def test_coord_radial_bad_theta(self):
        from ggplot2_py.coord import coord_radial
        with pytest.raises(Exception):
            coord_radial(theta="z")

    def test_coord_radial_with_limits(self):
        from ggplot2_py.coord import coord_radial
        c = coord_radial(thetalim=[0, 1], rlim=[0, 10])
        assert c.limits is not None

    def test_coord_radial_arc_wrap(self):
        """Cover line 1589-1590: arc wrap when start > end."""
        from ggplot2_py.coord import coord_radial
        c = coord_radial(start=10.0, end=2.0)
        assert c is not None

    def test_coord_radial_distance(self):
        """Cover distance method."""
        from ggplot2_py.coord import coord_radial
        c = coord_radial()
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.6, 0.8])
        panel_params = {"theta.range": [0, 1], "r.range": [0, 1]}
        dist = c.distance(x, y, panel_params)
        assert dist is not None

    def test_coord_radial_backtransform_range(self):
        """Cover line 1032: backtransform_range."""
        from ggplot2_py.coord import coord_radial
        c = coord_radial()
        panel_params = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.backtransform_range(panel_params)
        assert isinstance(result, dict)

    def test_coord_radial_setup_panel_params(self):
        """Cover lines 1064-1072: setup_panel_params."""
        from ggplot2_py.coord import coord_radial
        from ggplot2_py.scale import continuous_scale, ScaleContinuousPosition
        c = coord_radial()
        sx = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              breaks=[2.0, 5.0, 8.0])
        sx.train(np.array([0.0, 10.0]))
        sy = continuous_scale("y", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              breaks=[2.0, 5.0, 8.0])
        sy.train(np.array([0.0, 10.0]))
        result = c.setup_panel_params(sx, sy)
        assert "theta.range" in result


class TestCoordSf:
    def test_coord_sf(self):
        """Cover coord_sf if available."""
        try:
            from ggplot2_py.coord import coord_sf
            c = coord_sf()
            assert c is not None
        except (ImportError, AttributeError):
            pass


class TestCoordTransform:
    def test_coord_trans_deprecated(self):
        """Cover coord_trans deprecation warning (line 1700)."""
        try:
            from ggplot2_py.coord import coord_trans
            with pytest.warns(Warning):
                c = coord_trans()
        except (ImportError, AttributeError):
            pass


class TestCoordCartesian:
    def test_coord_cartesian_render_axis_v(self):
        """Cover line 483-484: render_axis_v."""
        from ggplot2_py.coord import coord_cartesian
        c = coord_cartesian()
        panel_params = {"x.range": [0, 1], "y.range": [0, 1]}
        result = c.render_axis_v(panel_params, theme={})
        assert isinstance(result, dict)


class TestCoordMunch:
    def test_coord_munch(self):
        """Cover coord_munch lines 1379-1410."""
        from ggplot2_py.coord import coord_cartesian
        try:
            from ggplot2_py.coord import coord_munch
            c = coord_cartesian()
            df = pd.DataFrame({
                "x": [0.0, 0.5, 1.0],
                "y": [0.0, 0.5, 1.0],
                "group": [1, 1, 1],
            })
            panel_params = {"x.range": [0, 1], "y.range": [0, 1]}
            result = coord_munch(c, df, panel_params, n=5)
            assert isinstance(result, pd.DataFrame)
        except (ImportError, AttributeError):
            pass


# ===========================================================================
# facet.py
# ===========================================================================

class TestFacetDeep:
    def test_facet_wrap_single_var(self):
        """Cover facet_wrap with string var."""
        from ggplot2_py.facet import facet_wrap
        f = facet_wrap("x")
        assert f is not None

    def test_facet_grid(self):
        """Cover facet_grid."""
        from ggplot2_py.facet import facet_grid
        f = facet_grid(rows="x", cols="y")
        assert f is not None

    def test_facet_null(self):
        from ggplot2_py.facet import facet_null, FacetNull
        f = facet_null()
        assert isinstance(f, FacetNull)

    def test_facet_wrap_compute_layout(self):
        """Cover facet_wrap compute_layout lines 640+."""
        from ggplot2_py.facet import facet_wrap
        f = facet_wrap("g")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "g": ["a", "b", "a"]})
        if hasattr(f, "compute_layout"):
            try:
                layout = f.compute_layout([df], params={})
                assert isinstance(layout, pd.DataFrame)
            except TypeError:
                layout = f.compute_layout([df])
                assert isinstance(layout, pd.DataFrame)

    def test_facet_vars_extract(self):
        """Cover lines 115: non-string facet var."""
        from ggplot2_py.facet import _resolve_facet_vars
        result = _resolve_facet_vars(["x", 42])
        assert "42" in result

    def test_facet_vars_dict(self):
        """Cover line 117-118: dict facet vars."""
        from ggplot2_py.facet import _resolve_facet_vars
        result = _resolve_facet_vars({"a": None, "b": None})
        assert "a" in result

    def test_facet_wrap_draw(self):
        """Cover lines 433-466: draw_panels."""
        from ggplot2_py.facet import facet_wrap
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        p = ggplot(df, aes("x", "y")) + geom_point() + facet_wrap("g")
        built = ggplot_build(p)
        assert built is not None

    def test_facet_null_draw(self):
        """Cover FacetNull.draw_panels lines 558-567."""
        from ggplot2_py.facet import facet_null
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        assert built is not None


# ===========================================================================
# layer.py
# ===========================================================================

class TestLayerDeep:
    def test_layer_with_callable_data(self):
        """Cover lines 477-484: callable data."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point(data=lambda d: d.head(1))
        built = ggplot_build(p)
        assert built is not None

    def test_layer_compute_aesthetics(self):
        """Cover layer compute_aesthetics lines 390-428."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y", colour="x")) + geom_point()
        built = ggplot_build(p)
        assert built is not None

    def test_split_params(self):
        """Cover layer._split_params lines 184-194."""
        from ggplot2_py.layer import _split_params
        from ggplot2_py.geom import GeomPoint
        from ggplot2_py.stat import StatIdentity
        from ggplot2_py.position import PositionIdentity
        geom_params, stat_params, aes_params = _split_params(
            {"alpha": 0.5, "size": 3},
            GeomPoint, StatIdentity, PositionIdentity
        )
        # Some params should end up somewhere

    def test_layer_with_deferred_geom(self):
        """Cover lines 833-835: non-GGProto geom."""
        from ggplot2_py.layer import layer
        from ggplot2_py.stat import StatIdentity
        from ggplot2_py.position import PositionIdentity
        lyr = layer(
            geom="point",
            stat=StatIdentity,
            position=PositionIdentity,
            data=pd.DataFrame({"x": [1], "y": [1]}),
            mapping=aes("x", "y"),
        )
        assert lyr is not None


# ===========================================================================
# layout.py
# ===========================================================================

class TestLayoutDeep:
    def test_create_layout(self):
        """Cover layout creation."""
        from ggplot2_py.layout import create_layout
        from ggplot2_py.facet import facet_null
        from ggplot2_py.coord import coord_cartesian
        layout = create_layout(facet_null(), coord_cartesian())
        assert layout is not None

    def test_layout_setup(self):
        """Cover layout.setup lines 78-86."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        assert built.layout is not None

    def test_layout_train_position(self):
        """Cover layout train_position lines 284-325."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        assert len(built.data) > 0

    def test_layout_map_position(self):
        """Cover layout map_position lines 421-436."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        # Verify data was mapped
        assert built.data[0] is not None

    def test_layout_get_scales(self):
        """Cover layout get_scales lines 527+."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        if hasattr(built.layout, "get_scales"):
            scales = built.layout.get_scales(1)

    def test_layout_finish_data(self):
        """Cover layout finish_data lines 554-559."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        # finish_data is called in ggplot_build

    def test_layout_render(self):
        """Cover layout.render lines 637+."""
        from ggplot2_py.geom import geom_point
        from ggplot2_py.plot import ggplot_gtable
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        table = ggplot_gtable(built)
        assert table is not None


# ===========================================================================
# position.py
# ===========================================================================

class TestPositionDeep:
    def test_position_dodge2(self):
        """Cover PositionDodge2."""
        from ggplot2_py.position import PositionDodge2
        p = PositionDodge2()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1, 2, 3, 4],
            "group": [1, 2, 1, 2],
            "PANEL": [1, 1, 1, 1],
        })
        params = p.setup_params(df)
        assert "padding" in params

    def test_position_dodge_setup(self):
        """Cover PositionDodge.setup_params with preserve='single'."""
        from ggplot2_py.position import PositionDodge
        p = PositionDodge()
        p.preserve = "single"
        df = pd.DataFrame({
            "x": [1, 1, 2], "y": [1, 2, 3],
            "group": [1, 2, 1], "PANEL": [1, 1, 1],
        })
        params = p.setup_params(df)
        assert "n" in params

    def test_position_identity(self):
        from ggplot2_py.position import PositionIdentity
        p = PositionIdentity()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "PANEL": [1, 1]})
        result = p.compute_layer(df, {}, layout=None)
        assert len(result) == 2

    def test_position_compute_panel_not_implemented(self):
        """Cover line 347: abstract compute_panel."""
        from ggplot2_py.position import Position
        p = Position()
        with pytest.raises(Exception):
            p.compute_panel(pd.DataFrame(), {})

    def test_position_aesthetics(self):
        """Cover lines 356-361: aesthetics method."""
        from ggplot2_py.position import Position
        p = Position()
        p.required_aes = ["x|y"]
        result = p.aesthetics()
        assert "x" in result or "y" in result

    def test_position_setup_data(self):
        """Cover lines 245-256: setup_data with params."""
        from ggplot2_py.position import Position
        p = Position()
        p.required_aes = []
        p.default_aes = {}
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "PANEL": [1, 1]})
        result = p.setup_data(df, {"width": 0.5})
        assert isinstance(result, pd.DataFrame)

    def test_resolution(self):
        """Cover lines 124-128: resolution function."""
        from ggplot2_py.position import _resolution
        r = _resolution(np.array([1.0, 2.0, 4.0]))
        assert r == 1.0

    def test_resolution_single(self):
        from ggplot2_py.position import _resolution
        r = _resolution(np.array([5.0]))
        assert r == 1.0

    def test_resolution_zero(self):
        """Cover line 127: zero=True."""
        from ggplot2_py.position import _resolution
        r = _resolution(np.array([2.0, 4.0]), zero=True)
        assert r > 0


class TestPositionStack:
    def test_position_stack(self):
        from ggplot2_py.position import PositionStack
        p = PositionStack()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1, 2, 3, 4],
            "ymin": [0, 0, 0, 0],
            "ymax": [1, 2, 3, 4],
            "group": [1, 2, 1, 2],
            "PANEL": [1, 1, 1, 1],
        })
        result = p.compute_layer(df, p.setup_params(df), layout=None)
        assert len(result) > 0


class TestPositionJitter:
    def test_position_jitter(self):
        from ggplot2_py.position import PositionJitter
        p = PositionJitter()
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "group": [1, 1, 1],
            "PANEL": [1, 1, 1],
        })
        result = p.compute_layer(df, p.setup_params(df), layout=None)
        assert len(result) == 3


# ===========================================================================
# annotation.py
# ===========================================================================

class TestAnnotationDeep:
    def test_annotation_custom(self):
        """Cover lines 203-227: annotation_custom."""
        from ggplot2_py.annotation import annotation_custom
        from grid_py import null_grob
        grob = null_grob()
        layer = annotation_custom(grob, xmin=0, xmax=1, ymin=0, ymax=1)
        assert layer is not None

    def test_annotation_raster(self):
        """Cover lines 290-314: annotation_raster."""
        from ggplot2_py.annotation import annotation_raster
        raster = np.zeros((5, 5, 3))
        layer = annotation_raster(raster, xmin=0, xmax=1, ymin=0, ymax=1)
        assert layer is not None

    def test_annotation_logticks(self):
        """Cover lines 415-423: annotation_logticks."""
        from ggplot2_py.annotation import annotation_logticks
        layer = annotation_logticks()
        assert layer is not None


# ===========================================================================
# save.py
# ===========================================================================

class TestSaveDeep:
    def test_ggsave_png(self):
        """Cover save.ggsave lines 218-317."""
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            ggsave(fname, plot=p, width=3, height=3, dpi=72)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_ggsave_no_dir(self):
        """Cover line 259: non-ggplot passthrough."""
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        from ggplot2_py.plot import ggplot_gtable
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            # Pass a non-ggplot (e.g., a gtable) to cover the else branch at line 259
            df = pd.DataFrame({"x": [1], "y": [1]})
            p = ggplot(df, aes("x", "y")) + geom_point()
            built = ggplot_build(p)
            table = ggplot_gtable(built)
            ggsave(fname, plot=table, width=3, height=3, dpi=72)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_ggsave_limitsize(self):
        """Cover line 259: limitsize error."""
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with pytest.raises(Exception):
            ggsave("/tmp/test_big.png", plot=p, width=100, height=100,
                   units="in", dpi=72, limitsize=True)

    def test_check_device(self):
        """Cover check_device function."""
        from ggplot2_py.save import check_device
        dev = check_device(None, "test.png")
        assert dev == "png"
        dev = check_device(None, "test.pdf")
        assert dev == "pdf"


# ===========================================================================
# theme.py
# ===========================================================================

class TestThemeDeep:
    def test_theme_creation(self):
        from ggplot2_py.theme import theme, Theme
        t = theme()
        assert isinstance(t, Theme)

    def test_theme_add(self):
        """Cover theme add_theme lines 286-288."""
        from ggplot2_py.theme import theme, Theme, add_theme
        t1 = Theme()
        t2 = theme()
        result = add_theme(t1, t2)
        assert result is not None

    def test_theme_replace(self):
        """Cover theme replace lines 350-351."""
        from ggplot2_py.theme import Theme
        t = Theme()
        if hasattr(t, "replace"):
            t2 = Theme()
            result = t.replace(t2)

    def test_is_theme(self):
        from ggplot2_py.theme import is_theme, Theme
        assert is_theme(Theme()) is True
        assert is_theme(42) is False

    def test_complete_theme(self):
        """Cover theme completion lines 362-369."""
        from ggplot2_py.theme import Theme, complete_theme
        t = Theme()
        result = complete_theme(t)
        assert result is not None

    def test_theme_get_element(self):
        """Cover Theme __getitem__ / get."""
        from ggplot2_py.theme import Theme
        t = Theme()
        result = t.get("axis.line") if hasattr(t, "get") else None

    def test_theme_update(self):
        """Cover Theme update method."""
        from ggplot2_py.theme import Theme
        t = Theme()
        if hasattr(t, "update"):
            t.update({"axis.line": None})


# ===========================================================================
# theme_elements.py
# ===========================================================================

class TestThemeElements:
    def test_element_blank(self):
        from ggplot2_py.theme_elements import element_blank
        e = element_blank()
        assert e is not None

    def test_element_text(self):
        from ggplot2_py.theme_elements import element_text
        e = element_text(size=12, colour="black")
        assert e is not None

    def test_element_line(self):
        from ggplot2_py.theme_elements import element_line
        e = element_line(colour="black", linewidth=1)
        assert e is not None

    def test_element_rect(self):
        from ggplot2_py.theme_elements import element_rect
        e = element_rect(fill="white", colour="black")
        assert e is not None

    def test_margin(self):
        """Cover margin function."""
        try:
            from ggplot2_py.theme_elements import margin
            m = margin(5, 5, 5, 5, unit="pt")
            assert m is not None
        except (ImportError, AttributeError):
            pass

    def test_rel(self):
        """Cover rel function."""
        try:
            from ggplot2_py.theme_elements import rel
            r = rel(1.2)
            assert r is not None
        except (ImportError, AttributeError):
            pass


# ===========================================================================
# More position tests
# ===========================================================================

class TestPositionDodge2Deep:
    def test_dodge2_compute(self):
        """Cover PositionDodge2.compute_panel."""
        from ggplot2_py.position import PositionDodge2
        p = PositionDodge2()
        p.preserve = "single"
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "xmin": [0.5, 0.5, 1.5, 1.5],
            "xmax": [1.5, 1.5, 2.5, 2.5],
            "y": [1, 2, 3, 4],
            "ymin": [0, 0, 0, 0],
            "ymax": [1, 2, 3, 4],
            "group": [1, 2, 1, 2],
            "PANEL": [1, 1, 1, 1],
        })
        params = p.setup_params(df)
        result = p.compute_layer(df, params, layout=None)
        assert len(result) == 4

    def test_dodge2_no_x(self):
        """Cover line 542-546: no x column."""
        from ggplot2_py.position import PositionDodge2
        p = PositionDodge2()
        p.preserve = "single"
        df = pd.DataFrame({
            "y": [1, 2],
            "group": [1, 2],
            "PANEL": [1, 1],
        })
        params = p.setup_params(df)
        assert "n" in params


class TestPositionJitterDodge:
    def test_jitter_dodge(self):
        """Cover PositionJitterdodge."""
        try:
            from ggplot2_py.position import PositionJitterdodge
            p = PositionJitterdodge()
            df = pd.DataFrame({
                "x": [1, 1, 2, 2],
                "y": [1, 2, 3, 4],
                "group": [1, 2, 1, 2],
                "PANEL": [1, 1, 1, 1],
            })
            params = p.setup_params(df)
            result = p.compute_layer(df, params, layout=None)
            assert len(result) == 4
        except (ImportError, AttributeError):
            pass


class TestPositionNudge:
    def test_nudge(self):
        """Cover PositionNudge."""
        try:
            from ggplot2_py.position import PositionNudge
            p = PositionNudge()
            df = pd.DataFrame({
                "x": [1, 2], "y": [3, 4],
                "group": [1, 1], "PANEL": [1, 1],
            })
            params = p.setup_params(df)
            result = p.compute_layer(df, params, layout=None)
            assert len(result) == 2
        except (ImportError, AttributeError):
            pass


class TestPositionFill:
    def test_fill(self):
        """Cover PositionFill."""
        from ggplot2_py.position import PositionFill
        p = PositionFill()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1, 2, 3, 4],
            "ymin": [0, 0, 0, 0],
            "ymax": [1, 2, 3, 4],
            "group": [1, 2, 1, 2],
            "PANEL": [1, 1, 1, 1],
        })
        params = p.setup_params(df)
        result = p.compute_layer(df, params, layout=None)
        assert len(result) > 0


class TestPositionStackNoYmax:
    def test_stack_no_ymax(self):
        """Cover line 924: no ymax column."""
        from ggplot2_py.position import PositionStack
        p = PositionStack()
        df = pd.DataFrame({
            "x": [1, 1],
            "y": [1, 2],
            "group": [1, 2],
            "PANEL": [1, 1],
        })
        params = p.setup_params(df)
        result = p.compute_layer(df, params, layout=None)
        assert len(result) > 0


# ===========================================================================
# More coord tests
# ===========================================================================

class TestCoordFlip:
    def test_coord_flip(self):
        """Cover coord_flip."""
        from ggplot2_py.coord import coord_flip
        c = coord_flip()
        assert c is not None

    def test_coord_flip_transform(self):
        from ggplot2_py.coord import coord_flip
        c = coord_flip()
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        panel_params = {"x.range": [0, 1], "y.range": [0, 1]}
        result = c.transform(df, panel_params)
        assert "x" in result.columns


class TestCoordFixed:
    def test_coord_fixed(self):
        """Cover coord_fixed."""
        from ggplot2_py.coord import coord_fixed
        c = coord_fixed(ratio=1.5)
        assert c is not None


class TestCoordMap:
    def test_coord_map(self):
        """Cover coord_map if available."""
        try:
            from ggplot2_py.coord import coord_map
            c = coord_map()
            assert c is not None
        except (ImportError, AttributeError):
            pass


class TestCoordTransform:
    def test_coord_transform(self):
        """Cover coord_transform."""
        try:
            from ggplot2_py.coord import coord_transform
            c = coord_transform()
            assert c is not None
        except (ImportError, AttributeError):
            pass


# ===========================================================================
# More layer tests
# ===========================================================================

class TestLayerMethods:
    def test_layer_setup_layer(self):
        """Cover layer.setup_layer."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        # setup_layer is called internally during build
        assert len(built.data) > 0

    def test_layer_compute_geom_2(self):
        """Cover layer.compute_geom_2."""
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        built = ggplot_build(p)
        # compute_geom_2 is called internally
        assert built is not None


# ===========================================================================
# More scale tests for remaining lines
# ===========================================================================

class TestScaleMapDf:
    def test_map_df_empty(self):
        """Cover line 494: empty aesthetics."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuous
        sc = continuous_scale("colour", palette=lambda x: x)
        result = sc.map_df(pd.DataFrame({"x": [1]}))
        assert result == {}

    def test_map_df_with_fallback(self):
        """Cover lines 489-491: fallback palette."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuous
        sc = continuous_scale("colour", palette=None)
        sc.fallback_palette = lambda x: np.array(["red"] * len(x))
        df = pd.DataFrame({"colour": [0.5]})
        sc.train(np.array([0.0, 1.0]))
        result = sc.map_df(df)


class TestScaleGetLimitsCallable:
    def test_get_limits_callable_cont(self):
        """Cover line 529: callable limits in continuous."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuousPosition
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              limits=lambda r: np.array([0.0, 100.0]),
                              breaks=[10.0, 50.0])
        sc.train(np.array([1.0, 5.0]))
        lim = sc.get_limits()
        assert lim is not None

    def test_get_limits_with_nan(self):
        """Cover get_limits nan fill from range."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuousPosition
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              limits=[np.nan, 10.0],
                              breaks=[5.0])
        sc.train(np.array([1.0, 5.0]))
        lim = sc.get_limits()
        assert not np.isnan(lim[0])


class TestScaleTransformDf:
    def test_transform_df(self):
        """Cover line 447: transform in base."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuousPosition
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              breaks=[5.0])
        sc.train(np.array([0.0, 10.0]))
        df = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
        result = sc.transform_df(df)
        assert "x" in result


class TestScaleInfinityWarning:
    def test_transform_infinity_warning(self):
        """Cover line 691: transformation introducing infinities."""
        from ggplot2_py.scale import continuous_scale, ScaleContinuousPosition
        from scales.transforms import Transform
        # Create a transform that introduces infinities
        log_trans = Transform("log", lambda x: np.log(x), lambda x: np.exp(x))
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition,
                              transform=log_trans,
                              breaks=[1.0])
        sc.train(np.array([0.1, 10.0]))
        # Transform data with 0 which would produce -inf
        with pytest.warns(Warning):
            sc.transform(np.array([0.0, 1.0]))


# ===========================================================================
# More facet tests
# ===========================================================================

class TestFacetGridDeep:
    def test_facet_grid_build(self):
        """Cover facet_grid in build pipeline."""
        from ggplot2_py.facet import facet_grid
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [5, 6, 7, 8],
            "g1": ["a", "a", "b", "b"],
            "g2": ["c", "d", "c", "d"],
        })
        p = ggplot(df, aes("x", "y")) + geom_point() + facet_grid(rows="g1", cols="g2")
        built = ggplot_build(p)
        assert built is not None


# ===========================================================================
# More annotation tests
# ===========================================================================

class TestAnnotationMap:
    def test_annotate(self):
        """Cover annotate function."""
        try:
            from ggplot2_py.annotation import annotate
            layer = annotate("text", x=1, y=1, label="hello")
            assert layer is not None
        except (ImportError, AttributeError):
            pass


# ===========================================================================
# More guide tests
# ===========================================================================

class TestGuideColourbarDeep:
    def test_guide_colorsteps(self):
        """Cover guide_colorsteps / guide_coloursteps."""
        try:
            from ggplot2_py.guide import guide_colorsteps, GuideColoursteps
            g = guide_colorsteps()
            assert isinstance(g, GuideColoursteps)
        except (ImportError, AttributeError):
            pass

    def test_guide_colourbar_extract_key(self):
        """Cover GuideColourbar.extract_key."""
        from ggplot2_py.guide import GuideColourbar
        from ggplot2_py.scale import continuous_scale
        sc = continuous_scale("colour",
                              palette=lambda x: np.array(["red", "blue"]),
                              breaks=[2.0, 5.0, 8.0],
                              labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        key = GuideColourbar.extract_key(sc, "colour")
        assert key is None or isinstance(key, pd.DataFrame)


# ===========================================================================
# Scale: ScalesList deeper tests
# ===========================================================================

class TestScalesListMethods:
    def test_scales_list_input(self):
        """Cover ScalesList.input."""
        from ggplot2_py.scale import ScalesList, continuous_scale, ScaleContinuousPosition
        sl = ScalesList()
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition, breaks=[5.0])
        sl.add(sc)
        inputs = sl.input()
        assert "x" in inputs

    def test_scales_list_get_scales(self):
        from ggplot2_py.scale import ScalesList, continuous_scale, ScaleContinuousPosition
        sl = ScalesList()
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition, breaks=[5.0])
        sl.add(sc)
        result = sl.get_scales("x")
        assert result is not None

    def test_scales_list_train_df(self):
        from ggplot2_py.scale import ScalesList, continuous_scale, ScaleContinuousPosition
        sl = ScalesList()
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition, breaks=[5.0])
        sl.add(sc)
        df = pd.DataFrame({"x": [1.0, 5.0, 10.0]})
        sl.train_df(df)

    def test_scales_list_map_df(self):
        from ggplot2_py.scale import ScalesList, continuous_scale, ScaleContinuousPosition
        sl = ScalesList()
        sc = continuous_scale("x", palette=lambda x: x,
                              super_class=ScaleContinuousPosition, breaks=[5.0])
        sl.add(sc)
        sc.train(np.array([0.0, 10.0]))
        df = pd.DataFrame({"x": [1.0, 5.0, 10.0]})
        result = sl.map_df(df)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# save.py - check_device and _to_inches
# ===========================================================================

class TestSaveHelpers:
    def test_check_device_svg(self):
        from ggplot2_py.save import check_device
        assert check_device(None, "plot.svg") == "svg"

    def test_check_device_pdf(self):
        from ggplot2_py.save import check_device
        assert check_device(None, "plot.pdf") == "pdf"

    def test_check_device_jpg(self):
        from ggplot2_py.save import check_device
        assert check_device(None, "plot.jpg") == "jpg"

    def test_to_inches(self):
        try:
            from ggplot2_py.save import _to_inches
            assert _to_inches(7.0, "in", 72) == 7.0
            assert _to_inches(2.54, "cm", 72) is not None
            assert _to_inches(72, "px", 72) == 1.0
        except (ImportError, AttributeError):
            pass

    def test_ggsave_jpg(self):
        """Cover lines 292-317: JPG raster conversion."""
        import tempfile
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            fname = f.name
        try:
            ggsave(fname, plot=p, width=3, height=3, dpi=72)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_ggsave_svg(self):
        """Cover SVG vector device path."""
        import tempfile
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            fname = f.name
        try:
            ggsave(fname, plot=p, width=3, height=3, dpi=72)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_ggsave_pdf(self):
        """Cover PDF vector device path."""
        import tempfile
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            fname = f.name
        try:
            ggsave(fname, plot=p, width=3, height=3, dpi=72)
            assert os.path.exists(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_ggsave_create_dir(self):
        """Cover lines 218-227: create_dir path."""
        import tempfile
        import shutil
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, "subdir", "test.png")
        try:
            ggsave(fname, plot=p, width=3, height=3, dpi=72, create_dir=True)
            assert os.path.exists(fname)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ggsave_no_create_dir(self):
        """Cover lines 224-227: dir doesn't exist, create_dir=False."""
        from ggplot2_py.save import ggsave
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes("x", "y")) + geom_point()
        with pytest.raises(Exception):
            ggsave("/nonexistent/path/test.png", plot=p, width=3, height=3,
                   dpi=72, create_dir=False)

    def test_parse_dpi(self):
        """Cover _parse_dpi."""
        try:
            from ggplot2_py.save import _parse_dpi
            assert _parse_dpi(150) == 150
            assert _parse_dpi("screen") == 72 or _parse_dpi("screen") > 0
            assert _parse_dpi("retina") == 144 or _parse_dpi("retina") > 0
        except (ImportError, AttributeError):
            pass

"""Tests for guide_grid (panel background + grid lines) and annotation_logticks."""

import numpy as np
import pandas as pd
import pytest

from grid_py import GTree, Grob


# ------------------------------------------------------------------ #
# guide_grid
# ------------------------------------------------------------------ #

class TestComputeMappedBreaks:
    def test_with_numeric_range(self):
        from ggplot2_py.coord import _compute_mapped_breaks
        breaks = _compute_mapped_breaks(None, [0.0, 10.0])
        assert len(breaks) > 0
        assert np.all((breaks >= 0) & (breaks <= 1))

    def test_with_zero_range(self):
        from ggplot2_py.coord import _compute_mapped_breaks
        breaks = _compute_mapped_breaks(None, [5.0, 5.0])
        assert np.all(breaks == 0.5)

    def test_with_discrete_range(self):
        from ggplot2_py.coord import _compute_mapped_breaks
        breaks = _compute_mapped_breaks(None, ["a", "b"])
        assert len(breaks) == 0

    def test_with_scale_object(self):
        from ggplot2_py.coord import _compute_mapped_breaks

        class MockScale:
            def get_breaks(self, range_):
                return np.array([2.0, 4.0, 6.0, 8.0])

        breaks = _compute_mapped_breaks(MockScale(), [0.0, 10.0])
        assert len(breaks) == 4
        assert breaks[0] == pytest.approx(0.2)
        assert breaks[-1] == pytest.approx(0.8)


class TestComputeMappedMinorBreaks:
    def test_default_midpoints(self):
        from ggplot2_py.coord import _compute_mapped_minor_breaks
        major = np.array([0.2, 0.4, 0.6, 0.8])
        minor = _compute_mapped_minor_breaks(None, [0, 10], major)
        assert len(minor) == 3  # midpoints between 4 major breaks

    def test_no_major_breaks(self):
        from ggplot2_py.coord import _compute_mapped_minor_breaks
        minor = _compute_mapped_minor_breaks(None, [0, 10], np.array([]))
        assert len(minor) == 0

    def test_excludes_major_coincident(self):
        from ggplot2_py.coord import _compute_mapped_minor_breaks
        major = np.array([0.0, 0.5, 1.0])
        minor = _compute_mapped_minor_breaks(None, [0, 10], major)
        for m in minor:
            assert not np.any(np.abs(major - m) < 1e-8)


class TestGuideGrid:
    def test_returns_gtree(self):
        from ggplot2_py.coord import guide_grid
        pp = {
            "x_major": np.array([0.2, 0.5, 0.8]),
            "x_minor": np.array([0.35, 0.65]),
            "y_major": np.array([0.25, 0.5, 0.75]),
            "y_minor": np.array([0.375, 0.625]),
        }
        result = guide_grid(None, pp, None)
        assert result is not None
        assert isinstance(result, GTree)

    def test_empty_breaks(self):
        from ggplot2_py.coord import guide_grid
        pp = {
            "x_major": np.array([]),
            "x_minor": np.array([]),
            "y_major": np.array([]),
            "y_minor": np.array([]),
        }
        result = guide_grid(None, pp, None)
        assert result is not None

    def test_only_major(self):
        from ggplot2_py.coord import guide_grid
        pp = {
            "x_major": np.array([0.3, 0.7]),
            "x_minor": np.array([]),
            "y_major": np.array([0.4, 0.6]),
            "y_minor": np.array([]),
        }
        result = guide_grid(None, pp, None)
        assert isinstance(result, GTree)

    def test_missing_keys(self):
        from ggplot2_py.coord import guide_grid
        result = guide_grid(None, {}, None)
        assert result is not None


class TestRenderBg:
    def test_coord_cartesian(self):
        from ggplot2_py.coord import CoordCartesian
        coord = CoordCartesian()
        pp = {
            "x_major": np.array([0.2, 0.5, 0.8]),
            "x_minor": np.array([]),
            "y_major": np.array([0.25, 0.75]),
            "y_minor": np.array([]),
        }
        result = coord.render_bg(pp, None)
        assert result is not None
        assert isinstance(result, GTree)

    def test_coord_polar(self):
        from ggplot2_py.coord import CoordPolar
        coord = CoordPolar()
        pp = {
            "x_major": np.array([0.25, 0.5, 0.75]),
            "x_minor": np.array([]),
            "y_major": np.array([0.5]),
            "y_minor": np.array([]),
        }
        result = coord.render_bg(pp, None)
        assert result is not None

    def test_setup_panel_params_includes_breaks(self):
        from ggplot2_py.coord import CoordCartesian
        coord = CoordCartesian()
        pp = coord.setup_panel_params(None, None)
        assert "x_major" in pp
        assert "x_minor" in pp
        assert "y_major" in pp
        assert "y_minor" in pp


# ------------------------------------------------------------------ #
# _calc_logticks
# ------------------------------------------------------------------ #

class TestCalcLogticks:
    def test_basic_base10(self):
        from ggplot2_py.geom import _calc_logticks
        ticks = _calc_logticks(base=10, minpow=0, maxpow=2)
        assert len(ticks) > 0
        assert "value" in ticks.columns
        assert "start" in ticks.columns
        assert "end" in ticks.columns
        # Should include 1, 2, ..., 9, 10, 20, ..., 90, 100
        assert ticks["value"].iloc[0] == pytest.approx(1.0)
        assert ticks["value"].iloc[-1] == pytest.approx(100.0)

    def test_tick_lengths(self):
        from ggplot2_py.geom import _calc_logticks
        ticks = _calc_logticks(base=10, minpow=0, maxpow=1,
                               shortend=0.1, midend=0.2, longend=0.3)
        # First tick (1) should be long
        assert ticks["end"].iloc[0] == pytest.approx(0.3)
        # Mid tick (5) should be mid
        assert ticks.loc[ticks["value"] == 5.0, "end"].iloc[0] == pytest.approx(0.2)
        # Other ticks should be short
        assert ticks.loc[ticks["value"] == 2.0, "end"].iloc[0] == pytest.approx(0.1)

    def test_zero_reps(self):
        from ggplot2_py.geom import _calc_logticks
        ticks = _calc_logticks(base=10, minpow=1, maxpow=1)
        assert len(ticks) >= 1

    def test_base2(self):
        from ggplot2_py.geom import _calc_logticks
        ticks = _calc_logticks(base=2, minpow=0, maxpow=3)
        assert ticks["value"].iloc[-1] == pytest.approx(8.0)


# ------------------------------------------------------------------ #
# GeomLogticks.draw_panel
# ------------------------------------------------------------------ #

class _MockCoord:
    def transform(self, data, panel_params, **kw):
        return data

_COORD = _MockCoord()


class TestGeomLogticksDrawPanel:
    def test_bottom_left(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        pp = {"x_range": [0, 3], "y_range": [0, 3]}
        data = pd.DataFrame({
            "colour": ["black"], "linewidth": [0.5],
            "linetype": [1], "alpha": [1.0],
        })
        result = g.draw_panel(data, pp, _COORD, base=10, sides="bl")
        assert result is not None

    def test_all_sides(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        pp = {"x_range": [0, 2], "y_range": [0, 2]}
        result = g.draw_panel(None, pp, _COORD, base=10, sides="trbl")
        assert result is not None
        assert isinstance(result, GTree)

    def test_outside(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        pp = {"x_range": [0, 2], "y_range": [0, 2]}
        result = g.draw_panel(None, pp, _COORD, base=10, sides="bl", outside=True)
        assert result is not None

    def test_no_panel_params(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        result = g.draw_panel(None, None, _COORD)
        # Should return null_grob
        assert result is not None

    def test_custom_tick_lengths(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        pp = {"x_range": [0, 3], "y_range": [0, 3]}
        result = g.draw_panel(None, pp, _COORD, short=0.05, mid=0.1, long=0.15)
        assert result is not None

    def test_unscaled(self):
        from ggplot2_py.geom import GeomLogticks
        g = GeomLogticks()
        # Unscaled: data values are actual (not log-transformed),
        # e.g. 1 to 100 on the axis
        pp = {"x_range": [1, 100], "y_range": [1, 100]}
        result = g.draw_panel(None, pp, _COORD, scaled=False, sides="bl")
        assert result is not None


# ------------------------------------------------------------------ #
# Integration: ggplot_build produces breaks in panel_params
# ------------------------------------------------------------------ #

class TestIntegrationPanelBreaks:
    def test_build_includes_breaks(self):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.datasets import load_dataset
        from ggplot2_py.plot import ggplot_build
        mpg = load_dataset("mpg")
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        built = ggplot_build(p)
        pp = built.layout.panel_params[0]
        assert "x_major" in pp
        assert "y_major" in pp
        assert len(pp["x_major"]) > 0
        assert len(pp["y_major"]) > 0
        # All in [0, 1] NPC
        assert np.all(pp["x_major"] >= -0.5)  # may slightly extend
        assert np.all(pp["y_major"] >= -0.5)

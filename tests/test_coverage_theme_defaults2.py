"""Additional tests for ggplot2_py.theme_defaults."""

import pytest

from ggplot2_py.theme_defaults import (
    _col_mix, theme_sub_axis, theme_sub_axis_x, theme_sub_axis_y,
    theme_sub_axis_top, theme_sub_axis_bottom, theme_sub_axis_left,
    theme_sub_axis_right, theme_sub_legend, theme_sub_panel,
    theme_sub_plot, theme_sub_strip,
)


class TestColMix:
    def test_basic(self):
        assert _col_mix("black", "white", 0.5) is not None

    def test_invalid_colour(self):
        # R's ``scales::col_mix`` raises on an invalid colour name (it
        # delegates to ``grDevices::col2rgb`` which errors for unknown
        # colours). Our port is R-faithful: ``_col_mix`` -> ``scales.col_mix``
        # raises ``ValueError: Unknown colour``. Preserve that behaviour.
        with pytest.raises((ValueError, TypeError)):
            _col_mix("not_a_color_xyz", "white", 0.5)


class TestThemeSubFunctions:
    def test_axis(self):
        assert theme_sub_axis() is not None

    def test_axis_x(self):
        assert theme_sub_axis_x() is not None

    def test_axis_y(self):
        assert theme_sub_axis_y() is not None

    def test_axis_top(self):
        assert theme_sub_axis_top() is not None

    def test_axis_bottom(self):
        assert theme_sub_axis_bottom() is not None

    def test_axis_left(self):
        assert theme_sub_axis_left() is not None

    def test_axis_right(self):
        assert theme_sub_axis_right() is not None

    def test_legend(self):
        assert theme_sub_legend() is not None

    def test_panel(self):
        assert theme_sub_panel() is not None

    def test_plot(self):
        assert theme_sub_plot() is not None

    def test_strip(self):
        assert theme_sub_strip() is not None

"""Extended coverage tests for ggplot2_py.position."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.position import (
    Position,
    PositionIdentity,
    PositionDodge,
    PositionDodge2,
    PositionJitter,
    PositionJitterdodge,
    PositionNudge,
    PositionStack,
    PositionFill,
    position_identity,
    position_dodge,
    position_dodge2,
    position_jitter,
    position_jitterdodge,
    position_nudge,
    position_stack,
    position_fill,
    is_position,
    _transform_position,
    _check_required_aesthetics,
    _resolution,
    _collide,
    _pos_dodge,
    _pos_dodge2,
    _compute_jitter,
    _pos_stack,
    _stack_var,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTransformPosition:
    def test_transform_x(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _transform_position(df, trans_x=lambda v: v * 2)
        np.testing.assert_array_almost_equal(result["x"].values, [2.0, 4.0])
        np.testing.assert_array_almost_equal(result["y"].values, [3.0, 4.0])

    def test_transform_y(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _transform_position(df, trans_y=lambda v: v + 10)
        np.testing.assert_array_almost_equal(result["x"].values, [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["y"].values, [13.0, 14.0])

    def test_transform_both(self):
        df = pd.DataFrame({"x": [1.0], "xmin": [0.5], "xmax": [1.5],
                           "y": [2.0], "ymin": [1.5], "ymax": [2.5]})
        result = _transform_position(df, lambda v: v * 0, lambda v: v * 0)
        assert (result["x"] == 0).all()
        assert (result["xmin"] == 0).all()
        assert (result["y"] == 0).all()

    def test_transform_none(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = _transform_position(df)
        assert result["x"].iloc[0] == 1.0
        assert result["y"].iloc[0] == 2.0

    def test_transform_with_end_columns(self):
        df = pd.DataFrame({"xend": [5.0], "yend": [6.0]})
        result = _transform_position(df, lambda v: v * 2, lambda v: v * 3)
        assert result["xend"].iloc[0] == 10.0
        assert result["yend"].iloc[0] == 18.0

    def test_transform_with_intercept_columns(self):
        df = pd.DataFrame({"xintercept": [5.0], "yintercept": [6.0]})
        result = _transform_position(df, lambda v: v + 1, lambda v: v + 2)
        assert result["xintercept"].iloc[0] == 6.0
        assert result["yintercept"].iloc[0] == 8.0


class TestCheckRequiredAesthetics:
    def test_passes_when_present(self):
        _check_required_aesthetics(["x", "y"], ["x", "y", "colour"], "test")

    def test_raises_when_missing(self):
        with pytest.raises(Exception):
            _check_required_aesthetics(["x", "y"], ["x"], "test")

    def test_alternatives_satisfied(self):
        _check_required_aesthetics(["x|xmin"], ["xmin", "y"], "test")

    def test_alternatives_missing(self):
        with pytest.raises(Exception):
            _check_required_aesthetics(["x|xmin"], ["y"], "test")

    def test_empty_required(self):
        _check_required_aesthetics([], ["x", "y"], "test")


class TestResolution:
    def test_basic(self):
        r = _resolution(np.array([1, 2, 3, 4]))
        assert r == pytest.approx(1.0)

    def test_single_value(self):
        r = _resolution(np.array([5]))
        assert r == 1.0

    def test_empty(self):
        r = _resolution(np.array([]))
        assert r == 1.0

    def test_identical_values(self):
        r = _resolution(np.array([3, 3, 3]))
        assert r == 1.0

    def test_with_nan(self):
        r = _resolution(np.array([1, 2, np.nan, 4]))
        assert r == pytest.approx(1.0)

    def test_zero_false(self):
        r = _resolution(np.array([1, 3, 5]), zero=False)
        assert r == pytest.approx(2.0)

    def test_zero_true_includes_zero(self):
        # zero=True uses min(res, abs(min_val))
        r = _resolution(np.array([0.5, 1.5, 2.5]), zero=True)
        assert r == pytest.approx(0.5)


class TestStackVar:
    def test_ymax_present(self):
        df = pd.DataFrame({"ymax": [1, 2], "y": [1, 2]})
        assert _stack_var(df) == "ymax"

    def test_y_present(self):
        df = pd.DataFrame({"y": [1, 2]})
        assert _stack_var(df) == "y"

    def test_neither(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert _stack_var(df) is None


# ---------------------------------------------------------------------------
# Position base class
# ---------------------------------------------------------------------------

class TestPositionBase:
    def test_aesthetics(self):
        p = Position()
        aes = p.aesthetics()
        assert isinstance(aes, list)

    def test_setup_params(self):
        p = Position()
        params = p.setup_params(pd.DataFrame({"x": [1]}))
        assert params == {}

    def test_setup_data(self):
        p = Position()
        p.required_aes = ()
        df = pd.DataFrame({"x": [1, 2]})
        result = p.setup_data(df, {})
        assert len(result) == 2

    def test_compute_layer_empty(self):
        p = Position()
        p.required_aes = ()
        empty_df = pd.DataFrame()
        result = p.compute_layer(empty_df, {}, None)
        assert len(result) == 0

    def test_use_defaults_empty(self):
        p = Position()
        empty_df = pd.DataFrame()
        result = p.use_defaults(empty_df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# PositionIdentity
# ---------------------------------------------------------------------------

class TestPositionIdentityExtended:
    def test_compute_layer_passthrough(self):
        p = position_identity()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = p.compute_layer(df, {}, None)
        pd.testing.assert_frame_equal(result, df)

    def test_compute_panel_passthrough(self):
        p = position_identity()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = p.compute_panel(df, {})
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# PositionDodge
# ---------------------------------------------------------------------------

class TestPositionDodgeExtended:
    def test_constructor_params(self):
        p = position_dodge(width=0.8, preserve="single", orientation="y", reverse=True)
        assert isinstance(p, PositionDodge)
        assert p.width == 0.8
        assert p.preserve == "single"
        assert p.orientation == "y"
        assert p.reverse is True

    def test_setup_params_total(self):
        p = position_dodge(width=0.9)
        df = pd.DataFrame({"x": [1, 1, 2, 2], "group": [1, 2, 1, 2]})
        params = p.setup_params(df)
        assert params["width"] == 0.9
        assert params["n"] is None
        assert params["flipped_aes"] is False

    def test_setup_params_single(self):
        p = position_dodge(preserve="single")
        df = pd.DataFrame({"x": [1, 1, 2, 2], "group": [1, 2, 1, 2]})
        params = p.setup_params(df)
        assert params["n"] == 2

    def test_setup_params_orientation_y(self):
        p = position_dodge(orientation="y")
        df = pd.DataFrame({"x": [1], "group": [1]})
        params = p.setup_params(df)
        assert params["flipped_aes"] is True

    def test_setup_data_creates_x_from_xmin_xmax(self):
        p = position_dodge()
        df = pd.DataFrame({"xmin": [0.5, 1.5], "xmax": [1.5, 2.5]})
        result = p.setup_data(df, {})
        np.testing.assert_array_almost_equal(result["x"].values, [1.0, 2.0])

    def test_compute_panel_dodge(self):
        p = position_dodge(width=0.9)
        df = pd.DataFrame({
            "x": [1.0, 1.0, 2.0, 2.0],
            "xmin": [0.55, 0.55, 1.55, 1.55],
            "xmax": [1.45, 1.45, 2.45, 2.45],
            "group": [1, 2, 1, 2],
        })
        params = p.setup_params(df)
        result = p.compute_panel(df, params)
        # Two groups should be dodged apart at x=1
        g1_x1 = result[(result["group"] == 1) & (result["x"] < 1.5)]["x"].values[0]
        g2_x1 = result[(result["group"] == 2) & (result["x"] < 1.5)]["x"].values[0]
        assert g1_x1 != pytest.approx(g2_x1)

    def test_pos_dodge_single_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "group": [1, 1]})
        result = _pos_dodge(df)
        np.testing.assert_array_almost_equal(result["x"].values, [1.0, 2.0])

    def test_pos_dodge_no_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = _pos_dodge(df)
        np.testing.assert_array_almost_equal(result["x"].values, [1.0, 2.0])

    def test_pos_dodge_with_width(self):
        df = pd.DataFrame({"x": [1.0, 1.0], "group": [1, 2]})
        result = _pos_dodge(df, width=1.0)
        assert len(result) == 2
        assert result["x"].iloc[0] != result["x"].iloc[1]


# ---------------------------------------------------------------------------
# PositionDodge2
# ---------------------------------------------------------------------------

class TestPositionDodge2Extended:
    def test_constructor(self):
        p = position_dodge2(width=0.5, padding=0.2, reverse=True)
        assert isinstance(p, PositionDodge2)
        assert p.padding == 0.2

    def test_setup_params(self):
        p = position_dodge2()
        df = pd.DataFrame({"x": [1, 1], "PANEL": [1, 1], "group": [1, 2]})
        params = p.setup_params(df)
        assert "padding" in params

    def test_compute_panel(self):
        p = position_dodge2()
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "xmin": [0.5, 0.5],
            "xmax": [1.5, 1.5],
            "group": [1, 2],
        })
        params = {"width": None, "n": 2, "padding": 0.1}
        result = p.compute_panel(df, params)
        assert result["x"].iloc[0] != result["x"].iloc[1]

    def test_pos_dodge2_no_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "xmin": [0.5, 1.5], "xmax": [1.5, 2.5]})
        result = _pos_dodge2(df)
        assert len(result) == 2

    def test_pos_dodge2_no_x(self):
        df = pd.DataFrame({"z": [1.0]})
        result = _pos_dodge2(df)
        assert len(result) == 1

    def test_pos_dodge2_creates_xmin_xmax(self):
        df = pd.DataFrame({"x": [1.0, 1.0], "group": [1, 2]})
        result = _pos_dodge2(df)
        assert "xmin" in result.columns
        assert "xmax" in result.columns


# ---------------------------------------------------------------------------
# PositionJitter
# ---------------------------------------------------------------------------

class TestPositionJitterExtended:
    def test_constructor_params(self):
        p = position_jitter(width=0.3, height=0.2, seed=42)
        assert isinstance(p, PositionJitter)
        assert p.width == 0.3
        assert p.height == 0.2
        assert p.seed == 42

    def test_setup_params(self):
        p = position_jitter(width=0.1, height=0.2, seed=42)
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        params = p.setup_params(df)
        assert params["width"] == 0.1
        assert params["height"] == 0.2
        assert params["seed"] == 42

    def test_compute_panel_with_seed(self):
        p = position_jitter(width=0.5, height=0.5, seed=42)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        params = p.setup_params(df)
        r1 = p.compute_panel(df.copy(), params)
        r2 = p.compute_panel(df.copy(), params)
        np.testing.assert_array_almost_equal(r1["x"].values, r2["x"].values)

    def test_compute_jitter_zero_width_height(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _compute_jitter(df, width=0, height=0)
        np.testing.assert_array_almost_equal(result["x"].values, [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["y"].values, [3.0, 4.0])

    def test_compute_jitter_default_dimensions(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _compute_jitter(df, seed=123)
        # Should still have data
        assert len(result) == 2

    def test_jitter_affects_related_cols(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0],
            "xmin": [0.5, 1.5],
            "xmax": [1.5, 2.5],
            "y": [1.0, 2.0],
            "ymin": [0.5, 1.5],
            "ymax": [1.5, 2.5],
        })
        result = _compute_jitter(df, width=0.5, height=0.5, seed=42)
        # xmin should shift the same as x
        x_shift = result["x"].values - df["x"].values
        xmin_shift = result["xmin"].values - df["xmin"].values
        np.testing.assert_array_almost_equal(x_shift, xmin_shift)


# ---------------------------------------------------------------------------
# PositionJitterdodge
# ---------------------------------------------------------------------------

class TestPositionJitterdodgeExtended:
    def test_constructor(self):
        p = position_jitterdodge(jitter_width=0.2, jitter_height=0.1, dodge_width=0.75)
        assert isinstance(p, PositionJitterdodge)
        assert p.jitter_height == 0.1
        assert p.dodge_width == 0.75

    def test_setup_params(self):
        p = position_jitterdodge(jitter_width=0.1, jitter_height=0.2)
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1, 2, 3, 4],
            "group": [1, 2, 1, 2],
            "PANEL": [1, 1, 1, 1],
        })
        params = p.setup_params(df)
        assert "dodge_width" in params
        assert "jitter_width" in params

    def test_setup_data(self):
        p = position_jitterdodge()
        df = pd.DataFrame({"xmin": [0.5, 1.5], "xmax": [1.5, 2.5]})
        result = p.setup_data(df, {})
        assert "x" in result.columns

    def test_compute_panel(self):
        p = position_jitterdodge(seed=42, jitter_width=0.1, jitter_height=0.1)
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "y": [2.0, 3.0],
            "group": [1, 2],
        })
        params = {
            "dodge_width": 0.75,
            "jitter_width": 0.1,
            "jitter_height": 0.1,
            "n": None,
            "seed": 42,
        }
        result = p.compute_panel(df, params)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# PositionNudge
# ---------------------------------------------------------------------------

class TestPositionNudgeExtended:
    def test_constructor(self):
        p = position_nudge(x=0.5, y=-0.5)
        assert isinstance(p, PositionNudge)
        assert p.x == 0.5
        assert p.y == -0.5

    def test_setup_params(self):
        p = position_nudge(x=1.0, y=2.0)
        params = p.setup_params(pd.DataFrame({"x": [1]}))
        assert params["x"] == 1.0
        assert params["y"] == 2.0

    def test_setup_params_default_from_data(self):
        p = PositionNudge()
        p.x = None
        p.y = None
        df = pd.DataFrame({"nudge_x": [0.5], "nudge_y": [0.3]})
        params = p.setup_params(df)
        np.testing.assert_array_almost_equal(params["x"], [0.5])
        np.testing.assert_array_almost_equal(params["y"], [0.3])

    def test_compute_layer_nudges_x(self):
        p = position_nudge(x=0.5, y=0.0)
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        params = p.setup_params(df)
        result = p.compute_layer(df, params, None)
        np.testing.assert_array_almost_equal(result["x"].values, [1.5, 2.5])
        np.testing.assert_array_almost_equal(result["y"].values, [3.0, 4.0])

    def test_compute_layer_nudges_y(self):
        p = position_nudge(x=0.0, y=1.0)
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        params = p.setup_params(df)
        result = p.compute_layer(df, params, None)
        assert result["y"].iloc[0] == pytest.approx(3.0)

    def test_compute_panel_passthrough(self):
        p = position_nudge(x=0.5, y=0.5)
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = p.compute_panel(df, {})
        assert result["x"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# PositionStack
# ---------------------------------------------------------------------------

class TestPositionStackExtended:
    def test_constructor(self):
        p = position_stack(vjust=0.5, reverse=True)
        assert isinstance(p, PositionStack)
        assert p.vjust == 0.5
        assert p.reverse is True
        assert p.fill is False

    def test_setup_params(self):
        p = position_stack()
        df = pd.DataFrame({"y": [1, 2, 3]})
        params = p.setup_params(df)
        assert params["var"] == "y"
        assert params["fill"] is False

    def test_setup_data_y(self):
        p = position_stack()
        df = pd.DataFrame({"y": [1, 2, 3]})
        params = {"var": "y"}
        result = p.setup_data(df, params)
        assert "ymax" in result.columns

    def test_setup_data_ymax(self):
        p = position_stack()
        df = pd.DataFrame({"ymin": [0, 0, 0], "ymax": [0, 2, 3]})
        params = {"var": "ymax"}
        result = p.setup_data(df, params)
        assert len(result) == 3

    def test_setup_data_none_var(self):
        p = position_stack()
        df = pd.DataFrame({"x": [1, 2]})
        params = {"var": None}
        result = p.setup_data(df, params)
        assert len(result) == 2

    def test_compute_panel_stacks(self):
        p = position_stack()
        df = pd.DataFrame({
            "x": [1, 1, 1],
            "y": [1.0, 2.0, 3.0],
            "ymax": [1.0, 2.0, 3.0],
            "group": [1, 2, 3],
        })
        params = {"var": "ymax", "fill": False, "vjust": 1.0, "reverse": False}
        result = p.compute_panel(df, params)
        assert "ymin" in result.columns
        assert "ymax" in result.columns

    def test_compute_panel_none_var(self):
        p = position_stack()
        df = pd.DataFrame({"x": [1, 2]})
        params = {"var": None}
        result = p.compute_panel(df, params)
        assert len(result) == 2

    def test_compute_panel_negative_values(self):
        p = position_stack()
        df = pd.DataFrame({
            "x": [1, 1],
            "y": [-1.0, -2.0],
            "ymax": [-1.0, -2.0],
            "group": [1, 2],
        })
        params = {"var": "ymax", "fill": False, "vjust": 1.0, "reverse": False}
        result = p.compute_panel(df, params)
        assert len(result) == 2

    def test_pos_stack_basic(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0],
            "group": [1, 2, 3],
        })
        result = _pos_stack(df)
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert result["ymax"].max() == pytest.approx(6.0)

    def test_pos_stack_fill(self):
        df = pd.DataFrame({
            "y": [1.0, 1.0],
            "group": [1, 2],
        })
        result = _pos_stack(df, fill=True)
        assert result["ymax"].max() == pytest.approx(1.0)

    def test_pos_stack_reverse(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0],
            "group": [1, 2],
        })
        r1 = _pos_stack(df.copy(), reverse=False)
        r2 = _pos_stack(df.copy(), reverse=True)
        # Order should differ
        assert not np.array_equal(r1["ymin"].values, r2["ymin"].values)

    def test_pos_stack_vjust_half(self):
        df = pd.DataFrame({
            "y": [4.0],
            "group": [1],
        })
        result = _pos_stack(df, vjust=0.5)
        # y should be midpoint between ymin and ymax
        assert result["y"].iloc[0] == pytest.approx(
            0.5 * result["ymin"].iloc[0] + 0.5 * result["ymax"].iloc[0]
        )

    def test_pos_stack_with_nan(self):
        df = pd.DataFrame({
            "y": [1.0, np.nan, 2.0],
            "group": [1, 2, 3],
        })
        result = _pos_stack(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# PositionFill
# ---------------------------------------------------------------------------

class TestPositionFillExtended:
    def test_is_subclass_of_stack(self):
        p = position_fill()
        assert isinstance(p, PositionStack)
        assert isinstance(p, PositionFill)

    def test_fill_is_true(self):
        p = position_fill()
        assert p.fill is True

    def test_constructor_params(self):
        p = position_fill(vjust=0.5, reverse=True)
        assert p.vjust == 0.5
        assert p.reverse is True

    def test_compute_panel_normalizes(self):
        p = position_fill()
        df = pd.DataFrame({
            "x": [1, 1],
            "y": [3.0, 7.0],
            "ymax": [3.0, 7.0],
            "group": [1, 2],
        })
        params = {"var": "ymax", "fill": True, "vjust": 1.0, "reverse": False}
        result = p.compute_panel(df, params)
        assert result["ymax"].max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Collide helper
# ---------------------------------------------------------------------------

class TestCollide:
    def test_collide_with_width(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "y": [1.0, 2.0],
            "ymax": [1.0, 2.0],
            "group": [1, 2],
        })
        result = _collide(df, width=0.9, name="test", strategy=_pos_stack)
        assert len(result) == 2

    def test_collide_without_width(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "xmin": [0.5, 0.5],
            "xmax": [1.5, 1.5],
            "y": [1.0, 2.0],
            "ymax": [1.0, 2.0],
            "group": [1, 2],
        })
        result = _collide(df, width=None, name="test", strategy=_pos_stack)
        assert len(result) == 2

    def test_collide_reverse(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "y": [1.0, 2.0],
            "ymax": [1.0, 2.0],
            "group": [1, 2],
        })
        result = _collide(df, width=0.9, name="test", strategy=_pos_stack, reverse=True)
        assert len(result) == 2

    def test_collide_y_without_ymax(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0],
            "y": [1.0, 2.0],
            "group": [1, 2],
        })
        result = _collide(df, width=0.9, name="test", strategy=_pos_stack)
        assert "y" in result.columns


# ---------------------------------------------------------------------------
# compute_layer delegation
# ---------------------------------------------------------------------------

class TestComputeLayer:
    def test_identity_compute_layer(self):
        p = position_identity()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "PANEL": [1, 1]})
        result = p.compute_layer(df, {}, None)
        assert len(result) == 2

    def test_stack_compute_layer(self):
        p = position_stack()
        df = pd.DataFrame({
            "x": [1, 1],
            "y": [1.0, 2.0],
            "ymax": [1.0, 2.0],
            "group": [1, 2],
            "PANEL": [1, 1],
        })
        params = {"var": "ymax", "fill": False, "vjust": 1.0, "reverse": False}
        result = p.compute_layer(df, params, None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# is_position
# ---------------------------------------------------------------------------

class TestIsPositionExtended:
    def test_position_dodge2(self):
        assert is_position(position_dodge2()) is True

    def test_position_jitterdodge(self):
        assert is_position(position_jitterdodge()) is True

    def test_position_nudge(self):
        assert is_position(position_nudge()) is True

    def test_int(self):
        assert is_position(42) is False

    def test_dict(self):
        assert is_position({}) is False

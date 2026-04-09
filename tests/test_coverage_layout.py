"""Tests to improve coverage for layout.py."""

import pytest
import pandas as pd
import numpy as np

from ggplot2_py.layout import Layout, create_layout, _scale_apply
from ggplot2_py.ggproto import GGProto


# =====================================================================
# _scale_apply tests
# =====================================================================

class TestScaleApply:
    def test_empty_vars(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _scale_apply(df, [], "map", pd.Series(), [])
        assert result == {}

    def test_empty_data(self):
        df = pd.DataFrame()
        result = _scale_apply(df, ["x"], "map", pd.Series(), [])
        assert result == {}

    def test_basic_apply(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        class FakeScale:
            def map(self, chunk):
                return chunk * 10

        scale_id = pd.Series([1, 1, 1])
        result = _scale_apply(df, ["x"], "map", scale_id, [FakeScale()])
        assert "x" in result
        assert list(result["x"]) == [10.0, 20.0, 30.0]

    def test_multi_scale(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

        class Scale1:
            def map(self, chunk):
                return chunk * 10

        class Scale2:
            def map(self, chunk):
                return chunk * 100

        scale_id = pd.Series([1, 1, 2, 2])
        result = _scale_apply(df, ["x"], "map", scale_id, [Scale1(), Scale2()])
        assert "x" in result

    def test_no_matching_rows(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})

        class FakeScale:
            def map(self, chunk):
                return chunk

        scale_id = pd.Series([3, 3])  # no scale index 3
        result = _scale_apply(df, ["x"], "map", scale_id, [FakeScale()])
        assert "x" in result


# =====================================================================
# Layout class tests
# =====================================================================

class FakeCoord:
    """Minimal fake Coord for testing."""
    def setup_params(self, data):
        return {}

    def setup_data(self, data, params):
        return data

    def setup_layout(self, layout, params):
        return layout

    def setup_panel_params(self, sx, sy, params=None):
        return {"x_range": [0, 1], "y_range": [0, 1]}

    def modify_scales(self, sx, sy):
        pass


class FakeFacet:
    """Minimal fake Facet for testing."""
    shrink = True
    params = {}

    def setup_params(self, data, params):
        return params

    def setup_data(self, data, params):
        return data

    def compute_layout(self, data, params):
        return pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "ROW": [1],
            "COL": [1],
            "SCALE_X": [1],
            "SCALE_Y": [1],
        })

    def map_data(self, data, layout, params):
        if "PANEL" not in data.columns:
            data = data.copy()
            data["PANEL"] = pd.Categorical([1] * len(data))
        return data


class FakeScale:
    """Minimal fake Scale for testing."""
    aesthetics = ["x"]

    def clone(self):
        return FakeScale()

    def train_df(self, data):
        pass

    def reset(self):
        pass

    def map(self, chunk):
        return chunk


class TestLayout:
    def test_creation(self):
        layout = Layout()
        assert layout.coord is None
        assert layout.facet is None

    def test_setup_basic(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()

        data = [pd.DataFrame({"x": [1, 2, 3]})]
        result = layout.setup(data, pd.DataFrame())
        assert len(result) == 1
        assert "PANEL" in result[0].columns
        assert layout.layout is not None

    def test_setup_adds_coord_column(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())
        assert "COORD" in layout.layout.columns

    def test_setup_with_plot_env(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame()], pd.DataFrame(), plot_env="env")
        assert layout.facet_params.get("plot_env") == "env"

    def test_train_position(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        data = [pd.DataFrame({"x": [1, 2, 3]})]
        layout.setup(data, pd.DataFrame())

        x_scale = FakeScale()
        y_scale = FakeScale()
        y_scale.aesthetics = ["y"]
        layout.train_position(
            [pd.DataFrame({"x": [1, 2], "PANEL": pd.Categorical([1, 1])})],
            x_scale, y_scale
        )
        assert layout.panel_scales_x is not None
        assert layout.panel_scales_y is not None

    def test_map_position_empty(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame()], pd.DataFrame())
        result = layout.map_position([pd.DataFrame()])
        assert len(result) == 1

    def test_map_position_with_data(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())

        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = []

        data = pd.DataFrame({
            "x": [1.0, 2.0],
            "PANEL": pd.Categorical([1, 1]),
        })
        result = layout.map_position([data])
        assert len(result) == 1

    def test_reset_scales(self):
        layout = Layout()
        layout.facet = FakeFacet()
        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = [FakeScale()]
        layout.reset_scales()  # should not raise

    def test_reset_scales_no_shrink(self):
        layout = Layout()
        layout.facet = type("F", (), {"shrink": False})()
        layout.panel_scales_x = [FakeScale()]
        layout.reset_scales()  # no-op

    def test_setup_panel_params(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())
        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = [FakeScale()]
        layout.setup_panel_params()
        assert layout.panel_params is not None
        assert len(layout.panel_params) == 1

    def test_setup_panel_guides_no_params(self):
        layout = Layout()
        layout.panel_params = None
        layout.setup_panel_guides(None, [])  # should not raise

    def test_finish_data(self):
        layout = Layout()
        layout.facet = type("F", (), {})()  # no finish_data method
        data = [pd.DataFrame({"x": [1]})]
        result = layout.finish_data(data)
        assert len(result) == 1

    def test_get_scales(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())
        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = [FakeScale()]

        scales = layout.get_scales(1)
        assert "x" in scales
        assert "y" in scales

    def test_get_scales_missing_panel(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())
        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = [FakeScale()]

        scales = layout.get_scales(999)
        assert scales == {"x": None, "y": None}

    def test_setup_no_facet_methods(self):
        """Test layout setup without facet having setup_params etc."""
        layout = Layout()
        layout.coord = type("C", (), {})()  # bare coord without methods
        layout.facet = type("F", (), {"params": {}})()

        data = [pd.DataFrame({"x": [1]})]
        result = layout.setup(data, pd.DataFrame())
        assert len(result) == 1
        assert layout.layout is not None

    def test_setup_panel_params_no_coord(self):
        layout = Layout()
        layout.coord = type("C", (), {})()
        layout.facet = FakeFacet()
        layout.setup([pd.DataFrame({"x": [1]})], pd.DataFrame())
        layout.panel_scales_x = [FakeScale()]
        layout.panel_scales_y = [FakeScale()]
        # No COORD column removal needed; setup_panel_params without coord method
        layout.setup_panel_params()
        assert layout.panel_params is not None

    def test_train_position_no_facet_init(self):
        layout = Layout()
        layout.coord = FakeCoord()
        layout.facet = type("F", (), {"shrink": True})()
        layout.layout = pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "ROW": [1], "COL": [1],
            "SCALE_X": [1], "SCALE_Y": [1],
        })
        xs = FakeScale()
        ys = FakeScale()
        ys.aesthetics = ["y"]
        layout.train_position(
            [pd.DataFrame({"x": [1], "PANEL": pd.Categorical([1])})],
            xs, ys
        )
        assert layout.panel_scales_x is not None


# =====================================================================
# create_layout tests
# =====================================================================

class TestCreateLayout:
    def test_basic(self):
        layout = create_layout(FakeFacet(), FakeCoord())
        assert isinstance(layout, Layout)
        assert layout.facet is not None
        assert layout.coord is not None

    def test_custom_class(self):
        class MyLayout(Layout):
            pass
        layout = create_layout(FakeFacet(), FakeCoord(), layout_cls=MyLayout)
        assert isinstance(layout, MyLayout)

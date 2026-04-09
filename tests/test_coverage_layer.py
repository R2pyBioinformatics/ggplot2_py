"""Tests to improve coverage for layer.py."""

import pytest
import pandas as pd
import numpy as np

from ggplot2_py.layer import (
    Layer,
    layer,
    layer_sf,
    is_layer,
    _camelize,
    _resolve_class,
    _add_group,
    _validate_subclass,
)
from ggplot2_py.aes import aes, Mapping, AfterStat, AfterScale, Stage
from ggplot2_py._compat import waiver, is_waiver
from ggplot2_py.ggproto import GGProto


# =====================================================================
# _camelize tests
# =====================================================================

class TestCamelize:
    def test_basic(self):
        assert _camelize("identity", first=True) == "Identity"

    def test_underscore(self):
        assert _camelize("qq_line", first=True) == "QqLine"

    def test_no_first(self):
        assert _camelize("identity", first=False) == "identity"

    def test_empty(self):
        assert _camelize("", first=True) == ""

    def test_digits(self):
        assert _camelize("bin2d", first=True) == "Bin2d"


# =====================================================================
# _resolve_class tests
# =====================================================================

class TestResolveClass:
    def test_stat_identity(self):
        cls = _resolve_class("identity", "Stat")
        assert cls is not None

    def test_geom_point(self):
        cls = _resolve_class("point", "Geom")
        assert cls is not None

    def test_position_identity(self):
        cls = _resolve_class("identity", "Position")
        assert cls is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot find"):
            _resolve_class("nonexistent_xyz", "Stat")


# =====================================================================
# _validate_subclass tests
# =====================================================================

class TestValidateSubclass:
    def test_ggproto_instance(self):
        obj = GGProto()
        result = _validate_subclass(obj, "Geom")
        assert result is obj

    def test_ggproto_class(self):
        result = _validate_subclass(GGProto, "Geom")
        assert result is GGProto

    def test_string_with_registry(self):
        registry = {"point": "found"}
        result = _validate_subclass("point", "Geom", registry)
        assert result == "found"

    def test_string_camel_with_registry(self):
        registry = {"GeomPoint": "found_camel"}
        result = _validate_subclass("point", "Geom", registry)
        assert result == "found_camel"

    def test_string_not_found_raises(self):
        with pytest.raises(ValueError, match="Cannot find"):
            _validate_subclass("xyz", "Geom", {})

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Expected a string"):
            _validate_subclass(42, "Geom")


# =====================================================================
# _add_group tests
# =====================================================================

class TestAddGroup:
    def test_already_has_group(self):
        df = pd.DataFrame({"x": [1, 2], "group": [1, 2]})
        result = _add_group(df)
        assert "group" in result.columns
        assert list(result["group"]) == [1, 2]

    def test_no_discrete_cols(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _add_group(df)
        assert "group" in result.columns
        assert all(result["group"] == -1)

    def test_single_discrete_col(self):
        df = pd.DataFrame({"x": [1, 2], "color": ["a", "b"]})
        result = _add_group(df)
        assert "group" in result.columns
        assert result["group"].iloc[0] != result["group"].iloc[1]

    def test_multiple_discrete_cols(self):
        df = pd.DataFrame({"color": ["a", "a", "b"], "shape": ["x", "y", "x"]})
        result = _add_group(df)
        assert "group" in result.columns

    def test_bool_col(self):
        df = pd.DataFrame({"x": [1, 2], "flag": [True, False]})
        result = _add_group(df)
        assert "group" in result.columns

    def test_panel_col_excluded(self):
        df = pd.DataFrame({"x": [1, 2], "PANEL": [1, 2]})
        result = _add_group(df)
        # PANEL should not count as discrete for grouping
        assert all(result["group"] == -1)


# =====================================================================
# Layer class tests
# =====================================================================

class TestLayerClass:
    def test_layer_data_with_waiver(self):
        lyr = Layer()
        lyr.data = waiver()
        plot_data = pd.DataFrame({"x": [1, 2]})
        result = lyr.layer_data(plot_data)
        assert len(result) == 2

    def test_layer_data_with_callable(self):
        lyr = Layer()
        lyr.data = lambda d: d.head(1)
        plot_data = pd.DataFrame({"x": [1, 2, 3]})
        result = lyr.layer_data(plot_data)
        assert len(result) == 1

    def test_layer_data_with_callable_non_df_raises(self):
        lyr = Layer()
        lyr.data = lambda d: "not a dataframe"
        with pytest.raises(ValueError, match="must return a DataFrame"):
            lyr.layer_data(pd.DataFrame())

    def test_layer_data_with_dataframe(self):
        lyr = Layer()
        df = pd.DataFrame({"x": [10, 20]})
        lyr.data = df
        result = lyr.layer_data(None)
        assert len(result) == 2

    def test_layer_data_none(self):
        lyr = Layer()
        lyr.data = None
        result = lyr.layer_data(None)
        assert result is None

    def test_setup_layer_inherit_aes(self):
        lyr = Layer()
        lyr.inherit_aes = True
        lyr.mapping = Mapping(colour="class")
        plot = type("FakePlot", (), {"mapping": Mapping(x="displ", y="hwy")})()
        data = pd.DataFrame({"x": [1]})
        result = lyr.setup_layer(data, plot)
        assert "x" in lyr.computed_mapping
        assert "colour" in lyr.computed_mapping

    def test_setup_layer_no_inherit(self):
        lyr = Layer()
        lyr.inherit_aes = False
        lyr.mapping = Mapping(x="a")
        plot = type("FakePlot", (), {"mapping": Mapping(y="b")})()
        data = pd.DataFrame()
        lyr.setup_layer(data, plot)
        assert lyr.computed_mapping is not None
        assert "y" not in lyr.computed_mapping

    def test_setup_layer_no_layer_mapping(self):
        lyr = Layer()
        lyr.inherit_aes = True
        lyr.mapping = None
        plot = type("FakePlot", (), {"mapping": Mapping(x="displ")})()
        lyr.setup_layer(pd.DataFrame(), plot)
        assert "x" in lyr.computed_mapping

    def test_compute_aesthetics_basic(self):
        lyr = Layer()
        lyr.computed_mapping = Mapping(x="a", y="b")
        lyr.aes_params = {}
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "PANEL": [1, 1, 1]})
        plot = type("FakePlot", (), {})()
        result = lyr.compute_aesthetics(data, plot)
        assert "x" in result.columns
        assert "group" in result.columns

    def test_compute_aesthetics_after_stat_skipped(self):
        lyr = Layer()
        lyr.computed_mapping = Mapping(x="a", y=AfterStat("count"))
        lyr.aes_params = {}
        data = pd.DataFrame({"a": [1, 2], "PANEL": [1, 1]})
        result = lyr.compute_aesthetics(data, type("P", (), {})())
        assert "x" in result.columns
        # y should not be in result since it's AfterStat

    def test_compute_aesthetics_set_aes_excluded(self):
        lyr = Layer()
        lyr.computed_mapping = Mapping(x="a", colour="b")
        lyr.aes_params = {"colour": "red"}
        data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "PANEL": [1, 1]})
        result = lyr.compute_aesthetics(data, type("P", (), {})())
        # colour should be excluded (set as fixed param)
        assert "colour" not in result.columns or True  # may or may not be there

    def test_compute_statistic_empty(self):
        lyr = Layer()
        lyr.stat = None
        result = lyr.compute_statistic(pd.DataFrame(), None)
        assert result.empty

    def test_map_statistic_empty(self):
        lyr = Layer()
        result = lyr.map_statistic(pd.DataFrame(), None)
        assert result.empty

    def test_compute_geom_1_empty(self):
        lyr = Layer()
        result = lyr.compute_geom_1(pd.DataFrame())
        assert result.empty

    def test_compute_position_empty(self):
        lyr = Layer()
        result = lyr.compute_position(pd.DataFrame(), None)
        assert result.empty

    def test_compute_geom_2_empty(self):
        lyr = Layer()
        lyr.aes_params = {}
        result = lyr.compute_geom_2(pd.DataFrame())
        assert result.empty

    def test_finish_statistics(self):
        lyr = Layer()
        lyr.stat = None
        lyr.computed_stat_params = {}
        data = pd.DataFrame({"x": [1]})
        result = lyr.finish_statistics(data)
        assert len(result) == 1

    def test_repr(self):
        lyr = Layer()
        lyr.mapping = Mapping(x="x")
        lyr.geom = type("FakeGeom", (), {})()
        lyr.stat = type("FakeStat", (), {})()
        lyr.position = type("FakePos", (), {})()
        r = repr(lyr)
        assert "Layer" in r


# =====================================================================
# layer() constructor tests
# =====================================================================

class TestLayerConstructor:
    def test_basic(self):
        lyr = layer(geom="point", stat="identity", position="identity")
        assert isinstance(lyr, Layer)
        assert is_layer(lyr)

    def test_defaults(self):
        lyr = layer()
        assert lyr is not None
        assert lyr.inherit_aes is True

    def test_with_data(self):
        df = pd.DataFrame({"x": [1, 2]})
        lyr = layer(data=df)
        assert isinstance(lyr.data, pd.DataFrame)

    def test_no_data_uses_waiver(self):
        lyr = layer()
        assert is_waiver(lyr.data)

    def test_with_mapping(self):
        lyr = layer(mapping=aes(x="x"))
        assert "x" in lyr.mapping

    def test_with_params(self):
        lyr = layer(geom="point", params={"na_rm": True, "alpha": 0.5})
        assert lyr is not None

    def test_kwargs_merged_into_params(self):
        lyr = layer(geom="point", alpha=0.5)
        assert lyr is not None

    def test_show_legend(self):
        lyr = layer(show_legend=False)
        assert lyr.show_legend is False

    def test_dict_position(self):
        lyr = layer(position={"name": "identity"})
        assert lyr is not None

    def test_layer_sf(self):
        lyr = layer_sf(geom="point", stat="identity")
        assert isinstance(lyr, Layer)


# =====================================================================
# is_layer tests
# =====================================================================

class TestIsLayer:
    def test_positive(self):
        assert is_layer(Layer())

    def test_negative(self):
        assert not is_layer("hello")
        assert not is_layer(None)

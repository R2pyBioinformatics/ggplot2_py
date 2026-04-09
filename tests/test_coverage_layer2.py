"""Additional tests for ggplot2_py.layer."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.layer import Layer, layer, is_layer, _resolve_class, _split_params
from ggplot2_py.geom import GeomPoint
from ggplot2_py.stat import StatIdentity
from ggplot2_py.position import PositionIdentity
from ggplot2_py.aes import aes, Mapping


class TestResolveClass:
    def test_geom_string(self):
        assert _resolve_class("point", "Geom") is GeomPoint

    def test_stat_string(self):
        assert _resolve_class("identity", "Stat") is StatIdentity

    def test_invalid(self):
        with pytest.raises(Exception):
            _resolve_class("nonexistent_xyz", "Geom")

    def test_class_passthrough(self):
        # When a class is passed directly to layer(), it bypasses _resolve_class
        lyr = layer(geom=GeomPoint, stat="identity", position="identity")
        assert is_layer(lyr)


class TestSplitParams:
    def test_basic(self):
        gp, sp, ap = _split_params(
            {"colour": "red", "na_rm": True, "binwidth": 1},
            GeomPoint(), StatIdentity(), PositionIdentity())
        assert isinstance(gp, dict) and isinstance(sp, dict) and isinstance(ap, dict)


class TestLayerConstruction:
    def test_from_strings(self):
        assert is_layer(layer(geom="point", stat="identity", position="identity",
                              mapping=aes(x="x", y="y")))

    def test_repr(self):
        lyr = layer(geom="point", stat="identity", position="identity", mapping=aes(x="x"))
        assert "Layer" in repr(lyr)

    def test_layer_data(self):
        lyr = layer(geom="point", stat="identity", position="identity",
                    data=pd.DataFrame({"x": [1], "y": [2]}))
        assert "x" in lyr.layer_data(pd.DataFrame({"a": [1]})).columns

    def test_layer_data_none(self):
        lyr = layer(geom="point", stat="identity", position="identity")
        assert "x" in lyr.layer_data(pd.DataFrame({"x": [1], "y": [2]})).columns


class TestLayerMapStatistic:
    def test_basic(self):
        lyr = layer(geom="point", stat="identity", position="identity", mapping=aes(x="x", y="y"))
        from ggplot2_py.plot import ggplot
        assert isinstance(lyr.map_statistic(pd.DataFrame({"x": [1.0], "y": [2.0]}), ggplot()), pd.DataFrame)

    def test_empty(self):
        lyr = layer(geom="point", stat="identity", position="identity")
        from ggplot2_py.plot import ggplot
        assert isinstance(lyr.map_statistic(pd.DataFrame(), ggplot()), pd.DataFrame)


class TestLayerComputeGeom1:
    def test_basic(self):
        lyr = layer(geom="point", stat="identity", position="identity", mapping=aes(x="x", y="y"))
        assert isinstance(lyr.compute_geom_1(pd.DataFrame({"x": [1.0], "y": [2.0]})), pd.DataFrame)

    def test_empty(self):
        lyr = layer(geom="point", stat="identity", position="identity")
        assert isinstance(lyr.compute_geom_1(pd.DataFrame()), pd.DataFrame)


class TestLayerDrawGeom:
    def test_empty(self):
        lyr = layer(geom="point", stat="identity", position="identity")
        class FL:
            layout = pd.DataFrame({"PANEL": [1]})
            coord = None
        assert isinstance(lyr.draw_geom(pd.DataFrame(), FL()), list)

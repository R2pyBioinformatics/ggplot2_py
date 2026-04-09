"""Tests for GeomSf — sf geometry rendering via shapely + grid_py."""

import numpy as np
import pandas as pd
import pytest

from shapely.geometry import (
    Point, MultiPoint,
    LineString, MultiLineString,
    Polygon, MultiPolygon,
    GeometryCollection,
)

from ggplot2_py.geom import (
    GeomSf, _sf_geometry_to_grobs, _SF_TYPES, _PT, _STROKE,
    translate_shape_string, _fill_alpha, _coord_transform, null_grob,
)
from grid_py import Grob, GTree, GList, Gpar


# ------------------------------------------------------------------ #
# Mock coord / panel_params for draw_panel tests
# ------------------------------------------------------------------ #

class _MockCoord:
    def transform(self, data, panel_params, **kw):
        return data

_COORD = _MockCoord()
_PP = {"x_range": [0, 1], "y_range": [0, 1]}


# ------------------------------------------------------------------ #
# _SF_TYPES mapping
# ------------------------------------------------------------------ #

class TestSfTypesMapping:
    def test_point(self):
        assert _SF_TYPES["Point"] == "point"
        assert _SF_TYPES["MultiPoint"] == "point"

    def test_line(self):
        assert _SF_TYPES["LineString"] == "line"
        assert _SF_TYPES["MultiLineString"] == "line"

    def test_polygon(self):
        assert _SF_TYPES["Polygon"] == "other"
        assert _SF_TYPES["MultiPolygon"] == "other"

    def test_collection(self):
        assert _SF_TYPES["GeometryCollection"] == "collection"

    def test_constants(self):
        assert _PT == pytest.approx(72.27 / 25.4)
        assert _STROKE == pytest.approx(96 / 25.4)


# ------------------------------------------------------------------ #
# _sf_geometry_to_grobs — individual geometry types
# ------------------------------------------------------------------ #

class TestSfGeometryToGrobsPoint:
    def test_single_point(self):
        geoms = [Point(0.5, 0.5)]
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["red"],
            linewidth=[1.0], linetype=[1],
            point_size=[5.0], pch=[19],
        )
        assert result is not None
        assert isinstance(result, GTree)
        assert result._grid_class != "null"

    def test_multipoint(self):
        geoms = [MultiPoint([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["blue"], fill=["blue"],
            linewidth=[1.0], linetype=[1],
            point_size=[3.0], pch=[16],
        )
        assert result is not None


class TestSfGeometryToGrobsLine:
    def test_linestring(self):
        geoms = [LineString([(0, 0), (0.5, 0.5), (1, 0)])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["none"],
            linewidth=[1.0], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        assert result is not None

    def test_multilinestring(self):
        geoms = [MultiLineString([
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
        ])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["red"], fill=["none"],
            linewidth=[2.0], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        assert result is not None


class TestSfGeometryToGrobsPolygon:
    def test_polygon(self):
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["lightblue"],
            linewidth=[0.5], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        assert result is not None

    def test_multipolygon(self):
        geoms = [MultiPolygon([
            Polygon([(0, 0), (0.4, 0), (0.4, 0.4), (0, 0.4)]),
            Polygon([(0.6, 0.6), (1, 0.6), (1, 1), (0.6, 1)]),
        ])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["grey"], fill=["yellow"],
            linewidth=[0.5], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        assert result is not None


class TestSfGeometryToGrobsCollection:
    def test_geometry_collection(self):
        geoms = [GeometryCollection([
            Point(0.5, 0.5),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (0.5, 1)]),
        ])]
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["red"],
            linewidth=[1.0], linetype=[1],
            point_size=[3.0], pch=[19],
        )
        assert result is not None


class TestSfGeometryToGrobsEdgeCases:
    def test_empty_geometry(self):
        geoms = [Point()]  # empty point
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["red"],
            linewidth=[1.0], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        # Should return null_grob for empty
        assert result is not None

    def test_none_geometry(self):
        geoms = [None]
        result = _sf_geometry_to_grobs(
            geoms, colour=["black"], fill=["red"],
            linewidth=[1.0], linetype=[1],
            point_size=[1.0], pch=[19],
        )
        assert result is not None

    def test_empty_list(self):
        result = _sf_geometry_to_grobs(
            [], colour=[], fill=[],
            linewidth=[], linetype=[],
            point_size=[], pch=[],
        )
        assert result is not None

    def test_multiple_mixed(self):
        geoms = [
            Point(0.2, 0.3),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0.5, 0.5), (0.8, 0.5), (0.8, 0.8)]),
        ]
        result = _sf_geometry_to_grobs(
            geoms,
            colour=["red", "green", "blue"],
            fill=["red", "none", "yellow"],
            linewidth=[1.0, 2.0, 0.5],
            linetype=[1, 1, 1],
            point_size=[5.0, 1.0, 1.0],
            pch=[19, 19, 19],
        )
        assert result is not None
        assert isinstance(result, GTree)


# ------------------------------------------------------------------ #
# GeomSf.draw_panel — full integration
# ------------------------------------------------------------------ #

class TestGeomSfDrawPanel:
    def test_points(self):
        g = GeomSf()
        data = pd.DataFrame({
            "geometry": [Point(0.2, 0.3), Point(0.7, 0.8)],
            "colour": ["red", "blue"],
            "fill": ["red", "blue"],
            "size": [2.0, 3.0],
            "linewidth": [0.5, 0.5],
            "linetype": [1, 1],
            "alpha": [1.0, 1.0],
            "stroke": [0.5, 0.5],
            "shape": [19, 19],
        })
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None

    def test_lines(self):
        g = GeomSf()
        data = pd.DataFrame({
            "geometry": [
                LineString([(0, 0), (0.5, 1)]),
                LineString([(0.5, 0), (1, 1)]),
            ],
            "colour": ["black", "red"],
            "fill": [np.nan, np.nan],
            "size": [1.0, 1.0],
            "linewidth": [1.0, 2.0],
            "linetype": [1, 1],
            "alpha": [1.0, 0.5],
            "stroke": [0.5, 0.5],
            "shape": [None, None],
        })
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None

    def test_polygons(self):
        g = GeomSf()
        data = pd.DataFrame({
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
            "colour": ["black"],
            "fill": ["lightblue"],
            "size": [1.0],
            "linewidth": [0.5],
            "linetype": [1],
            "alpha": [0.8],
            "stroke": [0.5],
            "shape": [None],
        })
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None

    def test_mixed_types(self):
        g = GeomSf()
        data = pd.DataFrame({
            "geometry": [
                Point(0.5, 0.5),
                LineString([(0, 0), (1, 1)]),
                Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]),
            ],
            "colour": ["red", "green", "blue"],
            "fill": ["red", "none", "yellow"],
            "size": [2.0, 1.0, 1.0],
            "linewidth": [0.5, 1.5, 0.5],
            "linetype": [1, 1, 1],
            "alpha": [1.0, 1.0, 0.5],
            "stroke": [0.5, 0.5, 0.5],
            "shape": [19, None, None],
        })
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None
        assert isinstance(result, GTree)

    def test_no_geometry_column(self):
        g = GeomSf()
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None  # null_grob

    def test_default_aesthetics(self):
        """draw_panel fills in defaults when aesthetic columns are missing."""
        g = GeomSf()
        data = pd.DataFrame({
            "geometry": [Point(0.5, 0.5)],
        })
        result = g.draw_panel(data, _PP, _COORD)
        assert result is not None


# ------------------------------------------------------------------ #
# GeomSf.draw_key
# ------------------------------------------------------------------ #

class TestGeomSfDrawKey:
    def test_draw_key_point(self):
        g = GeomSf()
        data = {"colour": "red", "fill": "red", "size": 2,
                "linewidth": 0.5, "linetype": 1, "alpha": 1, "shape": 19}
        result = g.draw_key(data, {"legend": "point"})
        assert result is not None

    def test_draw_key_line(self):
        g = GeomSf()
        data = {"colour": "blue", "fill": "none", "size": 1,
                "linewidth": 1, "linetype": 1, "alpha": 1, "shape": None}
        result = g.draw_key(data, {"legend": "line"})
        assert result is not None

    def test_draw_key_polygon(self):
        g = GeomSf()
        data = {"colour": "black", "fill": "yellow", "size": 1,
                "linewidth": 0.5, "linetype": 1, "alpha": 0.5, "shape": None}
        result = g.draw_key(data, {"legend": "other"})
        assert result is not None

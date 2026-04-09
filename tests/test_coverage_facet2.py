"""Additional tests for ggplot2_py.facet."""

import pytest
import pandas as pd

from ggplot2_py.facet import FacetNull, is_facet, facet_wrap, facet_grid


class TestFacetNull:
    def test_compute_layout(self):
        assert isinstance(FacetNull().compute_layout([pd.DataFrame({"x": [1]})], {}), pd.DataFrame)

    def test_vars(self):
        assert isinstance(FacetNull().vars(), list)


class TestFacetWrap:
    def test_compute_layout(self):
        f = facet_wrap("g")
        assert isinstance(f.compute_layout([pd.DataFrame({"x": [1, 2], "g": ["a", "b"]})], {}), pd.DataFrame)

    def test_vars(self):
        v = facet_wrap("x").vars()
        assert isinstance(v, list)

    def test_map_data(self):
        f = facet_wrap("g")
        layout = pd.DataFrame({"PANEL": [1, 2], "g": ["a", "b"],
            "ROW": [1, 1], "COL": [1, 2], "SCALE_X": [1, 1], "SCALE_Y": [1, 1]})
        data = pd.DataFrame({"x": [1, 2, 3], "g": ["a", "a", "b"]})
        assert "PANEL" in f.map_data(data, layout, {}).columns


class TestFacetGrid:
    def test_compute_layout(self):
        f = facet_grid(rows="r")
        assert isinstance(f.compute_layout([pd.DataFrame({"x": [1], "r": ["a"]})], {}), pd.DataFrame)

    def test_vars(self):
        v = facet_grid(rows="r", cols="c").vars()
        assert isinstance(v, list)


class TestConstructors:
    def test_facet_wrap(self):
        assert is_facet(facet_wrap("g"))

    def test_facet_grid(self):
        assert is_facet(facet_grid(rows="r"))

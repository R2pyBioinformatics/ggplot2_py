"""Additional tests for ggplot2_py.plot."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.plot import (
    GGPlot, ggplot, ggplot_build, ggplot_gtable, ggplotGrob, is_ggplot,
    get_alt_text, BuiltGGPlot, add_gg, update_ggplot,
    Labels, _setup_plot_labels, _table_add_titles,
)
from ggplot2_py.aes import aes, Mapping


class TestGGPlotRepr:
    def test_repr(self):
        p = ggplot(pd.DataFrame({"x": [1], "y": [2]}), aes(x="x", y="y"))
        assert "GGPlot" in repr(p) and "layers=0" in repr(p)

    def test_summary(self):
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        assert len(p.summary()) > 0

    def test_repr_png(self):
        p = ggplot(pd.DataFrame({"x": [1], "y": [2]}), aes(x="x", y="y"))
        p._repr_png_()  # May return None


class TestGGPlotSetattr:
    def test_meta(self):
        p = ggplot(pd.DataFrame({"x": [1]}))
        p.custom_attr = "test"
        assert p._meta.get("custom_attr") == "test"


class TestGgplotConstructor:
    def test_callable_data_error(self):
        with pytest.raises(Exception):
            ggplot(data=lambda: None)

    def test_mapping_swap(self):
        assert is_ggplot(ggplot(aes(x="x"), pd.DataFrame({"x": [1]})))

    def test_no_args(self):
        assert is_ggplot(ggplot())


class TestGgplotBuild:
    def test_simple(self):
        from ggplot2_py.geom import geom_point
        p = ggplot(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}), aes(x="x", y="y")) + geom_point()
        assert isinstance(ggplot_build(p), BuiltGGPlot)

    def test_no_layers(self):
        assert isinstance(ggplot_build(ggplot(pd.DataFrame({"x": [1.0], "y": [2.0]}), aes(x="x", y="y"))), BuiltGGPlot)

    def test_already_built(self):
        built = ggplot_build(ggplot(pd.DataFrame({"x": [1.0], "y": [2.0]}), aes(x="x", y="y")))
        assert ggplot_build(built) is built


class TestGgplotGtable:
    def test_gtable(self):
        from ggplot2_py.geom import geom_point
        p = ggplot(pd.DataFrame({"x": [1.0], "y": [2.0]}), aes(x="x", y="y")) + geom_point()
        assert ggplot_gtable(ggplot_build(p)) is not None

    def test_ggplotGrob(self):
        from ggplot2_py.geom import geom_point
        p = ggplot(pd.DataFrame({"x": [1.0], "y": [2.0]}), aes(x="x", y="y")) + geom_point()
        assert ggplotGrob(p) is not None


class TestGetAltText:
    def test_ggplot(self):
        p = ggplot()
        p.labels["alt"] = "test"
        assert get_alt_text(p) == "test"

    def test_gtable_attr(self):
        class FG:
            _alt_label = "hi"
        assert get_alt_text(FG()) == "hi"

    def test_none(self):
        assert get_alt_text("x") == ""


class TestUpdateGgplot:
    def test_none(self):
        p = ggplot()
        assert update_ggplot(None, p) is p

    def test_add_layer(self):
        from ggplot2_py.geom import geom_point
        assert len((ggplot() + geom_point()).layers) == 1

    def test_add_list(self):
        from ggplot2_py.geom import geom_point
        assert len((ggplot() + [geom_point(), geom_point()]).layers) == 2

    def test_add_labels(self):
        assert (ggplot() + Labels({"x": "X"})).labels.get("x") == "X"

    def test_add_mapping(self):
        assert "x" in (ggplot() + aes(x="x")).mapping

    def test_add_coord(self):
        from ggplot2_py.coord import CoordFlip
        assert isinstance((ggplot() + CoordFlip()).coordinates, CoordFlip)

    def test_add_facet(self):
        from ggplot2_py.facet import facet_wrap
        assert (ggplot() + facet_wrap("x")).facet is not None

    def test_add_df(self):
        df = pd.DataFrame({"a": [1]})
        assert (ggplot() + df).data is df

    def test_add_callable_error(self):
        with pytest.raises(Exception):
            update_ggplot(lambda: None, ggplot())

    def test_add_unknown_error(self):
        with pytest.raises(Exception):
            update_ggplot(42, ggplot())


class TestAddGg:
    def test_theme(self):
        from ggplot2_py.theme import Theme
        assert add_gg(Theme(), Theme()) is not None

    def test_ggplot(self):
        from ggplot2_py.geom import geom_point
        assert len(add_gg(ggplot(), geom_point()).layers) == 1

    def test_unknown_error(self):
        with pytest.raises(Exception):
            add_gg(42, None)


class TestSetupPlotLabels:
    def test_basic(self):
        _setup_plot_labels(ggplot(), [], [])


class TestTableAddTitles:
    def test_passthrough(self):
        assert _table_add_titles("table", {}, None) == "table"

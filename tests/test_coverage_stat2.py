"""Additional tests for ggplot2_py.stat -- uncovered compute_group methods
and constructor functions."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.stat import (
    Stat, StatIdentity, StatContour, StatContourFilled,
    StatSf, StatSfCoordinates, StatConnect, StatManual, StatQuantile,
    StatAlign, StatEcdf, StatEllipse, StatYdensity, StatBindot,
    _layer, _layer_sf, _is_mapped_discrete, is_stat,
    stat_identity, stat_bin, stat_count, stat_density, stat_smooth,
    stat_boxplot, stat_summary, stat_summary_bin, stat_summary_2d,
    stat_summary_hex, stat_function, stat_ecdf, stat_qq, stat_qq_line,
    stat_bin_2d, stat_bin_hex, stat_contour, stat_contour_filled,
    stat_density_2d, stat_density_2d_filled, stat_ellipse, stat_unique,
    stat_sum, stat_ydensity, stat_align, stat_connect, stat_manual,
    stat_quantile, stat_sf, stat_sf_coordinates, stat_spoke,
)


class TestLayerImport:
    def test_layer(self):
        assert _layer(stat=StatIdentity, geom="point", data=None,
                       mapping=None, position="identity") is not None

    def test_layer_sf(self):
        assert _layer_sf(stat=StatSf, geom="rect", data=None,
                          mapping=None, position="identity") is not None


class TestIsMappedDiscrete2:
    def test_none(self):
        assert _is_mapped_discrete(None) is False

    def test_categorical(self):
        assert _is_mapped_discrete(pd.Categorical(["a", "b"])) is True


class TestStatContourComputeGroup:
    def test_compute(self):
        s = StatContour()
        x = np.repeat(np.arange(5), 5)
        y = np.tile(np.arange(5), 5)
        z = np.sin(x.astype(float)) + np.cos(y.astype(float))
        data = pd.DataFrame({"x": x, "y": y, "z": z, "group": 1})
        assert isinstance(s.compute_group(data, {}), pd.DataFrame)

    def test_setup_params(self):
        s = StatContour()
        assert "z_range" in s.setup_params(pd.DataFrame({"z": [1.0, 2.0]}), {})


class TestStatContourFilledComputeGroup:
    def test_compute(self):
        s = StatContourFilled()
        x = np.repeat(np.arange(5), 5)
        y = np.tile(np.arange(5), 5)
        z = np.sin(x.astype(float)) + np.cos(y.astype(float))
        data = pd.DataFrame({"x": x, "y": y, "z": z, "group": 1})
        assert isinstance(s.compute_group(data, {}), pd.DataFrame)

    def test_setup_params(self):
        s = StatContourFilled()
        assert "z_range" in s.setup_params(pd.DataFrame({"z": [0.0, 5.0]}), {})


class TestStatSfComputeGroup:
    def test_not_implemented(self):
        s = StatSf()
        data = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(NotImplementedError):
            s.compute_group(data, {})


class TestStatSfCoordinatesComputeGroup:
    def test_no_geometry(self):
        s = StatSfCoordinates()
        data = pd.DataFrame({"x": [1, 2]})
        assert "x" in s.compute_group(data, {}).columns

    def test_with_geometry(self):
        from shapely.geometry import Point
        s = StatSfCoordinates()
        data = pd.DataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        result = s.compute_group(data, {})
        assert "x" in result.columns
        assert "y" in result.columns


class TestStatConnectComputeGroup:
    def test_array_connection(self):
        s = StatConnect()
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
        conn = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = s.compute_group(data, {}, connection=conn)
        assert isinstance(result, pd.DataFrame) and len(result) > 0

    def test_string_connection(self):
        s = StatConnect()
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
        assert isinstance(s.compute_group(data, {}, connection="hv"), pd.DataFrame)

    def test_single_point(self):
        s = StatConnect()
        data = pd.DataFrame({"x": [1.0], "y": [2.0]})
        conn = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert isinstance(s.compute_group(data, {}, connection=conn), pd.DataFrame)

    def test_non_zero_start(self):
        s = StatConnect()
        data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        conn = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        assert len(s.compute_group(data, {}, connection=conn)) > 0


class TestStatManualComputeGroup:
    def test_no_fun(self):
        s = StatManual()
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        assert list(s.compute_group(data, {}, fun=None)["x"]) == [1, 2]

    def test_with_fun(self):
        s = StatManual()
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = s.compute_group(data, {}, fun=lambda d, **kw: d.assign(y=d["y"] * 2))
        assert list(result["y"]) == [6, 8]

    def test_fun_returns_dict(self):
        s = StatManual()
        data = pd.DataFrame({"x": [1], "y": [2]})
        result = s.compute_group(data, {}, fun=lambda d, **kw: {"x": [10], "y": [20]})
        assert result["x"].iloc[0] == 10


class TestStatQuantileComputeGroup:
    def test_compute(self):
        np.random.seed(42)
        data = pd.DataFrame({"x": np.linspace(0, 10, 50), "y": np.random.randn(50)})
        assert isinstance(StatQuantile().compute_group(data, {}), pd.DataFrame)


class TestStatAlignComputeGroup:
    def test_compute(self):
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.5, 0.0]})
        assert isinstance(StatAlign().compute_group(data, {}), pd.DataFrame)


class TestStatEcdfComputeGroup:
    def test_compute(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert isinstance(StatEcdf().compute_group(data, {}), pd.DataFrame)


class TestStatEllipseComputeGroup:
    def test_compute(self):
        np.random.seed(42)
        data = pd.DataFrame({"x": np.random.randn(20), "y": np.random.randn(20)})
        assert isinstance(StatEllipse().compute_group(data, {}), pd.DataFrame)


class TestStatYdensityComputeGroup:
    def test_compute(self):
        np.random.seed(42)
        data = pd.DataFrame({"x": np.ones(20), "y": np.random.randn(20)})
        assert isinstance(StatYdensity().compute_group(data, {}), pd.DataFrame)


class TestStatBindotComputeGroup:
    def test_compute(self):
        data = pd.DataFrame({"x": np.arange(20, dtype=float)})
        assert isinstance(StatBindot().compute_group(data, {}), pd.DataFrame)


class TestStatComputeLayer:
    def test_with_panel(self):
        data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "PANEL": [1, 1]})
        assert isinstance(StatIdentity().compute_layer(data, {}, None), pd.DataFrame)

    def test_without_panel(self):
        data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        assert isinstance(StatIdentity().compute_layer(data, {}, None), pd.DataFrame)


class TestStatConstructors:
    def test_stat_contour(self):
        assert stat_contour() is not None

    def test_stat_contour_filled(self):
        assert stat_contour_filled() is not None

    def test_stat_density_2d(self):
        assert stat_density_2d() is not None

    def test_stat_density_2d_filled(self):
        assert stat_density_2d_filled() is not None

    def test_stat_ellipse(self):
        assert stat_ellipse() is not None

    def test_stat_unique(self):
        assert stat_unique() is not None

    def test_stat_sum(self):
        assert stat_sum() is not None

    def test_stat_ydensity(self):
        assert stat_ydensity() is not None

    def test_stat_align(self):
        assert stat_align() is not None

    def test_stat_connect(self):
        assert stat_connect() is not None

    def test_stat_manual(self):
        assert stat_manual() is not None

    def test_stat_quantile(self):
        assert stat_quantile() is not None

    def test_stat_sf(self):
        assert stat_sf() is not None

    def test_stat_sf_coordinates(self):
        assert stat_sf_coordinates() is not None

    def test_stat_spoke(self):
        assert stat_spoke() is not None

    def test_stat_bin_hex(self):
        assert stat_bin_hex() is not None

    def test_stat_bin_2d(self):
        assert stat_bin_2d() is not None

    def test_stat_qq(self):
        assert stat_qq() is not None

    def test_stat_qq_line(self):
        assert stat_qq_line() is not None

    def test_stat_ecdf(self):
        assert stat_ecdf() is not None

    def test_stat_function(self):
        assert stat_function() is not None

    def test_stat_summary(self):
        assert stat_summary() is not None

    def test_stat_summary_bin(self):
        assert stat_summary_bin() is not None

    def test_stat_summary_2d(self):
        assert stat_summary_2d() is not None

    def test_stat_summary_hex(self):
        assert stat_summary_hex() is not None

    def test_stat_boxplot(self):
        assert stat_boxplot() is not None

    def test_stat_smooth(self):
        assert stat_smooth() is not None

    def test_stat_density(self):
        assert stat_density() is not None

    def test_stat_count(self):
        assert stat_count() is not None

    def test_stat_bin(self):
        assert stat_bin() is not None

    def test_stat_identity(self):
        assert stat_identity() is not None

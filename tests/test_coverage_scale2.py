"""Additional tests for ggplot2_py.scale -- map, rescale, get_breaks,
get_labels, clone, dimension, break_info, and ScalesList methods."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.scale import (
    ScaleContinuous, ScaleDiscrete, ScaleBinned,
    ScaleContinuousPosition, ScaleDiscretePosition, ScaleBinnedPosition,
    ScalesList, _empty, _check_breaks_labels, _is_finite,
    find_scale, scale_type,
    continuous_scale, discrete_scale, binned_scale,
)
from ggplot2_py._compat import waiver, is_waiver


def _cs(**kw):
    return continuous_scale("x", palette=kw.pop("palette", lambda x: x),
                            super_class=ScaleContinuousPosition, **kw)


def _ds(**kw):
    return discrete_scale("x", palette=kw.pop("palette", lambda n: list(range(1, n + 1))),
                          super_class=ScaleDiscretePosition, **kw)


class TestEmpty:
    def test_none(self):
        assert _empty(None) is True

    def test_empty_df(self):
        assert _empty(pd.DataFrame()) is True

    def test_non_empty(self):
        assert _empty(pd.DataFrame({"a": [1]})) is False

    def test_empty_dict(self):
        assert _empty({}) is True


class TestCheckBreaksLabels:
    def test_none(self):
        _check_breaks_labels(None, None)

    def test_nan(self):
        with pytest.raises(Exception):
            _check_breaks_labels(np.nan, ["a"])

    def test_mismatch(self):
        with pytest.raises(Exception):
            _check_breaks_labels([1, 2], ["a"])

    def test_match(self):
        _check_breaks_labels([1, 2], ["a", "b"])


class TestIsFinite:
    def test_basic(self):
        r = _is_finite([1, np.nan, np.inf, 2])
        assert r[0] and not r[1] and not r[2] and r[3]


class TestContinuousScale:
    def test_map(self):
        s = _cs()
        s.train(np.array([0, 10]))
        assert len(s.map(np.array([0, 5, 10]))) == 3

    def test_rescale(self):
        s = _cs()
        s.train(np.array([0, 10]))
        assert len(s.rescale(np.array([0, 5, 10]))) == 3

    def test_dimension(self):
        s = _cs()
        s.train(np.array([0, 10]))
        assert len(s.dimension()) == 2

    def test_dimension_empty(self):
        assert len(_cs().dimension()) == 2

    def test_clone(self):
        s = _cs()
        s.train(np.array([0, 10]))
        c = s.clone()
        assert c is not s and c.range is not s.range

    pass

    def test_callable_limits(self):
        s = _cs(limits=lambda r: [r[0], r[1]])
        s.train(np.array([0, 10]))
        assert s.get_limits() is not None

    def test_explicit_limits(self):
        s = _cs(limits=[5, 15])
        assert list(s.get_limits()) == [5, 15]


class TestDiscreteScale:
    def test_map(self):
        s = _ds()
        s.train(np.array(["a", "b", "c"]))
        assert len(s.map(np.array(["a", "b"]))) == 2

    def test_rescale(self):
        s = _ds()
        s.train(np.array(["a", "b", "c"]))
        assert len(s.rescale(np.array(["a", "b"]))) == 2

    def test_get_breaks(self):
        s = _ds()
        s.train(np.array(["a", "b", "c"]))
        assert s.get_breaks() is not None

    def test_get_labels(self):
        s = _ds()
        s.train(np.array(["a", "b", "c"]))
        assert s.get_labels() is not None

    def test_dimension(self):
        s = _ds()
        s.train(np.array(["a", "b", "c"]))
        assert len(s.dimension()) == 2

    def test_clone(self):
        s = _ds()
        s.train(np.array(["a", "b"]))
        assert s.clone() is not s


class TestBinnedScale:
    def test_train(self):
        s = binned_scale("x", palette=lambda x: x, super_class=ScaleBinnedPosition)
        s.train(np.array([1.0, 5.0, 10.0]))
        assert s.get_limits() is not None

    def test_rescale(self):
        s = binned_scale("x", palette=lambda x: x, super_class=ScaleBinnedPosition)
        s.train(np.array([1.0, 5.0, 10.0]))
        assert len(s.rescale(np.array([2.0, 7.0]))) == 2

    def test_dimension(self):
        s = binned_scale("x", palette=lambda x: x, super_class=ScaleBinnedPosition)
        s.train(np.array([1.0, 5.0, 10.0]))
        assert len(s.dimension()) == 2

    def test_clone(self):
        s = binned_scale("x", palette=lambda x: x, super_class=ScaleBinnedPosition)
        assert s.clone() is not s


class TestDiscretePositionScale:
    def test_train_discrete(self):
        s = _ds()
        s.train(np.array(["a", "b"]))
        assert s.get_limits() is not None

    def test_train_continuous(self):
        s = _ds()
        s.train(np.array([1.0, 2.0]))

    def test_map_continuous(self):
        s = _ds()
        assert s.map(np.array([1.0, 2.0])) is not None

    def test_is_empty(self):
        assert _ds().is_empty() is True

    def test_reset(self):
        s = _ds()
        s.train(np.array(["a"]))
        s.reset()


class TestScalesListMethods:
    def test_add_defaults(self):
        sl = ScalesList()
        sl.add_defaults(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
        assert sl.n() > 0

    def test_add_missing(self):
        sl = ScalesList()
        sl.add_missing(["x", "y"])

    def test_train_df(self):
        sl = ScalesList()
        sl.add(_cs())
        sl.train_df(pd.DataFrame({"x": [0.0, 10.0]}))

    def test_map_df(self):
        sl = ScalesList()
        sl.add(_cs())
        data = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        sl.train_df(data)
        assert "x" in sl.map_df(data).columns

    def test_transform_df(self):
        sl = ScalesList()
        sl.add(_cs())
        assert "x" in sl.transform_df(pd.DataFrame({"x": [1.0]})).columns

    def test_clone(self):
        sl = ScalesList()
        sl.add(_cs())
        c = sl.clone()
        assert c is not sl and len(c.scales) == 1

    def test_non_position(self):
        sl = ScalesList()
        sl.add(continuous_scale("colour", palette=lambda x: x, super_class=ScaleContinuous))
        assert sl.non_position_scales().n() == 1

    def test_has_scale(self):
        sl = ScalesList()
        sl.add(_cs())
        assert sl.has_scale("x") and not sl.has_scale("z")

    def test_get_scales(self):
        sl = ScalesList()
        s = _cs()
        sl.add(s)
        assert sl.get_scales("x") is s and sl.get_scales("z") is None

    def test_add_replaces(self):
        sl = ScalesList()
        sl.add(_cs())
        sl.add(_cs())
        assert sl.n() == 1

    def test_train_empty(self):
        ScalesList().train_df(pd.DataFrame())

    def test_map_empty(self):
        assert isinstance(ScalesList().map_df(pd.DataFrame()), pd.DataFrame)


class TestScaleTitles:
    def test_make_title(self):
        assert _cs().make_title(guide_title="G", scale_title=waiver(), label_title=waiver()) == "G"

    def test_axis_order_left(self):
        s = _cs()
        s.position = "left"
        assert s.axis_order() == ["primary", "secondary"]

    def test_axis_order_right(self):
        s = _cs()
        s.position = "right"
        assert s.axis_order() == ["secondary", "primary"]


class TestScaleReset:
    def test_reset(self):
        s = _cs()
        s.train(np.array([0, 10]))
        s.reset()

    def test_is_empty(self):
        assert _cs().is_empty() is True

    def test_not_empty(self):
        s = _cs()
        s.train(np.array([0, 10]))
        assert s.is_empty() is False


class TestFindScale:
    def test_find_x(self):
        assert find_scale("x", pd.Series([1.0, 2.0])) is not None

    def test_scale_type_numeric(self):
        assert isinstance(scale_type(pd.Series([1.0, 2.0])), list)

    def test_scale_type_string(self):
        assert isinstance(scale_type(pd.Series(["a", "b"])), list)

    def test_scale_type_datetime(self):
        assert isinstance(scale_type(pd.Series(pd.to_datetime(["2020-01-01"]))), list)

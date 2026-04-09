"""Additional tests for ggplot2_py.limits."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.limits import xlim, ylim, lims, _limits_numeric, _limits_character, _limits_date, _limits_dispatch


class TestLimitsNumeric:
    def test_basic(self):
        assert _limits_numeric([0, 10], "x") is not None


class TestLimitsCharacter:
    def test_basic(self):
        assert _limits_character(["a", "b"], "x") is not None


class TestLimitsDate:
    def test_wrong_length(self):
        with pytest.raises(Exception):
            _limits_date([pd.Timestamp("2020-01-01")], "x")


class TestLimitsDispatch:
    def test_numeric(self):
        assert _limits_dispatch([0, 10], "x") is not None

    def test_character(self):
        assert _limits_dispatch(["a", "b"], "x") is not None

    def test_ndarray(self):
        assert _limits_dispatch(np.array([0, 10]), "x") is not None

    def test_unknown_type(self):
        with pytest.raises(Exception):
            _limits_dispatch("not_valid", "x")

    def test_empty(self):
        with pytest.raises(Exception):
            _limits_dispatch([], "x")

    def test_all_none(self):
        assert _limits_dispatch([None, None], "x") is not None


class TestXlimYlim:
    def test_xlim(self):
        assert xlim(0, 10) is not None

    def test_ylim(self):
        assert ylim(0, 10) is not None


class TestLims:
    def test_lims_xy(self):
        result = lims(x=[0, 10], y=[0, 5])
        assert isinstance(result, list) and len(result) == 2

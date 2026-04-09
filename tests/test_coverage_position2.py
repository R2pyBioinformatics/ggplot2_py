"""Additional tests for ggplot2_py.position."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.position import (
    PositionIdentity, PositionDodge, PositionDodge2, PositionJitter,
    PositionJitterdodge, PositionNudge, PositionStack, PositionFill,
    is_position, position_identity, position_dodge, position_dodge2,
    position_jitter, position_jitterdodge, position_nudge,
    position_stack, position_fill,
)


class TestPositionNudge:
    def test_compute(self):
        p = position_nudge(x=0.1, y=0.2)
        data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = p.compute_panel(data, {"x": 0.1, "y": 0.2}, {})
        assert isinstance(result, pd.DataFrame)


class TestPositionStack:
    def test_compute(self):
        data = pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 2.0],
            "ymin": [0.0, 0.0], "ymax": [1.0, 2.0], "group": [1, 2]})
        result = PositionStack().compute_panel(data, {"vjust": 1, "reverse": False}, {})
        assert isinstance(result, pd.DataFrame)


class TestPositionFill:
    def test_compute(self):
        data = pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 2.0],
            "ymin": [0.0, 0.0], "ymax": [1.0, 2.0], "group": [1, 2]})
        result = PositionFill().compute_panel(data, {"vjust": 1, "reverse": False}, {})
        assert isinstance(result, pd.DataFrame)


class TestConstructors:
    def test_identity(self):
        assert is_position(position_identity())

    def test_dodge(self):
        assert is_position(position_dodge())

    def test_dodge2(self):
        assert is_position(position_dodge2())

    def test_jitter(self):
        assert is_position(position_jitter())

    def test_jitterdodge(self):
        assert is_position(position_jitterdodge())

    def test_nudge(self):
        assert is_position(position_nudge())

    def test_stack(self):
        assert is_position(position_stack())

    def test_fill(self):
        assert is_position(position_fill())

"""Additional tests for ggplot2_py.annotation."""

import pytest
import numpy as np

from ggplot2_py.annotation import annotation_custom, annotation_raster, annotation_logticks


class TestAnnotationCustom:
    def test_basic(self):
        from grid_py import null_grob
        assert annotation_custom(null_grob()) is not None

    def test_with_bounds(self):
        from grid_py import null_grob
        assert annotation_custom(null_grob(), xmin=0, xmax=1, ymin=0, ymax=1) is not None


class TestAnnotationRaster:
    def test_basic(self):
        assert annotation_raster(np.zeros((10, 10, 3)), 0, 1, 0, 1) is not None

    def test_interpolate(self):
        assert annotation_raster(np.ones((5, 5)), 0, 10, 0, 10, interpolate=True) is not None


class TestAnnotationLogticks:
    def test_basic(self):
        assert annotation_logticks() is not None

    def test_with_color_alias(self):
        assert annotation_logticks(color="blue") is not None

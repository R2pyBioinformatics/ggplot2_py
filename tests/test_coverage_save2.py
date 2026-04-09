"""Additional tests for ggplot2_py.save."""

import pytest
import os
import tempfile
import pandas as pd

from ggplot2_py.save import ggsave, check_device, _parse_dpi, _to_inches


class TestParseDpi:
    def test_numeric(self):
        assert _parse_dpi(300) == 300

    def test_string_retina(self):
        assert _parse_dpi("retina") is not None

    def test_default(self):
        assert _parse_dpi(300) == 300


class TestCheckDevice:
    def test_png(self):
        assert check_device(None, "plot.png") == "png"

    def test_pdf(self):
        assert check_device(None, "plot.pdf") == "pdf"

    def test_explicit(self):
        assert check_device("png", "plot.xxx") == "png"


class TestToInches:
    def test_in(self):
        assert _to_inches(7, "in", 100) == 7

    def test_cm(self):
        assert abs(_to_inches(10, "cm", 100) - 10 / 2.54) < 0.01

    def test_px(self):
        assert _to_inches(300, "px", 100) == 3.0

    def test_none(self):
        assert _to_inches(None, "in", 100) is None


class TestGgsave:
    def test_no_plot_error(self):
        import ggplot2_py.plot as _pm
        old = _pm._last_plot
        _pm._last_plot = None
        try:
            with pytest.raises(Exception):
                ggsave("test.png")
        finally:
            _pm._last_plot = old

    def test_dir_not_exist_error(self):
        with pytest.raises(Exception):
            ggsave("/nonexistent/dir/test.png", plot="dummy", create_dir=False)

    def test_limitsize_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.png")
            from ggplot2_py.plot import ggplot
            from ggplot2_py.aes import aes
            p = ggplot(pd.DataFrame({"x": [1.0], "y": [2.0]}), aes(x="x", y="y"))
            with pytest.raises(Exception):
                ggsave(filepath, plot=p, width=100, height=100, units="in", limitsize=True)

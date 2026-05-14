"""Tests for ``ggplot_defaults`` — Python-exclusive scoped-default context manager.

R has no equivalent feature, so the principle is:

* **No-context case must byte-match R** (see also
  ``validation/validate_ggplot_defaults.py`` which talks to R directly).
* **Context-active case** is the Python-only spec: each ctx kwarg overlays the
  intrinsic default; ``coord`` is marked ``default=True`` on the copy so a
  later ``+ coord_X()`` replaces it silently (parity with R
  ``update_ggplot.Coord`` behaviour on default coords).
* **Caller-supplied ctx instances must not be mutated** by ``ggplot_defaults``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ggplot2_py import (
    ggplot,
    aes,
    ggplot_defaults,
    theme_minimal,
    facet_wrap,
    coord_polar,
)
from ggplot2_py.coord import coord_fixed, CoordCartesian, CoordFixed, CoordPolar
from ggplot2_py.facet import FacetNull, FacetWrap


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "g": ["a", "a", "b"]})


# ---------------------------------------------------------------------------
# No-context baseline — must match R byte-for-byte
# ---------------------------------------------------------------------------

class TestNoContextBaseline:
    """Mirror of ``validation/validate_ggplot_defaults.py`` — R-side values
    are encoded as the expected literals here; the actual R/Python diff is
    asserted by the validation script in CI."""

    def test_coord_is_cartesian_default(self, df):
        p = ggplot(df, aes("x", "y"))
        assert type(p.coordinates).__name__ == "CoordCartesian"
        assert p.coordinates.default is True

    def test_facet_is_null(self, df):
        p = ggplot(df, aes("x", "y"))
        assert type(p.facet).__name__ == "FacetNull"

    def test_theme_empty(self, df):
        p = ggplot(df, aes("x", "y"))
        assert len(p.theme) == 0

    def test_mapping_from_aes_only(self, df):
        p = ggplot(df, aes("x", "y"))
        assert list(p.mapping.keys()) == ["x", "y"]


# ---------------------------------------------------------------------------
# Context-active: each kwarg overlays the intrinsic default
# ---------------------------------------------------------------------------

class TestContextOverlay:
    def test_theme_path(self, df):
        with ggplot_defaults(theme=theme_minimal()):
            p = ggplot(df, aes("x", "y"))
        assert len(p.theme) > 0
        assert p.theme.complete is True

    def test_coord_path(self, df):
        with ggplot_defaults(coord=coord_fixed()):
            p = ggplot(df, aes("x", "y"))
        assert isinstance(p.coordinates, CoordFixed)
        # Soft-default flag is set on the copy so later + coord_X() is silent.
        assert p.coordinates.default is True

    def test_facet_path(self, df):
        with ggplot_defaults(facet=facet_wrap("g")):
            p = ggplot(df, aes("x", "y"))
        assert isinstance(p.facet, FacetWrap)

    def test_mapping_path_ctx_under_explicit(self, df):
        # ctx provides x and colour; explicit aes provides y and (implicitly)
        # nothing for x — the explicit mapping should win where it overlaps,
        # ctx fills the rest.
        with ggplot_defaults(mapping=aes(x="x", colour="g")):
            p = ggplot(df, aes(y="y"))
        m = dict(p.mapping)
        assert m["y"] == "y"
        assert m["x"] == "x"
        assert m["colour"] == "g"


# ---------------------------------------------------------------------------
# Outside-the-block restoration
# ---------------------------------------------------------------------------

class TestContextScope:
    def test_outside_block_no_effect(self, df):
        with ggplot_defaults(coord=coord_fixed(), facet=facet_wrap("g")):
            pass  # context exits, defaults must roll back
        p = ggplot(df, aes("x", "y"))
        assert isinstance(p.coordinates, CoordCartesian)
        assert type(p.facet).__name__ == "FacetNull"


# ---------------------------------------------------------------------------
# Subsequent ``+ coord_X()`` after ctx-supplied coord
# ---------------------------------------------------------------------------

class TestSilentReplaceAfterCtxCoord:
    """Mirrors R ``update_ggplot.Coord`` (plot-construction.R:202-215): when
    the existing coord is marked default, replacement is silent (no
    ``cli_inform``).  The ctx-supplied coord carries ``default=True`` for
    exactly this reason."""

    def test_plus_coord_replaces_silently(self, df, capsys):
        with ggplot_defaults(coord=coord_fixed()):
            p = ggplot(df, aes("x", "y")) + coord_polar()
        assert isinstance(p.coordinates, CoordPolar)
        captured = capsys.readouterr()
        assert "already present" not in (captured.out + captured.err).lower()


# ---------------------------------------------------------------------------
# Caller-supplied instances must not be mutated
# ---------------------------------------------------------------------------

class TestContextVarThreadIsolation:
    """README claim: ``ggplot_defaults`` uses ``contextvars.ContextVar`` for
    thread-safe scoped defaults.  Pin this so the claim is verified, not
    aspirational."""

    def test_other_thread_sees_no_defaults(self, df):
        import threading
        from ggplot2_py import theme_minimal

        results = {}
        barrier = threading.Barrier(2)

        def in_thread_with_ctx():
            barrier.wait()
            with ggplot_defaults(theme=theme_minimal()):
                # Hold inside the context so the other thread definitely
                # executes ``ggplot()`` during this window.
                import time
                time.sleep(0.05)
                p = ggplot(df, aes("x", "y"))
            results["inside"] = len(p.theme)

        def in_thread_without_ctx():
            barrier.wait()
            import time
            time.sleep(0.01)
            p = ggplot(df, aes("x", "y"))
            results["outside"] = len(p.theme)

        t1 = threading.Thread(target=in_thread_with_ctx)
        t2 = threading.Thread(target=in_thread_without_ctx)
        t1.start(); t2.start(); t1.join(); t2.join()

        assert results["inside"] > 0
        assert results["outside"] == 0, (
            "ContextVar leaked across threads — README's thread-safety "
            "claim violated."
        )


class TestNoMutationOfCtxInstances:
    def test_shared_coord_instance_not_mutated(self, df):
        shared = coord_fixed()
        assert shared.default is False
        with ggplot_defaults(coord=shared):
            ggplot(df, aes("x", "y"))
        assert shared.default is False, (
            "ggplot_defaults must not mutate the user-supplied coord; "
            "use copy.copy before applying the soft-default flag."
        )

    def test_shared_facet_instance_not_mutated(self, df):
        shared = facet_wrap("g")
        original_class = type(shared).__name__
        with ggplot_defaults(facet=shared):
            ggplot(df, aes("x", "y"))
        # Class identity / type unchanged is the basic invariant.
        assert type(shared).__name__ == original_class

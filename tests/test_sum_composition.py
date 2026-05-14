"""Functional / iterable-based composition styles.

Because :meth:`GGPlot.__add__` and :meth:`GGPlot.__radd__` are defined and the
``update_ggplot`` singledispatch generic handles every component family,
**Python's standard iterable-composition idioms compose a ggplot the same way
the ``+`` operator does** — without ``ggplot2_py`` shipping any extra helper.

This file pins that contract so it cannot regress.  The reference for the
``fnplot(data, *args)`` style is @mikedecr.computer's Bluesky post
(2026-04-29):

    def fnplot(data, *args):
        return sum(args, start=ggplot(data))

Every variant in :data:`WORKING_VARIANTS` is asserted to produce a build-
equivalent plot to the canonical ``+``-chain version; :data:`FAILING_VARIANTS`
documents two forms that look plausible but are not valid Python (the
``sum`` built-in signature is ``sum(iterable, /, start=0)``).
"""
from __future__ import annotations

import functools
import operator

import pandas as pd
import pytest

from ggplot2_py import (
    GGPlot,
    aes,
    facet_wrap,
    geom_point,
    geom_smooth,
    ggplot,
    ggplot_build,
    labs,
    theme_minimal,
)


# ---------------------------------------------------------------------------
# Fixtures: a fixed plot the canonical and functional builds must both produce
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [4, 5, 6, 5, 7, 8],
        "g": ["a", "a", "a", "b", "b", "b"],
    })


def _components():
    """Plot components as a fresh tuple — each call returns new objects so no
    Layer is shared across variants."""
    return (
        aes(x="x", y="y", colour="g"),
        geom_point(),
        geom_smooth(method="lm"),
        facet_wrap("g"),
        theme_minimal(),
        labs(title="sum() composition test"),
    )


def _signature(p: GGPlot) -> dict:
    """Compact fingerprint two builds must share to count as equivalent."""
    return {
        "type":      type(p).__name__,
        "n_layers":  len(p.layers),
        "mapping":   dict(p.mapping),
        "theme_len": len(p.theme),
        "labels":    dict(p.labels),
        "facet":     type(p.facet).__name__,
        "coord":     type(p.coordinates).__name__,
    }


@pytest.fixture
def canonical_signature(df):
    """Signature of ``ggplot(df, aes) + geom + geom + facet + theme + labs``."""
    p = ggplot(df, _components()[0])
    for c in _components()[1:]:
        p = p + c
    assert ggplot_build(p) is not None, "canonical + chain must build"
    return _signature(p)


# ---------------------------------------------------------------------------
# Working variants — each builds a plot equivalent to the canonical + chain
# ---------------------------------------------------------------------------

class TestWorkingSumCompositionStyles:
    """All of these must round-trip to the same plot signature and a
    successful ``ggplot_build`` call."""

    def test_v1_fnplot_helper(self, df, canonical_signature):
        """mikedecr's exact helper, verbatim from the Bluesky post."""
        def fnplot(data, *args):
            return sum(args, start=ggplot(data))

        p = fnplot(df, *_components())
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v2_sum_list_default_start(self, df, canonical_signature):
        """``sum([ggplot(d, m), comp, comp, ...])`` relies on
        ``__radd__(0)`` returning *self*, which is the contract used by
        Python's built-in ``sum`` when no ``start=`` is supplied."""
        seed = ggplot(df, _components()[0])
        p = sum([seed, *_components()[1:]])
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v3_sum_list_explicit_start(self, df, canonical_signature):
        p = sum(list(_components()), start=ggplot(df))
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v4_sum_tuple_iterable(self, df, canonical_signature):
        p = sum(tuple(_components()), start=ggplot(df))
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v5_sum_generator_iterable(self, df, canonical_signature):
        p = sum((c for c in _components()), start=ggplot(df))
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v6_sum_empty_iterable_returns_start(self, df):
        """Degenerate edge case — empty iterable must return the seed."""
        seed = ggplot(df)
        p = sum([], start=seed)
        assert isinstance(p, GGPlot)

    def test_v7_nested_list_inside_sum(self, df, canonical_signature):
        """Nested lists exercise the recursive ``update_ggplot.register(list)``
        path — every layer still ends up in the plot."""
        c = _components()
        seed = ggplot(df, c[0])
        nested = [c[1], c[2]]               # geom_point, geom_smooth
        rest = c[3:]                        # facet, theme, labs
        p = sum([seed, nested, *rest])
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v8_reduce_add_no_initial(self, df, canonical_signature):
        c = _components()
        parts = [ggplot(df, c[0]), *c[1:]]
        p = functools.reduce(operator.add, parts)
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v9_reduce_add_with_initial(self, df, canonical_signature):
        p = functools.reduce(operator.add, _components(), ggplot(df))
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None

    def test_v10_sum_with_genexp_inline(self, df, canonical_signature):
        """Same as V5 but the generator expression is inline — common
        list/dict-comprehension style."""
        p = sum((c for c in _components()), start=ggplot(df))
        assert _signature(p) == canonical_signature
        assert ggplot_build(p) is not None


# ---------------------------------------------------------------------------
# Failing variants — call out the two shapes that look plausible but aren't
# valid Python.  Pinning them ensures we don't accidentally start "supporting"
# them by shadowing ``sum`` (which would break ``from ggplot2_py import *``).
# ---------------------------------------------------------------------------

class TestNonWorkingFormsAreStillNonWorking:

    def test_sum_with_more_than_2_positional_args_raises(self, df):
        """``sum(a, b, c, d)`` looks tempting but Python's ``sum`` signature
        is ``sum(iterable, /, start=0)`` — only two args.  This documents
        the limitation so we never silently "fix" it by shadowing ``sum``."""
        with pytest.raises(TypeError, match=r"sum\(\) takes at most 2 arguments"):
            sum(ggplot(df), geom_point(), geom_smooth(), theme_minimal())  # noqa: B026

    def test_sum_with_iterable_as_keyword_raises(self, df):
        """``iterable`` is positional-only on ``sum``."""
        with pytest.raises(TypeError):
            # The "iterable" kwarg cannot be passed by name because it's
            # positional-only — passing it as keyword leaves the required
            # positional arg unfilled.
            sum(iterable=[ggplot(df), geom_point()])  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Rendered-output equivalence (byte-level)
# ---------------------------------------------------------------------------

class TestRenderedOutputByteEquivalent:
    """Compose the same plot four different ways and verify the rendered
    PNGs are bit-for-bit identical.  This is the strongest possible
    equivalence assertion."""

    def test_plus_fn_sum_reduce_render_identically(self, df, tmp_path):
        import hashlib
        from ggplot2_py import ggsave

        # 1. canonical +
        p_plus = ggplot(df, _components()[0])
        for c in _components()[1:]:
            p_plus = p_plus + c

        # 2. mikedecr helper
        def fnplot(data, *args):
            return sum(args, start=ggplot(data))
        p_fn = fnplot(df, *_components())

        # 3. sum(list, start=...)
        p_sum = sum(list(_components()), start=ggplot(df))

        # 4. reduce(add, ..., initial=...)
        p_red = functools.reduce(operator.add, _components(), ggplot(df))

        def hash_png(p, name):
            path = tmp_path / f"{name}.png"
            ggsave(str(path), plot=p, width=5, height=4, dpi=100)
            return hashlib.sha256(path.read_bytes()).hexdigest()

        h_plus = hash_png(p_plus, "plus")
        h_fn   = hash_png(p_fn,   "fnplot")
        h_sum  = hash_png(p_sum,  "sum")
        h_red  = hash_png(p_red,  "reduce")

        assert h_plus == h_fn == h_sum == h_red, (
            f"Rendered PNGs diverge:\n"
            f"  +     : {h_plus}\n"
            f"  fnplot: {h_fn}\n"
            f"  sum   : {h_sum}\n"
            f"  reduce: {h_red}"
        )

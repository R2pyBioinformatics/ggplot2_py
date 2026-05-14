"""Contract tests for ``ggplot2_py.protocols``.

The Protocol classes in ``ggplot2_py/protocols.py`` are the structural-typing
contract for every Geom / Stat / Scale / Coord / Facet / Position subclass.
This test sweeps every **shipped** subclass and asserts it satisfies its
Protocol via ``isinstance`` (the ``@runtime_checkable`` decorator's job).

Why this test exists
--------------------
Without it, the Protocols would silently drift from the base classes —
which is exactly what was happening before the Block-C refit (e.g.
``CoordProtocol.setup_params(self, data: list)`` had ``list`` annotated
when the base class took ``Any``).  The Protocols become live spec only
if a CI gauntlet keeps them honest.

``runtime_checkable`` Protocol checks attribute / method **existence and
callability**, not full signature compatibility — strict signature checking
belongs to static type checkers (mypy / pyright), not runtime.
"""
from __future__ import annotations

import importlib
from typing import Type

import pytest

from ggplot2_py.protocols import (
    GeomProtocol,
    StatProtocol,
    ScaleProtocol,
    CoordProtocol,
    FacetProtocol,
    PositionProtocol,
)
from ggplot2_py.geom import Geom
from ggplot2_py.stat import Stat
from ggplot2_py.scale import Scale
from ggplot2_py.coord import Coord
from ggplot2_py.facet import Facet
from ggplot2_py.position import Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry_subclasses(base_cls: Type) -> list:
    """Return distinct subclass values from a ``_registry`` (deduplicated)."""
    seen = set()
    out = []
    for v in getattr(base_cls, "_registry", {}).values():
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _discover_subclasses(base_cls: Type) -> list:
    """Best-effort discovery for classes without an auto-registry (Scale,
    Coord, Facet).  Walks ``__subclasses__`` transitively after importing
    the package modules that define the classes."""
    # Trigger imports of all modules that define subclasses.
    for mod in ("ggplot2_py.scale", "ggplot2_py.coord", "ggplot2_py.facet"):
        importlib.import_module(mod)
    seen = set()
    stack = list(base_cls.__subclasses__())
    out: list = []
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


def _instance_or_skip(cls):
    """Instantiate *cls* with no args; some classes need args we can't supply
    generically — in that case we ``pytest.skip`` for THAT class only."""
    try:
        return cls()
    except TypeError as e:
        pytest.skip(f"{cls.__name__} requires constructor args: {e}")


# ---------------------------------------------------------------------------
# Contract sweeps — one parametrised test per component family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _registry_subclasses(Geom), ids=lambda c: c.__name__)
def test_every_shipped_geom_satisfies_geomprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, GeomProtocol), (
        f"{cls.__name__} fails GeomProtocol — check required_aes / "
        f"default_aes / setup_params / setup_data / draw_panel / draw_key."
    )


@pytest.mark.parametrize("cls", _registry_subclasses(Stat), ids=lambda c: c.__name__)
def test_every_shipped_stat_satisfies_statprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, StatProtocol), (
        f"{cls.__name__} fails StatProtocol — check required_aes / "
        f"default_aes / setup_params / setup_data / compute_group."
    )


@pytest.mark.parametrize("cls", _registry_subclasses(Position), ids=lambda c: c.__name__)
def test_every_shipped_position_satisfies_positionprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, PositionProtocol), (
        f"{cls.__name__} fails PositionProtocol — check setup_params / "
        f"compute_layer."
    )


@pytest.mark.parametrize("cls", _discover_subclasses(Scale), ids=lambda c: c.__name__)
def test_every_shipped_scale_satisfies_scaleprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, ScaleProtocol), (
        f"{cls.__name__} fails ScaleProtocol — check aesthetics / train / "
        f"transform / map / get_breaks / get_labels / clone."
    )


@pytest.mark.parametrize("cls", _discover_subclasses(Coord), ids=lambda c: c.__name__)
def test_every_shipped_coord_satisfies_coordprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, CoordProtocol), (
        f"{cls.__name__} fails CoordProtocol — check setup_params / "
        f"transform / setup_panel_params."
    )


@pytest.mark.parametrize("cls", _discover_subclasses(Facet), ids=lambda c: c.__name__)
def test_every_shipped_facet_satisfies_facetprotocol(cls):
    inst = _instance_or_skip(cls)
    assert isinstance(inst, FacetProtocol), (
        f"{cls.__name__} fails FacetProtocol — check compute_layout / map_data."
    )


# ---------------------------------------------------------------------------
# Negative tests — Protocols reject obvious imposters
# ---------------------------------------------------------------------------

class TestProtocolsRejectImposters:
    """Make sure ``isinstance(_, XxxProtocol)`` is not vacuously true."""

    def test_int_is_not_geomprotocol(self):
        assert not isinstance(42, GeomProtocol)

    def test_dataframe_is_not_statprotocol(self):
        import pandas as pd
        assert not isinstance(pd.DataFrame(), StatProtocol)

    def test_object_missing_clone_is_not_scaleprotocol(self):
        class Partial:
            aesthetics = ["x"]
            def train(self, x): pass
            def transform(self, x): return x
            def map(self, x, limits=None): return x
            def get_breaks(self, limits=None): return []
            def get_labels(self, breaks=None): return []
            # clone missing
        assert not isinstance(Partial(), ScaleProtocol)

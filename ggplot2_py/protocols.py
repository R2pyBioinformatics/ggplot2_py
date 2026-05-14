"""
Structural typing protocols for ggplot2_py GOG components.

These :class:`~typing.Protocol` definitions are the **machine-readable
contract** that every Geom, Stat, Scale, Coord, Facet, and Position class —
shipped or user-supplied — must satisfy.

What this module is **not**
---------------------------
* It is not a replacement for the base classes (``Geom``, ``Stat``, etc.).
  Internal code dispatches off the base classes; the Protocols are an
  orthogonal, structural view.
* It does **not** make the base classes interchangeable with arbitrary
  duck-typed objects in the build pipeline — the singledispatch + auto-
  registration machinery still keys on the concrete base classes.

What this module **is**
-----------------------
1. **Static-typing aid.**  Extension authors writing ``MyStat(Stat)`` get
   mypy / pyright errors when they omit ``compute_group``, return the
   wrong type, etc.  The signatures here mirror the base classes
   verbatim — R's ``ggproto`` field signatures, transitively.
2. **Live contract test.**  ``tests/test_protocols_contract.py`` iterates
   every shipped subclass and asserts ``isinstance(instance, XxxProtocol)``.
   Signature drift in a base class — or accidental method removal in a
   subclass — fails CI immediately.  Without that test these Protocols
   would just be documentation; with it they are an executable spec.

R parity
--------
R has no Protocol mechanism (``ggproto`` is structural duck-typing all the
way down), so the Protocols themselves are Python-exclusive.  However each
method signature here is **derived from the corresponding R ``ggproto``
field** by going through the Python base class (``Geom`` / ``Stat`` / …),
which was ported from the R prototype.  So the contract is transitively
R-aligned.

Usage
-----
Static (mypy / pyright)::

    class MyStat(Stat):
        required_aes = ("x",)
        # Forgetting compute_group is a static type error.

Runtime (rare; the contract test in ``tests/`` already covers shipped
classes — extension authors needing dynamic checks can do)::

    from ggplot2_py.protocols import StatProtocol
    if not isinstance(my_stat, StatProtocol):
        raise TypeError("Stat extension is missing required methods")
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import pandas as pd

__all__ = [
    "GeomProtocol",
    "StatProtocol",
    "ScaleProtocol",
    "CoordProtocol",
    "FacetProtocol",
    "PositionProtocol",
]


# ---------------------------------------------------------------------------
# Geom — signatures from ``ggplot2_py.geom.Geom``
# ---------------------------------------------------------------------------

@runtime_checkable
class GeomProtocol(Protocol):
    """Contract for geometry objects.  Mirrors ``Geom`` (geom.py:462)."""

    required_aes: Union[Tuple[str, ...], List[str]]
    default_aes: Any            # Mapping or dict
    draw_key: Any               # callable (class-level attribute)

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]: ...
    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame: ...
    def draw_panel(self, *args: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Stat — signatures from ``ggplot2_py.stat.Stat``
# ---------------------------------------------------------------------------

@runtime_checkable
class StatProtocol(Protocol):
    """Contract for statistical transformation objects.  Mirrors ``Stat``
    (stat.py base class).

    The Protocol pins the three methods most likely to be user-overridden;
    a Stat may also override ``compute_layer`` or ``compute_panel`` instead
    of (or in addition to) ``compute_group``.
    """

    required_aes: Union[Tuple[str, ...], List[str]]
    default_aes: Any

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]: ...
    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame: ...
    def compute_group(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Scale — signatures from ``ggplot2_py.scale.Scale``
#
# The base class accepts optional ``limits`` on ``map`` / ``get_breaks`` /
# ``get_labels``.  The Protocol matches those signatures verbatim — narrower
# signatures (omitting the optional kwargs) caused the original drift.
# ---------------------------------------------------------------------------

@runtime_checkable
class ScaleProtocol(Protocol):
    """Contract for scale objects.  Mirrors ``Scale`` (scale.py:410)."""

    aesthetics: Any             # list of str

    def train(self, x: Any) -> None: ...
    def transform(self, x: Any) -> Any: ...
    def map(self, x: Any, limits: Optional[Any] = ...) -> Any: ...
    def get_breaks(self, limits: Optional[Any] = ...) -> Any: ...
    def get_labels(self, breaks: Optional[Any] = ...) -> Any: ...
    def clone(self) -> Any: ...


# ---------------------------------------------------------------------------
# Coord — signatures from ``ggplot2_py.coord.Coord``
# ---------------------------------------------------------------------------

@runtime_checkable
class CoordProtocol(Protocol):
    """Contract for coordinate system objects.  Mirrors ``Coord``
    (coord.py:494).

    ``setup_params`` takes ``Any`` (not ``list``) because the call sites
    pass per-layer data lists, single DataFrames, or panel-level dicts
    depending on the Coord subclass.
    """

    def setup_params(self, data: Any) -> Dict[str, Any]: ...
    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame: ...
    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = ...,
    ) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Facet — signatures from ``ggplot2_py.facet.Facet``
# ---------------------------------------------------------------------------

@runtime_checkable
class FacetProtocol(Protocol):
    """Contract for faceting specification objects.  Mirrors ``Facet``
    (facet.py:490)."""

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame: ...
    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Position — signatures from ``ggplot2_py.position.Position``
# ---------------------------------------------------------------------------

@runtime_checkable
class PositionProtocol(Protocol):
    """Contract for position adjustment objects.  Mirrors ``Position``
    (position.py:200)."""

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]: ...
    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
    ) -> pd.DataFrame: ...

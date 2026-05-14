"""Tests for pipeline dispatch extensibility and build hooks — Part B+C
of the GOG pipeline refactoring.

Covers:
- update_ggplot singledispatch (register custom types)
- ggplot_build singledispatch
- ggplot_gtable singledispatch
- by_layer helper
- BuildStage constants
- Build hook system
"""

import pandas as pd
import pytest

from ggplot2_py import (
    ggplot,
    aes,
    geom_point,
    ggplot_build,
    ggplot_add,
    BuildStage,
)
from ggplot2_py.plot import (
    update_ggplot,
    by_layer,
    GGPlot,
    BuiltGGPlot,
)


# -----------------------------------------------------------------------
# B1: update_ggplot singledispatch
# -----------------------------------------------------------------------

class TestUpdateGgplotDispatch:
    def test_layer_dispatched(self):
        """Layer objects should be added via singledispatch."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        assert len(p.layers) == 1

    def test_none_dispatched(self):
        """None should be a no-op."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = p + None
        assert len(p2.layers) == 0

    def test_list_dispatched(self):
        """Lists should be recursively added."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = p + [geom_point(), geom_point()]
        assert len(p2.layers) == 2

    def test_dataframe_dispatched(self):
        """DataFrames should replace default data."""
        df1 = pd.DataFrame({"x": [1], "y": [1]})
        df2 = pd.DataFrame({"x": [2, 3], "y": [4, 5]})
        p = ggplot(df1, aes(x="x", y="y"))
        p2 = p + df2
        assert len(p2.data) == 2

    def test_register_custom_type(self):
        """Extension packages should be able to register custom types."""

        class MyAnnotation:
            def __init__(self, text):
                self.text = text

        @update_ggplot.register(MyAnnotation)
        def _add_annotation(obj, plot, object_name=""):
            plot._meta["custom_annotation"] = obj.text
            return plot

        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = ggplot_add(MyAnnotation("hello"), p)
        assert p2._meta.get("custom_annotation") == "hello"

    def test_unknown_type_raises(self):
        """Unknown types should raise TypeError."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        with pytest.raises(Exception):
            p + 42  # int has no registered handler


# -----------------------------------------------------------------------
# B2: ggplot_build singledispatch
# -----------------------------------------------------------------------

class TestGgplotBuildDispatch:
    def test_ggplot_type(self):
        """GGPlot objects should build normally."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        built = ggplot_build(p)
        assert isinstance(built, BuiltGGPlot)

    def test_built_noop(self):
        """Already-built plots should be returned as-is."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        built = ggplot_build(p)
        built2 = ggplot_build(built)
        assert built is built2

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Cannot build"):
            ggplot_build("not a plot")

    def test_registry_has_ggplot(self):
        assert GGPlot in ggplot_build.registry

    def test_registry_has_built(self):
        assert BuiltGGPlot in ggplot_build.registry


# -----------------------------------------------------------------------
# B4: by_layer helper
# -----------------------------------------------------------------------

class TestByLayer:
    def test_basic(self):
        layers = ["a", "b", "c"]
        data = [1, 2, 3]
        result = by_layer(lambda l, d: f"{l}{d}", layers, data, "test")
        assert result == ["a1", "b2", "c3"]

    def test_error_context(self):
        layers = ["ok", "fail"]
        data = [1, 2]

        def fn(l, d):
            if l == "fail":
                raise ValueError("boom")
            return d

        with pytest.raises(RuntimeError, match="layer 2"):
            by_layer(fn, layers, data, "testing")


# -----------------------------------------------------------------------
# C: BuildStage + hooks
# -----------------------------------------------------------------------

class TestBuildStage:
    def test_constants_are_strings(self):
        assert isinstance(BuildStage.COMPUTE_STAT, str)
        assert BuildStage.COMPUTE_STAT == "compute_stat"
        assert BuildStage.LAYER_DATA == "layer_data"
        assert BuildStage.FINISH_DATA == "finish_data"


class TestBuildHooks:
    def test_hook_fires_after_compute_stat(self):
        log = []
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        p.add_build_hook(
            "after",
            BuildStage.COMPUTE_STAT,
            lambda data: (log.append("fired"), data)[-1],
        )
        ggplot_build(p)
        assert "fired" in log

    def test_hook_fires_before_compute_aesthetics(self):
        log = []
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        p.add_build_hook(
            "before",
            BuildStage.COMPUTE_AESTHETICS,
            lambda data: (log.append(len(data)), data)[-1],
        )
        ggplot_build(p)
        assert log == [1]  # 1 layer

    def test_hook_can_modify_data(self):
        """Hook that returns new data list replaces the pipeline data."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()

        def double_y(data):
            new_data = []
            for d in data:
                d = d.copy()
                if "y" in d.columns:
                    d["y"] = d["y"] * 2
                new_data.append(d)
            return new_data

        p.add_build_hook("after", BuildStage.COMPUTE_AESTHETICS, double_y)
        built = ggplot_build(p)
        # y values should be doubled (8, 10, 12 instead of 4, 5, 6)
        y_vals = built.data[0]["y"].values
        assert y_vals[0] == 8.0 or y_vals[0] == 8

    def test_invalid_timing_raises(self):
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        with pytest.raises(ValueError, match="before.*after"):
            p.add_build_hook("during", BuildStage.COMPUTE_STAT, lambda d: d)

    def test_chaining(self):
        """add_build_hook returns self for chaining."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        result = p.add_build_hook("after", BuildStage.FINISH_DATA, lambda d: d)
        assert result is p


# -----------------------------------------------------------------------
# C2: Every BuildStage constant actually fires in ggplot_build
# (R parity: each stage corresponds to a real operation in plot-build.R)
# -----------------------------------------------------------------------

class TestAllBuildStagesFire:
    """Each of the 16 ``BuildStage`` constants corresponds to a real stage
    in R's ``plot-build.R``.  Before the Block-A fix, 6 of them were
    "zombie" constants with no ``_h(...)`` wrapper.  This test pins every
    stage's before+after to fire on a minimal plot."""

    # COMPUTE_GEOM_2 needs a stat to produce something; we choose a
    # non-position aesthetic so TRAIN_NONPOSITION also fires.
    def _build_plot(self):
        from ggplot2_py import geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "a", "b"]})
        return ggplot(df, aes(x="x", y="y", colour="c")) + geom_point()

    @pytest.mark.parametrize("stage", [
        BuildStage.LAYER_DATA,
        BuildStage.SETUP_LAYER,
        BuildStage.SETUP_LAYOUT,
        BuildStage.COMPUTE_AESTHETICS,
        BuildStage.TRANSFORM_SCALES,
        BuildStage.TRAIN_POSITION,
        BuildStage.COMPUTE_STAT,
        BuildStage.MAP_STAT,
        BuildStage.COMPUTE_GEOM_1,
        BuildStage.COMPUTE_POSITION,
        BuildStage.RETRAIN_POSITION,
        BuildStage.SETUP_GUIDES,
        BuildStage.TRAIN_NONPOSITION,
        BuildStage.COMPUTE_GEOM_2,
        BuildStage.FINISH_STAT,
        BuildStage.FINISH_DATA,
    ])
    def test_stage_fires_before_and_after(self, stage):
        log = []
        p = self._build_plot()
        p.add_build_hook("before", stage, lambda data: log.append(("before", stage)))
        p.add_build_hook("after",  stage, lambda data: log.append(("after",  stage)))
        ggplot_build(p)
        assert ("before", stage) in log, f"BuildStage.{stage} 'before' did not fire"
        assert ("after",  stage) in log, f"BuildStage.{stage} 'after'  did not fire"


# -----------------------------------------------------------------------
# C3: Per-stage ctx kwargs are forwarded as documented
# -----------------------------------------------------------------------

class TestBuildHookCtxForwarding:
    """The BuildStage docstring's per-stage ctx table is the contract.  This
    asserts each ctx-bearing stage actually delivers the promised kwargs."""

    def _plot(self):
        from ggplot2_py import geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "a", "b"]})
        return ggplot(df, aes(x="x", y="y", colour="c")) + geom_point()

    def _captured(self, stage, capture):
        """Register a `**kw`-style hook, build, return captured kwargs dict."""
        p = self._plot()
        p.add_build_hook("after", stage, lambda data, **kw: capture.update(kw))
        ggplot_build(p)
        return capture

    def test_setup_layout_carries_layout(self):
        cap: dict = {}
        self._captured(BuildStage.SETUP_LAYOUT, cap)
        assert "layout" in cap and cap["layout"] is not None

    def test_transform_scales_carries_scales(self):
        cap: dict = {}
        self._captured(BuildStage.TRANSFORM_SCALES, cap)
        assert "scales" in cap and cap["scales"] is not None

    def test_train_position_carries_layout_and_scales(self):
        cap: dict = {}
        self._captured(BuildStage.TRAIN_POSITION, cap)
        assert {"layout", "scales"}.issubset(cap.keys())

    def test_compute_stat_carries_layout(self):
        cap: dict = {}
        self._captured(BuildStage.COMPUTE_STAT, cap)
        assert "layout" in cap

    def test_compute_position_carries_layout(self):
        cap: dict = {}
        self._captured(BuildStage.COMPUTE_POSITION, cap)
        assert "layout" in cap

    def test_retrain_position_carries_layout_and_scales(self):
        cap: dict = {}
        self._captured(BuildStage.RETRAIN_POSITION, cap)
        assert {"layout", "scales"}.issubset(cap.keys())

    def test_setup_guides_carries_layout_and_guides(self):
        cap: dict = {}
        self._captured(BuildStage.SETUP_GUIDES, cap)
        assert "layout" in cap and "guides" in cap

    def test_train_nonposition_carries_scales(self):
        cap: dict = {}
        self._captured(BuildStage.TRAIN_NONPOSITION, cap)
        assert "scales" in cap and cap["scales"] is not None

    def test_compute_geom_2_carries_theme(self):
        cap: dict = {}
        self._captured(BuildStage.COMPUTE_GEOM_2, cap)
        assert "theme" in cap


# -----------------------------------------------------------------------
# C4: Hook signature introspection — three styles must coexist
# -----------------------------------------------------------------------

class TestHookSignatureIntrospection:
    """:func:`_run_hooks` uses ``inspect.signature`` to forward only the
    kwargs each hook can accept.  Verify all three styles work for the
    same stage without TypeErrors."""

    def test_three_styles_coexist(self):
        from ggplot2_py import geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "a", "b"]})
        p = ggplot(df, aes(x="x", y="y", colour="c")) + geom_point()

        log = {}
        # data-only (no kwargs)
        p.add_build_hook("after", BuildStage.TRAIN_POSITION,
                         lambda data: log.setdefault("data_only", True))
        # **kw catch-all
        p.add_build_hook("after", BuildStage.TRAIN_POSITION,
                         lambda data, **kw: log.update({"kw": sorted(kw.keys())}))
        # named kwarg selecting one ctx field
        p.add_build_hook("after", BuildStage.TRAIN_POSITION,
                         lambda data, layout=None: log.update({"named_layout": layout is not None}))
        ggplot_build(p)
        assert log["data_only"] is True
        assert log["kw"] == ["layout", "scales"]
        assert log["named_layout"] is True

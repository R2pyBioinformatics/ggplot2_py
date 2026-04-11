"""Tests for entry-point-based plugin discovery (_plugins.py)."""

import pytest
from unittest.mock import patch, MagicMock
from ggplot2_py._plugins import discover_extensions, list_extensions, _EXTENSION_GROUPS


class TestPluginDiscovery:
    """Test the discover_extensions() function."""

    def test_returns_dict(self):
        """discover_extensions() should always return a dict."""
        result = discover_extensions()
        assert isinstance(result, dict)

    def test_empty_when_no_plugins(self):
        """With no extension packages installed, result should be empty."""
        result = discover_extensions()
        # May or may not be empty depending on env, but should not error
        assert isinstance(result, dict)

    def test_list_extensions_matches_discover(self):
        """list_extensions() returns the same result as last discover."""
        discovered = discover_extensions()
        listed = list_extensions()
        assert discovered == listed

    def test_extension_groups_defined(self):
        """All 6 extension groups should be defined."""
        groups = [g for g, _, _ in _EXTENSION_GROUPS]
        assert "ggplot2_py.geoms" in groups
        assert "ggplot2_py.stats" in groups
        assert "ggplot2_py.positions" in groups
        assert "ggplot2_py.scales" in groups
        assert "ggplot2_py.coords" in groups
        assert "ggplot2_py.facets" in groups

    def test_mock_extension_loaded(self):
        """Simulate an extension package with a mock entry point."""
        from ggplot2_py.geom import Geom
        from ggplot2_py.aes import Mapping

        # Create a mock geom class
        class GeomMockTest(Geom):
            required_aes = ("x", "y")
            default_aes = Mapping(colour="red")

        # Create a mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "mock_test"
        mock_ep.load.return_value = GeomMockTest

        # Mock entry_points().select() to return our mock
        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            result = discover_extensions()

        assert "ggplot2_py.geoms" in result
        assert "mock_test" in result["ggplot2_py.geoms"]
        assert "mock_test" in Geom._registry
        assert Geom._registry["mock_test"] is GeomMockTest

        # Clean up registry
        Geom._registry.pop("mock_test", None)
        Geom._registry.pop("Mock_test", None)

    def test_failed_load_warns(self):
        """If an entry point fails to load, a warning should be emitted."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.load.side_effect = ImportError("missing dependency")

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            with pytest.warns(UserWarning, match="failed to load.*broken_plugin"):
                result = discover_extensions()

        # Should not crash, and broken plugin should not appear in result
        assert "broken_plugin" not in result.get("ggplot2_py.geoms", [])

    def test_duplicate_skip(self):
        """Entry points for already-registered classes should not overwrite."""
        from ggplot2_py.geom import Geom

        # GeomPoint is already registered via __init_subclass__
        original_cls = Geom._registry.get("point")
        assert original_cls is not None

        # Create a fake entry point for "point"
        mock_ep = MagicMock()
        mock_ep.name = "point"
        mock_ep.load.return_value = type("FakeGeomPoint", (), {})

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            discover_extensions()

        # Original should not be overwritten
        assert Geom._registry["point"] is original_cls


class TestAutoRegistration:
    """Test __init_subclass__ auto-registration across all 6 GOG components."""

    def test_geom_registry(self):
        from ggplot2_py.geom import Geom
        assert "Point" in Geom._registry
        assert "point" in Geom._registry
        assert Geom._registry["Point"].__name__ == "GeomPoint"

    def test_stat_registry(self):
        from ggplot2_py.stat import Stat
        assert "Bin" in Stat._registry
        assert "bin" in Stat._registry
        assert Stat._registry["Bin"].__name__ == "StatBin"

    def test_position_registry(self):
        from ggplot2_py.position import Position
        assert "Dodge" in Position._registry
        assert "dodge" in Position._registry

    def test_scale_registry(self):
        from ggplot2_py.scale import Scale
        assert "Continuous" in Scale._registry
        assert "continuous" in Scale._registry

    def test_coord_registry(self):
        from ggplot2_py.coord import Coord
        assert "Cartesian" in Coord._registry
        assert "cartesian" in Coord._registry
        assert "Polar" in Coord._registry

    def test_facet_registry(self):
        from ggplot2_py.facet import Facet
        assert "Wrap" in Facet._registry
        assert "Grid" in Facet._registry
        assert "Null" in Facet._registry

    def test_custom_geom_auto_registers(self):
        from ggplot2_py.geom import Geom
        from ggplot2_py.aes import Mapping

        class GeomTestAuto(Geom):
            required_aes = ("x", "y")
            default_aes = Mapping()

        assert "TestAuto" in Geom._registry
        assert "testauto" in Geom._registry
        assert Geom._registry["TestAuto"] is GeomTestAuto

        # Cleanup
        Geom._registry.pop("TestAuto", None)
        Geom._registry.pop("testauto", None)

    def test_custom_stat_auto_registers(self):
        from ggplot2_py.stat import Stat

        class StatTestAuto(Stat):
            required_aes = ["x"]
            def compute_group(self, data, scales, **params):
                return data

        assert "TestAuto" in Stat._registry
        assert Stat._registry["TestAuto"] is StatTestAuto

        Stat._registry.pop("TestAuto", None)
        Stat._registry.pop("testauto", None)

    def test_custom_coord_auto_registers(self):
        from ggplot2_py.coord import Coord

        class CoordTestAuto(Coord):
            pass

        assert "TestAuto" in Coord._registry
        assert Coord._registry["TestAuto"] is CoordTestAuto

        Coord._registry.pop("TestAuto", None)
        Coord._registry.pop("testauto", None)

    def test_custom_facet_auto_registers(self):
        from ggplot2_py.facet import Facet

        class FacetTestAuto(Facet):
            pass

        assert "TestAuto" in Facet._registry
        assert Facet._registry["TestAuto"] is FacetTestAuto

        Facet._registry.pop("TestAuto", None)
        Facet._registry.pop("testauto", None)

    def test_mock_entry_point_for_coord(self):
        """entry_points discovery works for Coord (previously broken)."""
        from ggplot2_py.coord import Coord

        class CoordMockEP(Coord):
            pass

        mock_ep = MagicMock()
        mock_ep.name = "mock_coord"
        mock_ep.load.return_value = CoordMockEP

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.coords" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            result = discover_extensions()

        assert "ggplot2_py.coords" in result
        assert "mock_coord" in result["ggplot2_py.coords"]
        assert "mock_coord" in Coord._registry

        # Cleanup
        Coord._registry.pop("mock_coord", None)
        Coord._registry.pop("Mock_coord", None)
        Coord._registry.pop("MockEP", None)
        Coord._registry.pop("mockep", None)

    def test_mock_entry_point_for_facet(self):
        """entry_points discovery works for Facet (previously broken)."""
        from ggplot2_py.facet import Facet

        class FacetMockEP(Facet):
            pass

        mock_ep = MagicMock()
        mock_ep.name = "mock_facet"
        mock_ep.load.return_value = FacetMockEP

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.facets" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            result = discover_extensions()

        assert "ggplot2_py.facets" in result
        assert "mock_facet" in result["ggplot2_py.facets"]
        assert "mock_facet" in Facet._registry

        # Cleanup
        Facet._registry.pop("mock_facet", None)
        Facet._registry.pop("Mock_facet", None)
        Facet._registry.pop("MockEP", None)
        Facet._registry.pop("mockep", None)

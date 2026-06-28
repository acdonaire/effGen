"""Tests for ToolPlugin and PluginManager."""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from effgen.tools.plugin import PluginManager, ToolPlugin
from effgen.tools.registry import get_registry


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """ToolRegistry.__init__ needs an event loop (Python 3.9 compat)."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class TestToolPluginBase:
    """Test ToolPlugin base class."""

    def test_default_attributes(self):
        plugin = ToolPlugin()
        assert plugin.name == ""
        assert plugin.version == "0.0.0"
        assert plugin.description == ""
        assert plugin.tools == []

    def test_constructor_overrides(self):
        plugin = ToolPlugin(name="test", version="2.0", description="A plugin")
        assert plugin.name == "test"
        assert plugin.version == "2.0"
        assert plugin.description == "A plugin"

    def test_subclass_attributes(self):
        class MyPlugin(ToolPlugin):
            name = "my_plugin"
            version = "1.0.0"
            description = "My plugin"
            tools = []

        plugin = MyPlugin()
        assert plugin.name == "my_plugin"
        assert plugin.version == "1.0.0"

    def test_register_empty_tools(self):
        plugin = ToolPlugin(name="empty")
        registry = get_registry()
        registered = plugin.register(registry)
        assert registered == []


class TestPluginManagerDiscovery:
    """Test PluginManager discovery methods."""

    def test_discover_entry_points_returns_list(self):
        mgr = PluginManager()
        result = mgr.discover_entry_points()
        assert isinstance(result, list)

    def test_discover_user_dir_no_dir(self):
        mgr = PluginManager()
        with patch("effgen.tools.plugin.Path.home", return_value=Path("/nonexistent")):
            result = mgr.discover_user_dir()
        assert result == []

    def test_discover_env_dir_not_set(self):
        mgr = PluginManager()
        with patch.dict(os.environ, {}, clear=True):
            result = mgr.discover_env_dir()
        assert result == []

    def test_discover_env_dir_nonexistent(self):
        mgr = PluginManager()
        with patch.dict(os.environ, {"EFFGEN_PLUGINS_DIR": "/nonexistent/path"}):
            result = mgr.discover_env_dir()
        assert result == []

    def test_loaded_plugins_empty_initially(self):
        mgr = PluginManager()
        assert mgr.loaded_plugins == {}


class TestPluginManagerLoadFromDirectory:
    """Test loading plugins from a temp directory."""

    def test_load_from_directory_with_plugin(self, tmp_path):
        """Create a temp plugin file and verify it's discovered."""
        plugin_code = '''
from effgen.tools.plugin import ToolPlugin

class TempPlugin(ToolPlugin):
    name = "temp_plugin"
    version = "0.0.1"
    description = "A temporary test plugin"
    tools = []
'''
        plugin_file = tmp_path / "temp_plugin.py"
        plugin_file.write_text(plugin_code)

        mgr = PluginManager()
        loaded = mgr._discover_directory(tmp_path)
        assert "temp_plugin" in loaded
        assert "temp_plugin" in mgr.loaded_plugins

    def test_load_from_directory_skips_underscore_files(self, tmp_path):
        plugin_file = tmp_path / "_private.py"
        plugin_file.write_text("class X: pass")

        mgr = PluginManager()
        loaded = mgr._discover_directory(tmp_path)
        assert loaded == []

    def test_load_from_empty_directory(self, tmp_path):
        mgr = PluginManager()
        loaded = mgr._discover_directory(tmp_path)
        assert loaded == []

    def test_load_plugin_manually(self):
        plugin = ToolPlugin(name="manual_plugin")
        mgr = PluginManager()
        mgr.load_plugin(plugin)
        assert "manual_plugin" in mgr.loaded_plugins


_PLUGIN_WITH_TOOL = '''
from effgen.tools.base_tool import (
    BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata,
)
from effgen.tools.plugin import ToolPlugin


class AutoEchoTool(BaseTool):
    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="auto_echo_tool",
            description="Echo for auto-discovery test.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(name="text", type=ParameterType.STRING,
                                      description="text", required=True)],
        ))

    async def _execute(self, **kwargs):
        return {"echo": kwargs.get("text", "")}


class AutoPlugin(ToolPlugin):
    name = "auto_plugin"
    version = "0.0.1"
    tools = [AutoEchoTool]
'''


class TestRegistryAutoDiscovery:
    """A fresh registry must fold installed/locatable plugins in on first use,
    so a plugin's tools appear wherever built-in tools do — without any manual
    register() call. (Uses EFFGEN_PLUGINS_DIR so no pip install is needed.)"""

    def _fresh_registry(self):
        from effgen.tools.registry import ToolRegistry
        return ToolRegistry()

    def test_plugin_tool_auto_discovered(self, tmp_path):
        (tmp_path / "auto_plugin.py").write_text(_PLUGIN_WITH_TOOL)
        with patch.dict(os.environ, {"EFFGEN_PLUGINS_DIR": str(tmp_path)}, clear=False):
            os.environ.pop("EFFGEN_DISABLE_PLUGINS", None)
            reg = self._fresh_registry()
            # First query triggers lazy built-in + plugin discovery.
            assert "auto_echo_tool" in reg.list_tools()
            tool = reg.get_tool_sync("auto_echo_tool")
            out = asyncio.run(tool.execute(text="hi"))
            assert out.output == {"echo": "hi"}

    def test_disable_env_skips_discovery(self, tmp_path):
        (tmp_path / "auto_plugin.py").write_text(_PLUGIN_WITH_TOOL)
        with patch.dict(
            os.environ,
            {"EFFGEN_PLUGINS_DIR": str(tmp_path), "EFFGEN_DISABLE_PLUGINS": "1"},
            clear=False,
        ):
            reg = self._fresh_registry()
            assert "auto_echo_tool" not in reg.list_tools()
            # Built-ins still present — only plugin discovery is skipped.
            assert "calculator" in reg.list_tools()

    def test_discovery_runs_once(self, tmp_path):
        """Repeated queries do not re-run plugin discovery."""
        reg = self._fresh_registry()
        calls = {"n": 0}
        real = PluginManager.discover_all

        def counting(self):  # noqa: ANN001
            calls["n"] += 1
            return real(self)

        with patch.object(PluginManager, "discover_all", counting):
            reg.list_tools()
            reg.list_tools()
            reg.get_metadata("calculator")
        assert calls["n"] == 1

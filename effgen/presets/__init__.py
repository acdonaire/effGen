"""
effGen Agent Presets — Ready-to-use agent configurations.

Provides factory functions for common agent types:
- math: Calculator + PythonREPL for mathematical tasks
- research: WebSearch + URLFetch + Wikipedia + academic literature tools
- coding: CodeExecutor + PythonREPL + FileOperations + BashTool for coding tasks
- general: All available tools for general-purpose tasks
- minimal: No tools, direct model inference only

Usage:
    from effgen.presets import create_agent, list_presets
    from effgen import load_model

    model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
    agent = create_agent("math", model)
    result = agent.run("What is the square root of 144?")
"""

# Side-effect imports: register additional presets into PRESETS dict
from effgen.presets import media as _media_preset  # noqa: F401
from effgen.presets.registry import PRESETS, create_agent, get_preset, list_presets

__all__ = ["create_agent", "get_preset", "list_presets", "PRESETS"]

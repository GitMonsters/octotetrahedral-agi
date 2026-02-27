"""
Computer Agent - AI-Powered Computer Use Framework
===================================================

A framework for autonomous computer control using vision-language models.
Inspired by Claude Computer Use, built for any OpenAI-compatible LLM.

Architecture:
    Screen Capture → Vision LLM → Action Planning → Input Execution → Loop

Supports:
    - Screenshot capture (macOS native + cross-platform)
    - Mouse control (move, click, drag, scroll)
    - Keyboard control (type, hotkeys, key combos)
    - Multiple LLM backends (Mercury 2, GPT-4o, Claude, etc.)
    - Tool-use agent loop with safety controls
    - Bounded execution regions and kill switch

Usage:
    from computer_agent import ComputerAgent

    agent = ComputerAgent(api_key="your-key", provider="openai")
    agent.run("Open Safari and search for ARC Prize results")
"""

__version__ = "0.1.0"

from computer_agent.agent import ComputerAgent
from computer_agent.screen import ScreenCapture
from computer_agent.input_control import InputController
from computer_agent.llm_backend import LLMBackend
from computer_agent.tools import ToolRegistry
from computer_agent.config import AgentConfig

__all__ = [
    "ComputerAgent",
    "ScreenCapture",
    "InputController",
    "LLMBackend",
    "ToolRegistry",
    "AgentConfig",
]

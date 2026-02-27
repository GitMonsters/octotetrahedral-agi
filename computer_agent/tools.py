"""
Tool registry — defines and manages available computer-use tools.

Each tool maps to an InputController method with typed parameters
and safety validation.
"""

from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolDef:
    """Definition of a computer-use tool."""
    name: str
    description: str
    params: Dict[str, str]  # param_name -> type hint string
    required: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    Registry of available computer-use tools.

    Maps LLM action names → executable functions with parameter
    validation and safety checks.
    """

    # Built-in tool definitions
    TOOLS = [
        ToolDef("click", "Left-click at screen coordinates",
                {"x": "int", "y": "int"}, ["x", "y"]),
        ToolDef("double_click", "Double-click at coordinates",
                {"x": "int", "y": "int"}, ["x", "y"]),
        ToolDef("right_click", "Right-click at coordinates",
                {"x": "int", "y": "int"}, ["x", "y"]),
        ToolDef("type_text", "Type text at current cursor",
                {"text": "str"}, ["text"]),
        ToolDef("press", "Press a single key",
                {"key": "str"}, ["key"]),
        ToolDef("hotkey", "Press key combination",
                {"keys": "list[str]"}, ["keys"]),
        ToolDef("scroll", "Scroll at position",
                {"x": "int", "y": "int", "clicks": "int"}, ["x", "y"]),
        ToolDef("drag", "Click-drag between points",
                {"x1": "int", "y1": "int", "x2": "int", "y2": "int"},
                ["x1", "y1", "x2", "y2"]),
        ToolDef("move", "Move cursor to position",
                {"x": "int", "y": "int"}, ["x", "y"]),
        ToolDef("wait", "Wait before next action",
                {"seconds": "float"}, []),
        ToolDef("done", "Task is complete",
                {"message": "str"}, []),
        ToolDef("fail", "Task cannot be completed",
                {"reason": "str"}, []),
    ]

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {t.name: t for t in self.TOOLS}
        self._executors: Dict[str, Callable] = {}

    def register_executor(self, name: str, fn: Callable):
        """Register an executor function for a tool."""
        self._executors[name] = fn

    def register_custom_tool(self, tool_def: ToolDef, executor: Callable):
        """Register a custom tool with its executor."""
        self._tools[tool_def.name] = tool_def
        self._executors[tool_def.name] = executor

    def validate(self, action: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate an action + params. Returns error string or None if valid.
        """
        if action not in self._tools:
            return f"Unknown action: {action}"

        tool = self._tools[action]

        # Check required params
        for req in tool.required:
            if req not in params:
                return f"Missing required param '{req}' for {action}"

        # Type coercion and validation
        for key, val in params.items():
            if key in tool.params:
                expected = tool.params[key]
                if expected == "int" and not isinstance(val, int):
                    try:
                        params[key] = int(val)
                    except (ValueError, TypeError):
                        return f"Param '{key}' must be int, got {type(val).__name__}"
                elif expected == "float" and not isinstance(val, (int, float)):
                    try:
                        params[key] = float(val)
                    except (ValueError, TypeError):
                        return f"Param '{key}' must be float, got {type(val).__name__}"

        return None

    def execute(self, action: str, params: Dict[str, Any]) -> bool:
        """Execute a tool action. Returns True if executed."""
        if action in self._executors:
            self._executors[action](**params)
            return True
        return False

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the LLM prompt."""
        lines = []
        for tool in self._tools.values():
            param_str = ", ".join(f"{k}: {v}" for k, v in tool.params.items())
            lines.append(f"- {tool.name}({param_str}) — {tool.description}")
        return "\n".join(lines)

    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self._tools.keys())

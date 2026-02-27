"""
Computer Agent — main agent loop that ties everything together.

The agent loop:
    1. Capture screenshot
    2. Send to vision LLM with task + history
    3. Parse action response
    4. Validate and execute action
    5. Repeat until done/fail/max_steps
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from computer_agent.config import AgentConfig
from computer_agent.screen import ScreenCapture
from computer_agent.input_control import InputController
from computer_agent.llm_backend import LLMBackend
from computer_agent.tools import ToolRegistry


class ComputerAgent:
    """
    AI-powered computer use agent.

    Takes a natural language task description, then autonomously
    controls the computer via screenshot → LLM reasoning → action loop.

    Example:
        agent = ComputerAgent(api_key="sk-...", provider="openai")
        result = agent.run("Open Terminal and run 'python3 --version'")
        print(result)  # {'status': 'done', 'message': '...', 'steps': 5}
    """

    def __init__(self, api_key: str = None, provider: str = "openai",
                 model: str = None, config: AgentConfig = None, **kwargs):
        # Build config
        if config:
            self.config = config
        else:
            self.config = AgentConfig(
                api_key=api_key,
                provider=provider,
                model=model,
                **{k: v for k, v in kwargs.items() if hasattr(AgentConfig, k)}
            )

        # Initialize components
        self.screen = ScreenCapture(
            output_dir=self.config.screenshot_dir,
            scale=self.config.screenshot_scale,
        )
        self.input = InputController(
            bounds=self.config.bounds,
        )
        self.llm = LLMBackend(
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.config.get_base_url(),
            model=self.config.get_model(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.tools = ToolRegistry()
        self._register_executors()

        # State
        self.action_history: List[Dict] = []
        self.screenshots: List[str] = []
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            "before_action": [],
            "after_action": [],
            "on_screenshot": [],
            "on_complete": [],
            "on_error": [],
        }

    def _register_executors(self):
        """Wire up tool executors to the InputController."""
        self.tools.register_executor("click",
            lambda x, y: self.input.click(int(x), int(y)))
        self.tools.register_executor("double_click",
            lambda x, y: self.input.double_click(int(x), int(y)))
        self.tools.register_executor("right_click",
            lambda x, y: self.input.right_click(int(x), int(y)))
        self.tools.register_executor("type_text",
            lambda text: self.input.type_text(str(text)))
        self.tools.register_executor("press",
            lambda key: self.input.press(str(key)))
        self.tools.register_executor("hotkey",
            lambda keys=None, **kw: self.input.hotkey(*keys) if keys else None)
        self.tools.register_executor("scroll",
            lambda x, y, clicks=-3: self.input.scroll(int(x), int(y), int(clicks)))
        self.tools.register_executor("drag",
            lambda x1, y1, x2, y2: self.input.drag(int(x1), int(y1), int(x2), int(y2)))
        self.tools.register_executor("move",
            lambda x, y: self.input.move(int(x), int(y)))
        self.tools.register_executor("wait",
            lambda seconds=1: time.sleep(float(seconds)))

    def on(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, **data):
        """Emit an event to registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(**data)
            except Exception as e:
                if self.config.verbose:
                    print(f"  [callback error] {e}")

    def run(self, task: str, max_steps: int = None) -> Dict[str, Any]:
        """
        Execute a task autonomously.

        Args:
            task: Natural language description of what to do
            max_steps: Override max steps (default from config)

        Returns:
            Dict with 'status' ('done'/'fail'/'timeout'), 'message', 'steps', 'history'
        """
        max_steps = max_steps or self.config.max_steps
        self.action_history = []
        self.screenshots = []
        self._running = True
        self.llm.reset()

        screen_size = self.screen.get_screen_size()

        if self.config.verbose:
            print(f"{'='*60}")
            print(f"Computer Agent — Task: {task}")
            print(f"Provider: {self.config.provider} | Model: {self.config.get_model()}")
            print(f"Screen: {screen_size[0]}x{screen_size[1]} | Max steps: {max_steps}")
            print(f"{'='*60}")

        for step in range(max_steps):
            if not self._running:
                break

            if self.config.verbose:
                print(f"\n--- Step {step + 1}/{max_steps} ---")

            # 1. Capture screenshot
            try:
                b64, filepath = self.screen.capture(
                    region=self.config.bounds if self.config.safe_mode else None
                )
                self.screenshots.append(filepath)
                self._emit("on_screenshot", filepath=filepath, step=step)
            except Exception as e:
                if self.config.verbose:
                    print(f"  Screenshot error: {e}")
                self._emit("on_error", error=e, step=step)
                continue

            # 2. Get action from LLM
            try:
                action_data = self.llm.get_action(
                    screenshot_b64=b64,
                    task=task,
                    action_history=self.action_history,
                    screen_size=screen_size,
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"  LLM error: {e}")
                self._emit("on_error", error=e, step=step)
                time.sleep(self.config.step_delay)
                continue

            thought = action_data.get("thought", "")
            action = action_data.get("action", "wait")
            params = action_data.get("params", {})

            if self.config.verbose:
                print(f"  Thought: {thought}")
                print(f"  Action:  {action}({params})")

            # 3. Check for terminal actions
            if action == "done":
                msg = params.get("message", "Task completed")
                if self.config.verbose:
                    print(f"\n✓ DONE: {msg}")
                self._emit("on_complete", message=msg, steps=step+1)
                return {
                    "status": "done",
                    "message": msg,
                    "steps": step + 1,
                    "history": self.action_history,
                }

            if action == "fail":
                reason = params.get("reason", "Unknown failure")
                if self.config.verbose:
                    print(f"\n✗ FAIL: {reason}")
                return {
                    "status": "fail",
                    "message": reason,
                    "steps": step + 1,
                    "history": self.action_history,
                }

            # 4. Validate action
            error = self.tools.validate(action, params)
            if error:
                if self.config.verbose:
                    print(f"  Validation error: {error}")
                self.action_history.append({
                    "action": action, "params": params,
                    "thought": thought, "error": error
                })
                continue

            # 5. Confirm if needed
            if self.config.confirm_actions:
                confirm = input(f"  Execute {action}({params})? [y/N]: ").strip().lower()
                if confirm != "y":
                    if self.config.verbose:
                        print("  Skipped by user")
                    continue

            # 6. Execute action
            self._emit("before_action", action=action, params=params, step=step)
            try:
                self.tools.execute(action, params)
                self.action_history.append({
                    "action": action, "params": params,
                    "thought": thought, "success": True
                })
            except Exception as e:
                if self.config.verbose:
                    print(f"  Execution error: {e}")
                self.action_history.append({
                    "action": action, "params": params,
                    "thought": thought, "error": str(e)
                })
                self._emit("on_error", error=e, step=step)

            self._emit("after_action", action=action, params=params, step=step)

            # 7. Delay between steps
            time.sleep(self.config.step_delay)

        # Max steps reached
        if self.config.verbose:
            print(f"\n⚠ TIMEOUT: Reached max steps ({max_steps})")
        return {
            "status": "timeout",
            "message": f"Reached max steps ({max_steps})",
            "steps": max_steps,
            "history": self.action_history,
        }

    def stop(self):
        """Stop the agent loop."""
        self._running = False

    def step(self, task: str) -> Dict[str, Any]:
        """
        Execute a single step (screenshot → LLM → action).
        Useful for manual control or debugging.
        """
        screen_size = self.screen.get_screen_size()
        b64, filepath = self.screen.capture()
        action_data = self.llm.get_action(
            screenshot_b64=b64,
            task=task,
            action_history=self.action_history,
            screen_size=screen_size,
        )
        return action_data

    def execute_action(self, action: str, params: Dict = None):
        """Manually execute a single action."""
        params = params or {}
        error = self.tools.validate(action, params)
        if error:
            raise ValueError(error)
        self.tools.execute(action, params)

    def save_session(self, filepath: str = None):
        """Save action history to JSON."""
        filepath = filepath or f"/tmp/computer_agent_session_{int(time.time())}.json"
        with open(filepath, "w") as f:
            json.dump({
                "history": self.action_history,
                "screenshots": self.screenshots,
                "config": {
                    "provider": self.config.provider,
                    "model": self.config.get_model(),
                }
            }, f, indent=2)
        return filepath

    def replay(self, session_file: str, delay: float = 1.0):
        """Replay a saved session's actions."""
        with open(session_file) as f:
            session = json.load(f)

        for entry in session["history"]:
            action = entry.get("action")
            params = entry.get("params", {})
            if action in ("done", "fail", "wait"):
                continue
            if self.config.verbose:
                print(f"Replay: {action}({params})")
            try:
                self.tools.execute(action, params)
            except Exception as e:
                print(f"Replay error: {e}")
            time.sleep(delay)

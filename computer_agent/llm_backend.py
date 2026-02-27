"""
LLM Backend — vision-language model integration for computer use.

Sends screenshots + action history to a vision LLM and parses
structured action responses. Supports OpenAI, Anthropic, and
Mercury 2 compatible APIs.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any


class LLMBackend:
    """
    Handles communication with vision-language models.

    The backend sends:
        1. System prompt (computer use instructions + available tools)
        2. Screenshot (base64 PNG)
        3. Action history (what was done so far)
        4. User's task description

    And receives structured tool-call responses that map to
    InputController actions.
    """

    SYSTEM_PROMPT = """You are a computer-use agent. You can see the screen and control the mouse and keyboard to accomplish tasks.

Available tools:
- click(x, y) — Left-click at screen coordinates
- double_click(x, y) — Double-click at coordinates
- right_click(x, y) — Right-click at coordinates
- type_text(text) — Type text at the current cursor position
- press(key) — Press a key (enter, tab, escape, up, down, left, right, backspace, delete, space, f1-f12)
- hotkey(key1, key2, ...) — Press a key combination (e.g., hotkey("cmd", "c") for copy)
- scroll(x, y, clicks) — Scroll at position (negative clicks = scroll down)
- drag(x1, y1, x2, y2) — Click-drag from one point to another
- move(x, y) — Move cursor without clicking
- wait(seconds) — Wait before next action
- done(message) — Task is complete, with summary message
- fail(reason) — Task cannot be completed

Respond with a JSON object containing:
{
    "thought": "Brief reasoning about what you see and what to do next",
    "action": "tool_name",
    "params": { ... tool parameters ... }
}

Guidelines:
- Look at the screenshot carefully before acting
- Click on UI elements at their CENTER coordinates
- After typing, often press Enter to confirm
- Use hotkey("cmd", "a") to select all, hotkey("cmd", "c") to copy, etc.
- If you're unsure, take small steps and observe the result
- Use wait() if you need to let an animation or page load finish
- Call done() when the task is fully complete
- Call fail() if the task is impossible or you're stuck after multiple attempts"""

    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o", temperature: float = 0.0,
                 max_tokens: int = 4096):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or \
                       os.environ.get("MERCURY_API_KEY") or \
                       os.environ.get("LLM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._history: List[Dict] = []

    def reset(self):
        """Clear conversation history."""
        self._history = []

    def get_action(self, screenshot_b64: str, task: str,
                   action_history: List[Dict] = None,
                   screen_size: tuple = None) -> Dict[str, Any]:
        """
        Send screenshot + context to LLM, get next action.

        Args:
            screenshot_b64: Base64-encoded PNG screenshot
            task: User's task description
            action_history: List of previous actions taken
            screen_size: (width, height) of the screen

        Returns:
            Dict with 'thought', 'action', 'params' keys
        """
        messages = self._build_messages(screenshot_b64, task, action_history, screen_size)

        response = self._call_api(messages)
        if response is None:
            return {"thought": "API call failed", "action": "fail",
                    "params": {"reason": "LLM API error"}}

        return self._parse_response(response)

    def _build_messages(self, screenshot_b64: str, task: str,
                        action_history: List[Dict] = None,
                        screen_size: tuple = None) -> List[Dict]:
        """Build the message array for the API call."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Build user message with screenshot + context
        content = []

        # Task description
        task_text = f"Task: {task}\n"
        if screen_size:
            task_text += f"Screen size: {screen_size[0]}x{screen_size[1]}\n"

        # Action history
        if action_history:
            task_text += f"\nActions taken so far ({len(action_history)} steps):\n"
            for i, action in enumerate(action_history[-10:]):  # Last 10 actions
                task_text += f"  {i+1}. {action.get('action', '?')}({action.get('params', {})})"
                if action.get('thought'):
                    task_text += f"  — {action['thought']}"
                task_text += "\n"
            task_text += "\nWhat should the next action be?"

        content.append({"type": "text", "text": task_text})

        # Screenshot
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_b64}",
                "detail": "high"
            }
        })

        messages.append({"role": "user", "content": content})

        return messages

    def _call_api(self, messages: List[Dict]) -> Optional[str]:
        """Make the API call and return response text."""
        if not self.api_key:
            print("[LLMBackend] Error: No API key. Set OPENAI_API_KEY or LLM_API_KEY.")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                json.JSONDecodeError, TimeoutError) as e:
            print(f"[LLMBackend] API error: {e}")
            return None

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured action dict."""
        # Try direct JSON parse
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            result = json.loads(text)
            if "action" in result:
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if "action" in result:
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse freeform text
        return self._parse_freeform(text)

    def _parse_freeform(self, text: str) -> Dict[str, Any]:
        """Last-resort parser for non-JSON responses."""
        text_lower = text.lower()

        if "done" in text_lower or "complete" in text_lower:
            return {"thought": text, "action": "done",
                    "params": {"message": "Task completed"}}

        if "fail" in text_lower or "cannot" in text_lower or "impossible" in text_lower:
            return {"thought": text, "action": "fail",
                    "params": {"reason": text[:200]}}

        # Try to find click coordinates
        import re
        coords = re.findall(r'(?:click|at)\s*\(?(\d+)\s*,\s*(\d+)\)?', text_lower)
        if coords:
            x, y = int(coords[0][0]), int(coords[0][1])
            return {"thought": text, "action": "click", "params": {"x": x, "y": y}}

        # Default: wait and retry
        return {"thought": f"Could not parse response: {text[:100]}",
                "action": "wait", "params": {"seconds": 1}}

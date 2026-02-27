"""
Input controller — mouse and keyboard automation.

Uses PyAutoGUI when available, falls back to macOS osascript/cliclick
or xdotool on Linux.
"""

import sys
import time
import subprocess
from typing import Optional, Tuple, List


class InputController:
    """
    Controls mouse and keyboard input programmatically.

    Mouse:
        move(x, y)              — Move cursor to absolute position
        click(x, y, button)     — Click at position
        double_click(x, y)      — Double-click
        right_click(x, y)       — Right-click
        drag(x1, y1, x2, y2)   — Click-drag from point to point
        scroll(x, y, clicks)    — Scroll wheel at position

    Keyboard:
        type_text(text)         — Type text character by character
        press(key)              — Press a single key
        hotkey(*keys)           — Press key combination (e.g., hotkey('cmd', 'c'))
        key_down(key) / key_up(key) — Hold/release a key

    Safety:
        set_bounds(x, y, w, h) — Restrict all actions to a region
        set_failsafe(enabled)  — Enable corner-abort failsafe
    """

    # Key name mapping for cross-platform compatibility
    KEY_MAP = {
        "enter": "Return", "return": "Return",
        "tab": "Tab", "escape": "Escape", "esc": "Escape",
        "space": "space", "backspace": "BackSpace", "delete": "Delete",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "cmd": "command", "command": "command",
        "ctrl": "control", "control": "control",
        "alt": "option", "option": "option",
        "shift": "shift",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
        "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
        "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
        "home": "Home", "end": "End",
        "pageup": "Page_Up", "pagedown": "Page_Down",
    }

    def __init__(self, bounds: Optional[Tuple[int, int, int, int]] = None,
                 action_delay: float = 0.05):
        self.bounds = bounds  # (x, y, w, h) safety region
        self.action_delay = action_delay
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            return "pyautogui"
        except ImportError:
            pass
        if sys.platform == "darwin":
            return "macos_native"
        if subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0:
            return "xdotool"
        return "none"

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to bounds if set."""
        if self.bounds:
            bx, by, bw, bh = self.bounds
            x = max(bx, min(x, bx + bw))
            y = max(by, min(y, by + bh))
        return x, y

    # --- Mouse ---

    def move(self, x: int, y: int):
        """Move cursor to absolute position."""
        x, y = self._clamp(x, y)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.moveTo(x, y, duration=0.1)
        elif self._backend == "macos_native":
            self._applescript(f'''
                tell application "System Events"
                    set mouseLocation to {{{x}, {y}}}
                end tell
            ''')
            # Use cliclick if available
            subprocess.run(["cliclick", f"m:{x},{y}"], capture_output=True)
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "mousemove", str(x), str(y)])
        time.sleep(self.action_delay)

    def click(self, x: int, y: int, button: str = "left"):
        """Click at position. button: 'left', 'right', 'middle'."""
        x, y = self._clamp(x, y)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.click(x, y, button=button)
        elif self._backend == "macos_native":
            btn_flag = {"left": "c", "right": "rc", "middle": "mc"}.get(button, "c")
            subprocess.run(["cliclick", f"{btn_flag}:{x},{y}"], capture_output=True)
        elif self._backend == "xdotool":
            btn_num = {"left": "1", "right": "3", "middle": "2"}.get(button, "1")
            subprocess.run(["xdotool", "mousemove", str(x), str(y),
                          "click", btn_num])
        time.sleep(self.action_delay)

    def double_click(self, x: int, y: int):
        """Double-click at position."""
        x, y = self._clamp(x, y)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.doubleClick(x, y)
        elif self._backend == "macos_native":
            subprocess.run(["cliclick", f"dc:{x},{y}"], capture_output=True)
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "mousemove", str(x), str(y),
                          "click", "--repeat", "2", "1"])
        time.sleep(self.action_delay)

    def right_click(self, x: int, y: int):
        """Right-click at position."""
        self.click(x, y, button="right")

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        """Click-drag from (x1,y1) to (x2,y2)."""
        x1, y1 = self._clamp(x1, y1)
        x2, y2 = self._clamp(x2, y2)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.moveTo(x1, y1)
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
        elif self._backend == "macos_native":
            subprocess.run(["cliclick", f"dd:{x1},{y1}", f"du:{x2},{y2}"],
                         capture_output=True)
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "mousemove", str(x1), str(y1),
                          "mousedown", "1", "mousemove", "--delay", "500",
                          str(x2), str(y2), "mouseup", "1"])
        time.sleep(self.action_delay)

    def scroll(self, x: int, y: int, clicks: int = -3):
        """Scroll at position. Negative = down, positive = up."""
        x, y = self._clamp(x, y)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.moveTo(x, y)
            pyautogui.scroll(clicks)
        elif self._backend == "macos_native":
            self.move(x, y)
            # AppleScript scroll
            direction = "down" if clicks < 0 else "up"
            for _ in range(abs(clicks)):
                self._applescript(f'''
                    tell application "System Events"
                        scroll {direction}
                    end tell
                ''')
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "mousemove", str(x), str(y)])
            btn = "5" if clicks < 0 else "4"
            for _ in range(abs(clicks)):
                subprocess.run(["xdotool", "click", btn])
        time.sleep(self.action_delay)

    # --- Keyboard ---

    def type_text(self, text: str, interval: float = 0.02):
        """Type text character by character."""
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.typewrite(text, interval=interval) if text.isascii() else pyautogui.write(text)
        elif self._backend == "macos_native":
            # Escape special chars for AppleScript
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            self._applescript(f'''
                tell application "System Events"
                    keystroke "{escaped}"
                end tell
            ''')
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "type", "--clearmodifiers", text])
        time.sleep(self.action_delay)

    def press(self, key: str):
        """Press a single key."""
        mapped = self.KEY_MAP.get(key.lower(), key)
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.press(key.lower())
        elif self._backend == "macos_native":
            if mapped in ("Return", "Tab", "Escape", "BackSpace", "Delete",
                         "Up", "Down", "Left", "Right", "Home", "End",
                         "Page_Up", "Page_Down") or mapped.startswith("F"):
                code = self._key_code(mapped)
                if code is not None:
                    self._applescript(f'''
                        tell application "System Events"
                            key code {code}
                        end tell
                    ''')
            else:
                self._applescript(f'''
                    tell application "System Events"
                        keystroke "{key}"
                    end tell
                ''')
        elif self._backend == "xdotool":
            subprocess.run(["xdotool", "key", mapped])
        time.sleep(self.action_delay)

    def hotkey(self, *keys: str):
        """Press a key combination. E.g., hotkey('cmd', 'c') for copy."""
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.hotkey(*[k.lower() for k in keys])
        elif self._backend == "macos_native":
            # Build AppleScript with modifiers
            modifiers = []
            char_key = None
            for k in keys:
                mapped = self.KEY_MAP.get(k.lower(), k.lower())
                if mapped in ("command", "control", "option", "shift"):
                    modifiers.append(mapped)
                else:
                    char_key = k
            if char_key and modifiers:
                mod_str = " down, ".join(modifiers) + " down"
                self._applescript(f'''
                    tell application "System Events"
                        keystroke "{char_key}" using {{{mod_str}}}
                    end tell
                ''')
        elif self._backend == "xdotool":
            combo = "+".join(self.KEY_MAP.get(k.lower(), k) for k in keys)
            subprocess.run(["xdotool", "key", combo])
        time.sleep(self.action_delay)

    def key_down(self, key: str):
        """Hold a key down."""
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.keyDown(key.lower())

    def key_up(self, key: str):
        """Release a held key."""
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.keyUp(key.lower())

    # --- Safety ---

    def set_bounds(self, x: int, y: int, w: int, h: int):
        """Restrict all mouse actions to this region."""
        self.bounds = (x, y, w, h)

    def clear_bounds(self):
        """Remove mouse action restriction."""
        self.bounds = None

    def set_failsafe(self, enabled: bool = True):
        """Enable/disable PyAutoGUI failsafe (move to corner to abort)."""
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.FAILSAFE = enabled

    # --- Helpers ---

    def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        if self._backend == "pyautogui":
            import pyautogui
            pos = pyautogui.position()
            return pos.x, pos.y
        if self._backend == "macos_native":
            result = subprocess.run(["cliclick", "p:."], capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
        return 0, 0

    def _applescript(self, script: str):
        """Run an AppleScript."""
        subprocess.run(["osascript", "-e", script], capture_output=True)

    def _key_code(self, key_name: str) -> Optional[int]:
        """Map key names to macOS virtual key codes."""
        codes = {
            "Return": 36, "Tab": 9, "Escape": 53, "BackSpace": 51,
            "Delete": 117, "Up": 126, "Down": 125, "Left": 123, "Right": 124,
            "Home": 115, "End": 119, "Page_Up": 116, "Page_Down": 121,
            "F1": 122, "F2": 120, "F3": 99, "F4": 118, "F5": 96, "F6": 97,
            "F7": 98, "F8": 100, "F9": 101, "F10": 109, "F11": 103, "F12": 111,
            "space": 49,
        }
        return codes.get(key_name)

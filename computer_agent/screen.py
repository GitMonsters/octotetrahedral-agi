"""
Screen capture module — captures screenshots for the vision LLM.

Supports macOS native (screencapture), cross-platform (Pillow), and
fallback to subprocess-based capture.
"""

import os
import sys
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class ScreenCapture:
    """
    Captures screenshots of the desktop or specific regions.

    Methods:
        capture()       → Returns (base64_png, filepath)
        capture_region() → Capture a specific rectangular region
        get_screen_size() → Returns (width, height)
    """

    def __init__(self, output_dir: str = "/tmp/computer_agent_screenshots",
                 scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self._frame_count = 0
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect the best available screenshot backend."""
        if sys.platform == "darwin":
            return "macos"
        # Try Pillow
        try:
            from PIL import ImageGrab
            return "pillow"
        except ImportError:
            pass
        # Try scrot (Linux)
        if subprocess.run(["which", "scrot"], capture_output=True).returncode == 0:
            return "scrot"
        return "none"

    def capture(self, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, str]:
        """
        Capture a screenshot.

        Args:
            region: Optional (x, y, width, height) to capture a sub-region.

        Returns:
            (base64_encoded_png, filepath)
        """
        self._frame_count += 1
        timestamp = int(time.time() * 1000)
        filename = f"frame_{self._frame_count:04d}_{timestamp}.png"
        filepath = str(self.output_dir / filename)

        if self._backend == "macos":
            self._capture_macos(filepath, region)
        elif self._backend == "pillow":
            self._capture_pillow(filepath, region)
        elif self._backend == "scrot":
            self._capture_scrot(filepath, region)
        else:
            raise RuntimeError("No screenshot backend available. Install Pillow or scrot.")

        # Downscale if needed
        if self.scale < 1.0:
            self._downscale(filepath)

        # Read and encode
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return b64, filepath

    def capture_region(self, x: int, y: int, w: int, h: int) -> Tuple[str, str]:
        """Capture a specific rectangular region."""
        return self.capture(region=(x, y, w, h))

    def get_screen_size(self) -> Tuple[int, int]:
        """Return (width, height) of the primary display."""
        if sys.platform == "darwin":
            # Use Python/AppKit if available (fast, no subprocess)
            try:
                from AppKit import NSScreen
                frame = NSScreen.mainScreen().frame()
                return int(frame.size.width), int(frame.size.height)
            except ImportError:
                pass
            # Fallback: system_profiler with timeout
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split("\n"):
                    if "Resolution" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "x" and i > 0 and i + 1 < len(parts):
                                return int(parts[i-1]), int(parts[i+1])
            except (subprocess.TimeoutExpired, Exception):
                pass
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return img.size
        except ImportError:
            pass
        return 1920, 1080  # Fallback default

    # --- Backend implementations ---

    def _capture_macos(self, filepath: str, region: Optional[Tuple] = None):
        """macOS native screencapture."""
        cmd = ["screencapture", "-x"]  # -x = no sound
        if region:
            x, y, w, h = region
            cmd.extend(["-R", f"{x},{y},{w},{h}"])
        cmd.append(filepath)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(filepath):
            # Fallback: try without -x flag, or use Pillow
            try:
                self._capture_pillow(filepath, region)
            except ImportError:
                raise RuntimeError(
                    "screencapture failed (grant Screen Recording permission in "
                    "System Settings > Privacy & Security). "
                    "Or install Pillow: pip install Pillow"
                )

    def _capture_pillow(self, filepath: str, region: Optional[Tuple] = None):
        """Cross-platform capture via Pillow."""
        from PIL import ImageGrab
        if region:
            x, y, w, h = region
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            img = ImageGrab.grab()
        img.save(filepath)

    def _capture_scrot(self, filepath: str, region: Optional[Tuple] = None):
        """Linux capture via scrot."""
        if region:
            x, y, w, h = region
            cmd = ["scrot", "-a", f"{x},{y},{w},{h}", filepath]
        else:
            cmd = ["scrot", filepath]
        subprocess.run(cmd, check=True, capture_output=True)

    def _downscale(self, filepath: str):
        """Downscale image to reduce token usage."""
        try:
            from PIL import Image
            img = Image.open(filepath)
            new_w = int(img.width * self.scale)
            new_h = int(img.height * self.scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(filepath)
        except ImportError:
            # If Pillow not available, use sips on macOS
            if sys.platform == "darwin":
                w = int(1920 * self.scale)
                subprocess.run(
                    ["sips", "--resampleWidth", str(w), filepath],
                    capture_output=True
                )

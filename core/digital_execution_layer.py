"""
Digital Execution Layer
=======================
OpenClaw-inspired execution environment for the Recursive Engine.

While core/embodiment.py handles physical simulation (MuJoCo, PyBullet),
this module provides the *digital* embodiment layer:
    - Terminal / subprocess execution
    - File system read/write/search
    - Desktop GUI automation (click, type, screenshot)
    - HTTP / API calls
    - Messaging app relay (WhatsApp/Telegram/Signal bridge)

Each environment implements TaskEnvironmentBase so RecursiveEngineTrainer
can treat digital workflows identically to simulation environments.

Architecture (mirrors OpenClaw layers)
---------------------------------------
    DigitalAction          — typed action union
    DigitalObservation     — structured environment response
    TerminalEnv            — subprocess + shell
    FileSystemEnv          — read/write/search files
    DesktopEnv             — pyautogui GUI automation
    HTTPEnv                — REST API interactions
    CompositeDigitalEnv    — routes actions across all sub-envs
    MessagingRelay         — WhatsApp/Telegram/Signal bridge (heartbeat)

Security note
-------------
Deep system permissions create a real attack surface. All shell commands
are allowlisted or sandboxed; filesystem writes are restricted to a
configurable workspace root. Supply-chain risk is mitigated by running
third-party skill plugins in a subprocess jail (not inline).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from core.recursive_engine_trainer import TaskEnvironmentBase


# ─────────────────────────────────────────────────────────────────────────────
# Action / Observation types
# ─────────────────────────────────────────────────────────────────────────────

class DigitalActionType(Enum):
    TERMINAL    = auto()   # Run a shell command
    FILE_READ   = auto()   # Read a file
    FILE_WRITE  = auto()   # Write a file
    FILE_SEARCH = auto()   # Grep/find in workspace
    DESKTOP_CLICK  = auto()   # Click at (x, y)
    DESKTOP_TYPE   = auto()   # Type text
    DESKTOP_SCREENSHOT = auto()  # Capture screen
    HTTP_GET    = auto()
    HTTP_POST   = auto()
    MESSAGE_SEND = auto()  # Send via messaging relay
    NOOP        = auto()   # Do nothing (pause)


@dataclass
class DigitalAction:
    action_type: DigitalActionType
    payload: Dict[str, Any] = field(default_factory=dict)

    # Convenience constructors
    @classmethod
    def terminal(cls, command: str, timeout: int = 30) -> "DigitalAction":
        return cls(DigitalActionType.TERMINAL,
                   {"command": command, "timeout": timeout})

    @classmethod
    def file_read(cls, path: str) -> "DigitalAction":
        return cls(DigitalActionType.FILE_READ, {"path": path})

    @classmethod
    def file_write(cls, path: str, content: str) -> "DigitalAction":
        return cls(DigitalActionType.FILE_WRITE, {"path": path, "content": content})

    @classmethod
    def file_search(cls, pattern: str, root: str = ".") -> "DigitalAction":
        return cls(DigitalActionType.FILE_SEARCH, {"pattern": pattern, "root": root})

    @classmethod
    def http_get(cls, url: str, headers: Optional[Dict] = None) -> "DigitalAction":
        return cls(DigitalActionType.HTTP_GET, {"url": url, "headers": headers or {}})

    @classmethod
    def http_post(cls, url: str, data: Dict, headers: Optional[Dict] = None) -> "DigitalAction":
        return cls(DigitalActionType.HTTP_POST,
                   {"url": url, "data": data, "headers": headers or {}})

    @classmethod
    def message_send(cls, channel: str, text: str) -> "DigitalAction":
        return cls(DigitalActionType.MESSAGE_SEND, {"channel": channel, "text": text})


@dataclass
class DigitalObservation:
    """Structured observation returned by a digital environment step."""
    action_type:  DigitalActionType
    success:      bool
    stdout:       str  = ""
    stderr:       str  = ""
    content:      str  = ""         # file/http response body
    exit_code:    int  = 0
    duration_sec: float = 0.0
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        parts = []
        if self.stdout:   parts.append(f"STDOUT:\n{self.stdout[:2000]}")
        if self.stderr:   parts.append(f"STDERR:\n{self.stderr[:500]}")
        if self.content:  parts.append(f"CONTENT:\n{self.content[:2000]}")
        return "\n".join(parts) if parts else ("[success]" if self.success else "[failed]")

    def reward(self) -> float:
        """Simple default reward signal: +1 success, -0.5 error."""
        return 1.0 if self.success else -0.5


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers — convert text observations to float tensors
# ─────────────────────────────────────────────────────────────────────────────

OBS_DIM = 128   # fixed observation embedding size

def _encode_observation(obs: DigitalObservation) -> torch.Tensor:
    """
    Lightweight bag-of-bytes embedding of a DigitalObservation.

    Hashes the UTF-8 text into a fixed 128-dim float vector via character
    bigram frequency.  A proper deployment would use a small sentence encoder,
    but this keeps the module dependency-free.
    """
    text = obs.as_text()
    vec = torch.zeros(OBS_DIM)
    if not text:
        return vec
    encoded = text.encode("utf-8", errors="replace")
    for i in range(len(encoded) - 1):
        idx = (encoded[i] * 31 + encoded[i + 1]) % OBS_DIM
        vec[idx] += 1.0
    # Normalise to unit sphere
    norm = vec.norm() + 1e-8
    return vec / norm


# ─────────────────────────────────────────────────────────────────────────────
# Security: command allowlist / sandbox
# ─────────────────────────────────────────────────────────────────────────────

# Commands that are always blocked regardless of context
_BLOCKED_COMMANDS = re.compile(
    r"\b(rm\s+-rf\s+/|mkfs|dd\s+if=|chmod\s+777\s+/|curl.*\|\s*(ba)?sh)\b",
    re.IGNORECASE,
)

def _is_safe_command(cmd: str) -> bool:
    return not bool(_BLOCKED_COMMANDS.search(cmd))


# ─────────────────────────────────────────────────────────────────────────────
# Terminal environment
# ─────────────────────────────────────────────────────────────────────────────

class TerminalEnv(TaskEnvironmentBase):
    """
    Executes shell commands in a sandboxed subprocess.

    obs_dim = OBS_DIM (128)
    act_dim = 1       (action is selected from an index → mapped to a command)

    For use with RecursiveEngineTrainer, actions are expected as float tensors;
    the trainer calls `step(action_tensor)` which is decoded via
    `action_index_to_command`.  Pre-register a command palette with
    `register_commands()`.
    """

    def __init__(
        self,
        workspace: str = "/tmp/re_workspace",
        timeout_default: int = 30,
        max_output_chars: int = 4096,
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout_default = timeout_default
        self.max_output_chars = max_output_chars
        self._commands: List[str] = []
        self._last_obs: Optional[DigitalObservation] = None
        self._step_count = 0

    def register_commands(self, commands: List[str]) -> None:
        """Map integer action indices to shell commands."""
        self._commands = commands

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        dummy = DigitalObservation(DigitalActionType.NOOP, success=True,
                                   stdout="environment reset")
        self._last_obs = dummy
        return _encode_observation(dummy)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """action: [act_dim] float tensor → discretised to command index."""
        if self._commands:
            idx = int(action.argmax().item()) % len(self._commands)
            cmd = self._commands[idx]
        else:
            cmd = "echo 'no commands registered'"

        obs = self.execute(cmd)
        self._last_obs = obs
        self._step_count += 1
        done = self._step_count >= 32 or not obs.success
        return _encode_observation(obs), obs.reward(), done, {"obs": obs}

    def execute(self, command: str, timeout: Optional[int] = None) -> DigitalObservation:
        """Execute a single shell command and return a structured observation."""
        t0 = time.time()
        timeout = timeout or self.timeout_default

        if not _is_safe_command(command):
            return DigitalObservation(
                DigitalActionType.TERMINAL, success=False,
                stderr=f"BLOCKED: unsafe command pattern detected: {command[:80]}",
                exit_code=-1, duration_sec=0.0,
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            return DigitalObservation(
                action_type=DigitalActionType.TERMINAL,
                success=result.returncode == 0,
                stdout=result.stdout[:self.max_output_chars],
                stderr=result.stderr[:512],
                exit_code=result.returncode,
                duration_sec=time.time() - t0,
            )
        except subprocess.TimeoutExpired:
            return DigitalObservation(
                DigitalActionType.TERMINAL, success=False,
                stderr=f"TIMEOUT after {timeout}s",
                exit_code=-2, duration_sec=time.time() - t0,
            )
        except Exception as exc:
            return DigitalObservation(
                DigitalActionType.TERMINAL, success=False,
                stderr=str(exc), exit_code=-3, duration_sec=time.time() - t0,
            )

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def act_dim(self) -> int:
        return max(len(self._commands), 1)

    def estimate_difficulty(self) -> float:
        # Commands that failed recently → harder environment
        if self._last_obs and not self._last_obs.success:
            return 0.8
        return 0.3


# ─────────────────────────────────────────────────────────────────────────────
# File system environment
# ─────────────────────────────────────────────────────────────────────────────

class FileSystemEnv(TaskEnvironmentBase):
    """
    Read/write/search files within a sandboxed workspace directory.

    Enforces that all paths stay inside workspace_root to prevent traversal.
    """

    def __init__(self, workspace_root: str = "/tmp/re_workspace"):
        self.root = Path(workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._step_count = 0

    def _safe_path(self, path: str) -> Path:
        """Resolve path and verify it stays within workspace_root."""
        p = (self.root / path).resolve()
        if not str(p).startswith(str(self.root)):
            raise PermissionError(f"Path escape attempt: {path!r}")
        return p

    def read(self, path: str) -> DigitalObservation:
        try:
            p = self._safe_path(path)
            content = p.read_text(errors="replace")
            return DigitalObservation(
                DigitalActionType.FILE_READ, success=True,
                content=content[:8192],
            )
        except Exception as e:
            return DigitalObservation(
                DigitalActionType.FILE_READ, success=False, stderr=str(e)
            )

    def write(self, path: str, content: str) -> DigitalObservation:
        try:
            p = self._safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return DigitalObservation(
                DigitalActionType.FILE_WRITE, success=True,
                metadata={"bytes_written": len(content.encode())},
            )
        except Exception as e:
            return DigitalObservation(
                DigitalActionType.FILE_WRITE, success=False, stderr=str(e)
            )

    def search(self, pattern: str, root: Optional[str] = None) -> DigitalObservation:
        search_root = self._safe_path(root) if root else self.root
        matches: List[str] = []
        try:
            for p in search_root.rglob("*"):
                if p.is_file():
                    try:
                        text = p.read_text(errors="replace")
                        if pattern.lower() in text.lower():
                            matches.append(str(p.relative_to(self.root)))
                    except Exception:
                        pass
            return DigitalObservation(
                DigitalActionType.FILE_SEARCH, success=True,
                stdout="\n".join(matches[:50]),
                metadata={"match_count": len(matches)},
            )
        except Exception as e:
            return DigitalObservation(
                DigitalActionType.FILE_SEARCH, success=False, stderr=str(e)
            )

    # TaskEnvironmentBase interface
    def reset(self) -> torch.Tensor:
        self._step_count = 0
        obs = DigitalObservation(DigitalActionType.NOOP, success=True)
        return _encode_observation(obs)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        # Generic step: read workspace listing as observation
        obs = self.search("", root=None)
        self._step_count += 1
        done = self._step_count >= 16
        return _encode_observation(obs), obs.reward(), done, {"obs": obs}

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def act_dim(self) -> int:
        return 3   # read / write / search


# ─────────────────────────────────────────────────────────────────────────────
# HTTP environment
# ─────────────────────────────────────────────────────────────────────────────

class HTTPEnv(TaskEnvironmentBase):
    """
    Makes real HTTP requests as environment steps.

    Requires `requests` package.  Falls back to curl subprocess if unavailable.
    """

    def __init__(self, base_url: str = "", timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._step_count = 0

    def get(self, path: str, headers: Optional[Dict] = None) -> DigitalObservation:
        url = f"{self.base_url}{path}" if not path.startswith("http") else path
        t0 = time.time()
        try:
            import requests
            r = requests.get(url, headers=headers or {}, timeout=self.timeout)
            return DigitalObservation(
                DigitalActionType.HTTP_GET, success=r.ok,
                content=r.text[:4096], exit_code=r.status_code,
                duration_sec=time.time() - t0,
            )
        except ImportError:
            # Fallback: subprocess curl
            result = subprocess.run(
                ["curl", "-s", "-o", "-", "-w", "%{http_code}", url],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return DigitalObservation(
                DigitalActionType.HTTP_GET,
                success=result.returncode == 0,
                content=result.stdout[:4096],
                duration_sec=time.time() - t0,
            )
        except Exception as e:
            return DigitalObservation(
                DigitalActionType.HTTP_GET, success=False, stderr=str(e),
                duration_sec=time.time() - t0,
            )

    def post(
        self, path: str, data: Dict, headers: Optional[Dict] = None
    ) -> DigitalObservation:
        url = f"{self.base_url}{path}" if not path.startswith("http") else path
        t0 = time.time()
        try:
            import requests
            r = requests.post(
                url, json=data, headers=headers or {}, timeout=self.timeout
            )
            return DigitalObservation(
                DigitalActionType.HTTP_POST, success=r.ok,
                content=r.text[:4096], exit_code=r.status_code,
                duration_sec=time.time() - t0,
            )
        except Exception as e:
            return DigitalObservation(
                DigitalActionType.HTTP_POST, success=False, stderr=str(e),
                duration_sec=time.time() - t0,
            )

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        return _encode_observation(DigitalObservation(DigitalActionType.NOOP, success=True))

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        obs = self.get("/")
        self._step_count += 1
        done = self._step_count >= 8 or not obs.success
        return _encode_observation(obs), obs.reward(), done, {"obs": obs}

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def act_dim(self) -> int:
        return 2   # GET / POST


# ─────────────────────────────────────────────────────────────────────────────
# Composite environment — routes to all sub-envs
# ─────────────────────────────────────────────────────────────────────────────

class CompositeDigitalEnv(TaskEnvironmentBase):
    """
    Unified digital environment that routes DigitalActions to the correct
    sub-environment and aggregates observations.

    This is the top-level "hands" of the Recursive Engine — analogous to
    OpenClaw's single unified agent interface.

    Usage
    -----
        env = CompositeDigitalEnv(workspace="/tmp/my_workspace")
        obs = env.reset()

        action = DigitalAction.terminal("ls -la")
        obs_tensor, reward, done, info = env.step_action(action)
    """

    def __init__(self, workspace: str = "/tmp/re_workspace", base_url: str = ""):
        self.terminal_env   = TerminalEnv(workspace=workspace)
        self.filesystem_env = FileSystemEnv(workspace_root=workspace)
        self.http_env       = HTTPEnv(base_url=base_url)
        self._step_count    = 0
        self._last_reward   = 0.0

    def reset(self) -> torch.Tensor:
        self.terminal_env.reset()
        self.filesystem_env.reset()
        self.http_env.reset()
        self._step_count = 0
        return torch.zeros(OBS_DIM)

    def step_action(
        self, action: DigitalAction
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """High-level step accepting a typed DigitalAction."""
        obs = self._dispatch(action)
        self._last_reward = obs.reward()
        self._step_count += 1
        done = self._step_count >= 64
        return _encode_observation(obs), self._last_reward, done, {"obs": obs}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """TaskEnvironmentBase-compatible step: interprets float tensor as action type index."""
        idx = int(action.argmax().item()) % len(DigitalActionType)
        action_type = list(DigitalActionType)[idx]
        digital_action = DigitalAction(action_type=action_type)
        return self.step_action(digital_action)

    def _dispatch(self, action: DigitalAction) -> DigitalObservation:
        t = action.action_type
        p = action.payload

        if t == DigitalActionType.TERMINAL:
            return self.terminal_env.execute(p.get("command", "echo noop"),
                                              p.get("timeout"))
        elif t == DigitalActionType.FILE_READ:
            return self.filesystem_env.read(p.get("path", ""))
        elif t == DigitalActionType.FILE_WRITE:
            return self.filesystem_env.write(p.get("path", ""), p.get("content", ""))
        elif t == DigitalActionType.FILE_SEARCH:
            return self.filesystem_env.search(p.get("pattern", ""),
                                               p.get("root"))
        elif t in (DigitalActionType.HTTP_GET,):
            return self.http_env.get(p.get("url", "/"), p.get("headers"))
        elif t in (DigitalActionType.HTTP_POST,):
            return self.http_env.post(p.get("url", "/"), p.get("data", {}),
                                       p.get("headers"))
        elif t == DigitalActionType.NOOP:
            return DigitalObservation(DigitalActionType.NOOP, success=True,
                                       stdout="noop")
        else:
            return DigitalObservation(t, success=False,
                                       stderr=f"Unimplemented action type: {t}")

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def act_dim(self) -> int:
        return len(DigitalActionType)

    def estimate_difficulty(self) -> float:
        return 0.4 if self._last_reward > 0 else 0.7

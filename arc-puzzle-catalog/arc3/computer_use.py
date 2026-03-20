"""
ARC-AGI-3 Computer Use Framework
================================

A visual agent framework inspired by Claude's Computer Use capability.
Treats ARC-AGI-3 game environments as a "computer screen" — the agent
observes pixel frames (screenshots), reasons about what it sees, and
takes actions (clicks, movement) through structured tool calls.

Architecture:
    ┌─────────────────────────────────────────────┐
    │           ComputerUseAgent                   │
    │  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
    │  │ Screen  │→│ Reasoner │→│  Executor  │  │
    │  │ Reader  │  │ (LLM/    │  │  (Actions) │  │
    │  │         │  │  Local)   │  │            │  │
    │  └─────────┘  └──────────┘  └───────────┘  │
    │       ↑              ↑              │        │
    │       │         ┌────┴────┐         │        │
    │       │         │ Memory  │         │        │
    │       │         │ + Tools │         ↓        │
    │       └─────────┴─────────┴─── env.step() ──│
    └─────────────────────────────────────────────┘

Usage:
    from arc3.computer_use import ComputerUseAgent

    agent = ComputerUseAgent()
    result = agent.play(env, GameAction)
"""

from __future__ import annotations
import logging
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger("arc3.computer_use")


# ─── Screen Analysis ────────────────────────────────────────────────

@dataclass
class ScreenObject:
    """A detected object on screen."""
    id: int
    bbox: tuple[int, int, int, int]  # (row, col, height, width)
    color: int
    pixels: list[tuple[int, int]]
    center: tuple[int, int]
    size: int

    @property
    def clickable_pos(self) -> tuple[int, int]:
        """Return (x=col, y=row) for clicking."""
        return (self.center[1], self.center[0])


@dataclass
class ScreenState:
    """Parsed screen state from a frame observation."""
    frame: np.ndarray
    objects: list[ScreenObject]
    background_color: int
    unique_colors: set[int]
    changed_pixels: int = 0
    changed_regions: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.frame.shape[1]

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    def pixel_at(self, row: int, col: int) -> int:
        if 0 <= row < self.height and 0 <= col < self.width:
            return int(self.frame[row, col])
        return -1

    def region(self, r: int, c: int, h: int, w: int) -> np.ndarray:
        return self.frame[r:r+h, c:c+w].copy()

    def to_text(self, max_rows: int = 64) -> str:
        """RLE-compressed text representation for LLM consumption."""
        lines = []
        for r in range(min(max_rows, self.height)):
            row = self.frame[r]
            runs = []
            i = 0
            while i < len(row):
                val = int(row[i])
                count = 1
                while i + count < len(row) and int(row[i + count]) == val:
                    count += 1
                if count > 2:
                    runs.append(f"{val}x{count}")
                else:
                    runs.extend([str(val)] * count)
                i += count
            lines.append(",".join(runs))
        return "\n".join(lines)

    def summary(self) -> str:
        return (f"{len(self.objects)} objects, {len(self.unique_colors)} colors, "
                f"bg={self.background_color}")


class ScreenReader:
    """Reads and parses game frames into structured ScreenState objects."""

    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None

    def read(self, frame: np.ndarray) -> ScreenState:
        """Parse a 64x64 frame into a ScreenState."""
        f = frame[:, :] if frame.ndim == 2 else frame

        bg = int(np.argmax(np.bincount(f.flatten().astype(int))))
        colors = set(int(v) for v in np.unique(f))
        colors.discard(bg)
        colors.discard(0)

        objects = self._find_objects(f, bg)

        changed = 0
        changed_regions = []
        if self.prev_frame is not None:
            diff = (f != self.prev_frame)
            changed = int(np.sum(diff))
            if changed > 0:
                rows, cols = np.where(diff)
                if len(rows) > 0:
                    changed_regions.append((
                        int(rows.min()), int(cols.min()),
                        int(rows.max() - rows.min() + 1),
                        int(cols.max() - cols.min() + 1)
                    ))

        self.prev_frame = f.copy()

        return ScreenState(
            frame=f, objects=objects, background_color=bg,
            unique_colors=colors, changed_pixels=changed,
            changed_regions=changed_regions,
        )

    def _find_objects(self, frame: np.ndarray, bg: int) -> list[ScreenObject]:
        """Detect distinct objects via flood fill."""
        h, w = frame.shape
        visited = np.zeros((h, w), dtype=bool)
        objects = []
        obj_id = 0

        for r in range(h):
            for c in range(w):
                color = int(frame[r, c])
                if color == bg or color == 0 or visited[r, c]:
                    continue

                pixels = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    pr, pc = queue.popleft()
                    pixels.append((pr, pc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = pr + dr, pc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and not visited[nr, nc]
                                and int(frame[nr, nc]) == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                if len(pixels) < 2:
                    continue

                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                center = (sum(rows) // len(rows), sum(cols) // len(cols))
                bbox = (min(rows), min(cols),
                        max(rows) - min(rows) + 1,
                        max(cols) - min(cols) + 1)

                objects.append(ScreenObject(
                    id=obj_id, bbox=bbox, color=color,
                    pixels=pixels, center=center, size=len(pixels),
                ))
                obj_id += 1

        return objects


# ─── Tool System ────────────────────────────────────────────────────

class ToolType(Enum):
    SCREENSHOT = "screenshot"
    CLICK = "click"
    MOVE = "move"
    FIND_OBJECTS = "find_objects"
    PIXEL_AT = "pixel_at"
    DETECT_CHANGES = "detect_changes"
    RESET = "reset"
    WAIT = "wait"


@dataclass
class ToolCall:
    """A structured tool invocation (like Claude's tool_use)."""
    tool: ToolType
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tool": self.tool.value, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> ToolCall:
        return cls(tool=ToolType(d["tool"]), params=d.get("params", {}))


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool: ToolType
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_text(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, ScreenState):
            return self.data.summary()
        if isinstance(self.data, list):
            return json.dumps(self.data[:20], default=str)
        return str(self.data)


# ─── Action Memory ──────────────────────────────────────────────────

@dataclass
class ActionRecord:
    """Record of a single action and its observed effect."""
    step: int
    tool_call: ToolCall
    screen_before: Optional[ScreenState]
    screen_after: Optional[ScreenState]
    pixels_changed: int
    level_advanced: bool
    game_over: bool
    timestamp: float


class ActionMemory:
    """Tracks action history and learns cause-effect patterns."""

    def __init__(self, max_history: int = 500):
        self.history: list[ActionRecord] = []
        self.max_history = max_history
        self.action_effects: dict[str, list[int]] = {}
        self.successful_sequences: list[list[ToolCall]] = []
        self.failed_actions: dict[str, int] = {}

    def record(self, rec: ActionRecord):
        self.history.append(rec)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        key = rec.tool_call.tool.value
        if key not in self.action_effects:
            self.action_effects[key] = []
        self.action_effects[key].append(rec.pixels_changed)

        if rec.game_over:
            self.failed_actions[key] = self.failed_actions.get(key, 0) + 1

    def get_effective_actions(self) -> list[str]:
        """Return actions that typically cause pixel changes."""
        effective = []
        for action, changes in self.action_effects.items():
            if changes and np.mean(changes) > 0:
                effective.append(action)
        return effective

    def recent_summary(self, n: int = 10) -> str:
        """Text summary of recent actions for LLM context."""
        lines = []
        for rec in self.history[-n:]:
            tool = rec.tool_call.tool.value
            params = rec.tool_call.params
            effect = f"{rec.pixels_changed}px"
            if rec.level_advanced:
                effect += " LEVEL_UP"
            if rec.game_over:
                effect += " GAME_OVER"
            lines.append(f"  [{rec.step}] {tool}({params}) -> {effect}")
        return "\n".join(lines)

    @property
    def total_actions(self) -> int:
        return len(self.history)


# ─── Tool Executor ──────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls against the game environment."""

    def __init__(self, env, gac, screen_reader: ScreenReader):
        self.env = env
        self.gac = gac
        self.screen = screen_reader
        self.AMAP = {a.value: a for a in gac}
        self.obs = None
        self.current_screen: Optional[ScreenState] = None
        self.actions_taken = 0

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        try:
            handler = {
                ToolType.SCREENSHOT: self._screenshot,
                ToolType.CLICK: self._click,
                ToolType.MOVE: self._move,
                ToolType.FIND_OBJECTS: self._find_objects,
                ToolType.PIXEL_AT: self._pixel_at,
                ToolType.DETECT_CHANGES: self._detect_changes,
                ToolType.RESET: self._reset,
                ToolType.WAIT: self._wait,
            }.get(call.tool)

            if handler is None:
                return ToolResult(call.tool, False, error=f"Unknown tool: {call.tool}")

            return handler(call.params)
        except Exception as e:
            return ToolResult(call.tool, False, error=str(e))

    def _screenshot(self, params: dict) -> ToolResult:
        """Capture current screen state."""
        if self.obs is None:
            return ToolResult(ToolType.SCREENSHOT, False, error="No observation")
        frame = self.obs.frame[0]
        self.current_screen = self.screen.read(frame)
        return ToolResult(ToolType.SCREENSHOT, True, data=self.current_screen)

    def _click(self, params: dict) -> ToolResult:
        """Click at (x, y) coordinates."""
        x = params.get("x", 0)
        y = params.get("y", 0)
        self.obs = self.env.step(self.gac.ACTION6, data={"x": x, "y": y})
        self.actions_taken += 1
        self.current_screen = self.screen.read(self.obs.frame[0])
        return ToolResult(ToolType.CLICK, True, data={
            "state": str(self.obs.state),
            "levels_completed": self.obs.levels_completed,
            "changed": self.current_screen.changed_pixels,
        })

    def _move(self, params: dict) -> ToolResult:
        """Move in a direction (1=up, 2=right, 3=down, 4=left)."""
        direction = params.get("direction", 1)
        if direction not in self.AMAP:
            return ToolResult(ToolType.MOVE, False, error=f"Invalid direction: {direction}")
        self.obs = self.env.step(self.AMAP[direction])
        self.actions_taken += 1
        self.current_screen = self.screen.read(self.obs.frame[0])
        return ToolResult(ToolType.MOVE, True, data={
            "state": str(self.obs.state),
            "levels_completed": self.obs.levels_completed,
            "changed": self.current_screen.changed_pixels,
        })

    def _find_objects(self, params: dict) -> ToolResult:
        """Find all objects on screen, optionally filtered."""
        if self.current_screen is None:
            self._screenshot({})
        color_filter = params.get("color")
        min_size = params.get("min_size", 2)
        objs = self.current_screen.objects
        if color_filter is not None:
            objs = [o for o in objs if o.color == color_filter]
        objs = [o for o in objs if o.size >= min_size]
        result = [{"id": o.id, "center": o.center, "color": o.color,
                    "size": o.size, "bbox": o.bbox} for o in objs]
        return ToolResult(ToolType.FIND_OBJECTS, True, data=result)

    def _pixel_at(self, params: dict) -> ToolResult:
        """Read pixel color at (row, col)."""
        if self.current_screen is None:
            self._screenshot({})
        row = params.get("row", 0)
        col = params.get("col", 0)
        val = self.current_screen.pixel_at(row, col)
        return ToolResult(ToolType.PIXEL_AT, True, data=val)

    def _detect_changes(self, params: dict) -> ToolResult:
        """Report what changed since last screenshot."""
        if self.current_screen is None:
            return ToolResult(ToolType.DETECT_CHANGES, True, data={
                "changed": 0, "regions": []})
        return ToolResult(ToolType.DETECT_CHANGES, True, data={
            "changed": self.current_screen.changed_pixels,
            "regions": self.current_screen.changed_regions,
        })

    def _reset(self, params: dict) -> ToolResult:
        """Reset the game."""
        self.obs = self.env.step(self.gac.RESET)
        self.actions_taken += 1
        self.current_screen = self.screen.read(self.obs.frame[0])
        return ToolResult(ToolType.RESET, True, data={
            "state": str(self.obs.state),
            "levels_completed": self.obs.levels_completed,
        })

    def _wait(self, params: dict) -> ToolResult:
        """No-op observation."""
        return ToolResult(ToolType.WAIT, True, data="ok")


# ─── Reasoning Strategies ──────────────────────────────────────────

class ReasoningStrategy(Enum):
    SYSTEMATIC_EXPLORE = "systematic_explore"
    OBJECT_INTERACT = "object_interact"
    PATTERN_MATCH = "pattern_match"
    LLM_GUIDED = "llm_guided"
    TOGGLE_SOLVE = "toggle_solve"
    BFS_SEARCH = "bfs_search"


class LocalReasoner:
    """Rule-based reasoning without LLM — fast heuristic strategies."""

    def __init__(self):
        self.movement_map: dict[int, tuple[int, int]] = {}
        self.click_effects: dict[tuple[int, int], int] = {}
        self.strategy = ReasoningStrategy.SYSTEMATIC_EXPLORE
        self.explore_queue: list[ToolCall] = []
        self.step = 0

    def decide(self, screen: ScreenState, memory: ActionMemory,
               available_actions: list[int]) -> list[ToolCall]:
        """Decide next tool calls based on current screen and history."""
        self.step += 1

        if self.strategy == ReasoningStrategy.SYSTEMATIC_EXPLORE:
            return self._systematic_explore(screen, memory, available_actions)
        elif self.strategy == ReasoningStrategy.OBJECT_INTERACT:
            return self._object_interact(screen, memory)
        elif self.strategy == ReasoningStrategy.TOGGLE_SOLVE:
            return self._toggle_probe(screen, memory)
        else:
            return self._systematic_explore(screen, memory, available_actions)

    def _systematic_explore(self, screen: ScreenState,
                            memory: ActionMemory,
                            available: list[int]) -> list[ToolCall]:
        """Try each available action systematically."""
        calls = []
        has_move = any(a in available for a in [1, 2, 3, 4])
        has_click = 6 in available

        if has_click and screen.objects:
            # Click the first untried object
            tried_positions = set()
            for rec in memory.history[-50:]:
                if rec.tool_call.tool == ToolType.CLICK:
                    p = rec.tool_call.params
                    tried_positions.add((p.get("x", 0), p.get("y", 0)))

            for obj in screen.objects:
                pos = obj.clickable_pos
                if pos not in tried_positions:
                    calls.append(ToolCall(ToolType.CLICK, {"x": pos[0], "y": pos[1]}))
                    if len(calls) >= 3:
                        break

        if not calls and has_move:
            # Try movement directions
            for d in [1, 2, 3, 4]:
                if d in available:
                    calls.append(ToolCall(ToolType.MOVE, {"direction": d}))
                    break

        if not calls:
            calls.append(ToolCall(ToolType.SCREENSHOT, {}))

        return calls

    def _object_interact(self, screen: ScreenState,
                         memory: ActionMemory) -> list[ToolCall]:
        """Click on detected objects."""
        calls = []
        for obj in sorted(screen.objects, key=lambda o: o.size, reverse=True)[:5]:
            x, y = obj.clickable_pos
            calls.append(ToolCall(ToolType.CLICK, {"x": x, "y": y}))
        return calls or [ToolCall(ToolType.SCREENSHOT, {})]

    def _toggle_probe(self, screen: ScreenState,
                      memory: ActionMemory) -> list[ToolCall]:
        """Probe click effects for toggle puzzles."""
        calls = []
        for obj in screen.objects[:10]:
            x, y = obj.clickable_pos
            calls.append(ToolCall(ToolType.CLICK, {"x": x, "y": y}))
        return calls or [ToolCall(ToolType.SCREENSHOT, {})]

    def update_strategy(self, memory: ActionMemory, screen: ScreenState):
        """Switch strategy based on observations."""
        if memory.total_actions < 10:
            self.strategy = ReasoningStrategy.SYSTEMATIC_EXPLORE
            return

        recent = memory.history[-20:]
        clicks_effective = sum(1 for r in recent
                               if r.tool_call.tool == ToolType.CLICK
                               and r.pixels_changed > 0)
        moves_effective = sum(1 for r in recent
                              if r.tool_call.tool == ToolType.MOVE
                              and r.pixels_changed > 0)

        if clicks_effective > moves_effective * 2:
            self.strategy = ReasoningStrategy.OBJECT_INTERACT
        elif moves_effective > clicks_effective * 2:
            self.strategy = ReasoningStrategy.SYSTEMATIC_EXPLORE
        elif clicks_effective > 5:
            self.strategy = ReasoningStrategy.TOGGLE_SOLVE


class LLMReasoner:
    """LLM-backed reasoning using Mercury or any OpenAI-compatible model."""

    def __init__(self, mercury=None):
        self.mercury = mercury
        self.available = mercury is not None and mercury.available

    def decide(self, screen: ScreenState, memory: ActionMemory,
               game_info: dict) -> Optional[list[ToolCall]]:
        """Ask LLM for next actions."""
        if not self.available:
            return None

        actions = self.mercury.suggest_actions(
            screen.frame,
            history=[{"action": r.tool_call.tool.value,
                       "changed": r.pixels_changed}
                      for r in memory.history[-10:]],
            game_info=game_info,
        )
        if not actions:
            return None

        calls = []
        for a in actions:
            if isinstance(a, dict):
                if a.get("type") == "click":
                    calls.append(ToolCall(ToolType.CLICK,
                                          {"x": a.get("x", 0), "y": a.get("y", 0)}))
                elif a.get("type") == "move":
                    calls.append(ToolCall(ToolType.MOVE,
                                          {"direction": a.get("dir", 1)}))
            elif isinstance(a, int) and 1 <= a <= 4:
                calls.append(ToolCall(ToolType.MOVE, {"direction": a}))
        return calls if calls else None


# ─── Main Agent ─────────────────────────────────────────────────────

class ComputerUseAgent:
    """Computer-use style agent for ARC-AGI-3 games.

    Runs a screenshot→reason→act loop, using tool calls to interact
    with the game environment — similar to Claude's computer use but
    for pixel-based game environments.

    Supports both local heuristic reasoning and LLM-guided reasoning
    (Mercury 2 or any OpenAI-compatible model).
    """

    def __init__(self, max_actions: int = 2000, verbose: bool = False,
                 use_llm: bool = True):
        self.max_actions = max_actions
        self.verbose = verbose
        self.screen_reader = ScreenReader()
        self.memory = ActionMemory()
        self.local_reasoner = LocalReasoner()
        self.llm_reasoner = None

        if use_llm:
            try:
                from arc3.mercury import MercuryReasoner
                mercury = MercuryReasoner()
                self.llm_reasoner = LLMReasoner(mercury)
                if self.llm_reasoner.available and verbose:
                    logger.info("Computer Use: LLM reasoning enabled (Mercury 2)")
            except Exception:
                pass

    def reset(self):
        """Reset agent state for a new game."""
        self.screen_reader = ScreenReader()
        self.memory = ActionMemory()
        self.local_reasoner = LocalReasoner()

    def play(self, env, game_action_class) -> dict:
        """Play a game using the computer-use loop.

        Returns dict with levels_completed, total_actions, won, etc.
        """
        from arcengine import GameState

        t0 = time.time()
        executor = ToolExecutor(env, game_action_class, self.screen_reader)
        self.memory = ActionMemory()

        # Initial reset
        result = executor.execute(ToolCall(ToolType.RESET))
        obs = executor.obs
        wl = obs.win_levels
        available = sorted(obs.available_actions)
        step = 0

        # Take initial screenshot
        executor.execute(ToolCall(ToolType.SCREENSHOT))
        screen = executor.current_screen

        if self.verbose:
            logger.info(f"Game started: {wl} win_levels, "
                        f"actions={available}, screen: {screen.summary()}")

        prev_lc = obs.levels_completed

        while step < self.max_actions:
            if obs.state == GameState.WIN:
                break
            if obs.state == GameState.GAME_OVER:
                # Auto-reset on game over
                executor.execute(ToolCall(ToolType.RESET))
                obs = executor.obs
                screen = executor.current_screen
                step += 1
                continue

            # Update strategy based on what we've learned
            self.local_reasoner.update_strategy(self.memory, screen)

            # Decide next tool calls
            calls = None

            # Try LLM reasoning first (if available and periodic)
            if (self.llm_reasoner and self.llm_reasoner.available
                    and step % 50 == 0 and step > 0):
                calls = self.llm_reasoner.decide(screen, self.memory, {
                    "level": obs.levels_completed,
                    "budget": self.max_actions - step,
                    "strategy": self.local_reasoner.strategy.value,
                })

            # Fall back to local reasoning
            if not calls:
                calls = self.local_reasoner.decide(screen, self.memory, available)

            # Execute tool calls
            for call in calls:
                if step >= self.max_actions:
                    break

                screen_before = screen
                result = executor.execute(call)
                obs = executor.obs

                # Update screen state
                if executor.current_screen:
                    screen = executor.current_screen

                # Track the action
                changed = screen.changed_pixels if screen else 0
                lc = obs.levels_completed if obs else prev_lc
                advanced = lc > prev_lc
                game_over = obs.state == GameState.GAME_OVER if obs else False

                self.memory.record(ActionRecord(
                    step=step, tool_call=call,
                    screen_before=screen_before, screen_after=screen,
                    pixels_changed=changed, level_advanced=advanced,
                    game_over=game_over, timestamp=time.time(),
                ))

                step += 1

                if advanced:
                    prev_lc = lc
                    if self.verbose:
                        logger.info(f"[{step}] Level up! Now at L{lc}")
                    # Take fresh screenshot after level change
                    executor.execute(ToolCall(ToolType.SCREENSHOT))
                    screen = executor.current_screen

                if obs and obs.state == GameState.WIN:
                    break
                if game_over:
                    break

            # Periodic logging
            if self.verbose and step % 100 == 0 and step > 0:
                logger.info(f"[{step}] L{obs.levels_completed}/{wl} "
                            f"strategy={self.local_reasoner.strategy.value} "
                            f"objects={len(screen.objects) if screen else '?'}")

        elapsed = round(time.time() - t0, 1)
        won = obs.state == GameState.WIN if obs else False
        lc = obs.levels_completed if obs else 0

        if self.verbose:
            logger.info(f"Game done: {lc}/{wl} levels, {step} actions, "
                        f"{'WON' if won else 'not won'}, {elapsed}s")

        return {
            "levels_completed": lc,
            "win_levels": wl,
            "total_actions": step,
            "won": won,
            "elapsed_seconds": elapsed,
            "strategy_used": self.local_reasoner.strategy.value,
            "memory_size": self.memory.total_actions,
        }


# ─── Hybrid Agent (Computer Use + Toggle Solver) ───────────────────

class HybridComputerUseAgent:
    """Combines ComputerUseAgent with the proven toggle solver.

    Uses the toggle solver for click-based puzzle levels and
    computer-use exploration for movement/interaction levels.
    """

    def __init__(self, max_actions: int = 2000, verbose: bool = False):
        self.max_actions = max_actions
        self.verbose = verbose

    def play(self, env, game_action_class) -> dict:
        """Play using toggle solver first, computer-use for the rest."""
        from arc3.agent import OctoTetraAgent

        # Primary: proven algorithmic agent (BFS + toggle solver)
        algo_agent = OctoTetraAgent(
            max_actions_per_level=self.max_actions,
            verbose=self.verbose,
            use_mercury=True,
        )
        result = algo_agent.play_game(env, game_action_class)

        # If the algorithmic agent didn't fully solve, try computer-use
        # on unsolved levels (future enhancement)
        if not result.get("won") and self.verbose:
            logger.info(f"Algo agent got {result['levels_completed']}/"
                        f"{result['win_levels']}, computer-use could extend")

        return result

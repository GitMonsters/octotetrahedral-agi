"""
Action Planner — Plan efficient action sequences to reach goals.

Uses the inferred world model to pathfind toward goals while
avoiding obstacles, minimizing total actions (action efficiency).
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from collections import deque

from arc3.reasoning import RuleInferenceEngine


class ActionPlanner:
    """Plans action sequences based on inferred world model."""

    def __init__(self, reasoning: RuleInferenceEngine):
        self.reasoning = reasoning
        self.current_plan: list[int] = []
        self.plan_step: int = 0

    def plan_path_to_goal(self, frame: np.ndarray,
                          available_actions: list[int]) -> list[int]:
        """Plan a sequence of actions to reach the goal."""
        player_pos = self._find_player(frame)
        goal_pos = self._find_goal(frame)

        if player_pos is None or goal_pos is None:
            return []

        # Build action→direction map from reasoning
        action_dirs = {}
        for action in available_actions:
            direction = self.reasoning.get_movement_for_action(action)
            if direction is not None:
                action_dirs[action] = direction

        if not action_dirs:
            return []

        # BFS pathfinding on the grid
        path = self._bfs_pathfind(
            frame, player_pos, goal_pos, action_dirs
        )

        self.current_plan = path
        self.plan_step = 0
        return path

    def get_next_action(self, frame: np.ndarray,
                        available_actions: list[int]) -> Optional[int]:
        """Get the next planned action, re-planning if needed."""
        # If we have a plan, follow it
        if self.plan_step < len(self.current_plan):
            action = self.current_plan[self.plan_step]
            self.plan_step += 1
            if action in available_actions:
                return action

        # Re-plan
        plan = self.plan_path_to_goal(frame, available_actions)
        if plan:
            self.plan_step = 1
            return plan[0]

        return None

    def _find_player(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """Find player position on the frame."""
        if self.reasoning.player_color is None:
            return None

        positions = list(zip(*np.where(frame == self.reasoning.player_color)))
        if not positions:
            return None

        # Return center of player pixels
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        return (int(np.mean(rows)), int(np.mean(cols)))

    def _find_goal(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """Find goal position on the frame."""
        if self.reasoning.goal_color is None:
            # Try to find the most likely goal: smallest non-player, non-bg object
            bg = self._get_background(frame)
            player = self.reasoning.player_color

            color_counts = {}
            for val in np.unique(frame):
                val = int(val)
                if val != bg and val != player:
                    count = int(np.sum(frame == val))
                    color_counts[val] = count

            if not color_counts:
                return None

            # Goal is often a smaller, distinct object
            goal_color = min(color_counts, key=color_counts.get)
        else:
            goal_color = self.reasoning.goal_color

        positions = list(zip(*np.where(frame == goal_color)))
        if not positions:
            return None

        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        return (int(np.mean(rows)), int(np.mean(cols)))

    def _bfs_pathfind(self, frame: np.ndarray,
                      start: tuple[int, int], goal: tuple[int, int],
                      action_dirs: dict[int, tuple[int, int]]) -> list[int]:
        """BFS pathfinding using available movement actions."""
        # Simplified grid-level pathfinding
        # Compute direction needed
        dr = goal[0] - start[0]
        dc = goal[1] - start[1]

        # Find which actions move in needed directions
        path = []

        # Movement step size (infer from frame changes)
        step_size = self._estimate_step_size(frame)

        # Greedy approach: move toward goal
        remaining_r = dr
        remaining_c = dc

        max_steps = 200

        for _ in range(max_steps):
            if abs(remaining_r) < step_size and abs(remaining_c) < step_size:
                break

            best_action = None
            best_reduction = 0

            for action, (adr, adc) in action_dirs.items():
                # How much does this action reduce distance?
                new_r = remaining_r - adr * step_size
                new_c = remaining_c - adc * step_size
                reduction = (abs(remaining_r) + abs(remaining_c)) - (abs(new_r) + abs(new_c))

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_action = action

            if best_action is None:
                break

            path.append(best_action)
            adr, adc = action_dirs[best_action]
            remaining_r -= adr * step_size
            remaining_c -= adc * step_size

        return path

    def _estimate_step_size(self, frame: np.ndarray) -> int:
        """Estimate how many pixels one action moves."""
        # Default to a reasonable step size based on frame resolution
        # Can be refined from observation
        if self.reasoning.action_effects:
            for effects in self.reasoning.action_effects.values():
                changes = [e['pixels_changed'] for e in effects
                          if 0 < e['pixels_changed'] < 500]
                if changes:
                    # Step size is roughly sqrt(avg pixel change / 2)
                    avg = np.mean(changes)
                    return max(1, int(np.sqrt(avg / 2)))
        return 4  # Default step size

    def _get_background(self, frame: np.ndarray) -> int:
        values, counts = np.unique(frame, return_counts=True)
        return int(values[np.argmax(counts)])

    def invalidate_plan(self):
        """Invalidate current plan (e.g., after unexpected state change)."""
        self.current_plan = []
        self.plan_step = 0

    def reset(self):
        """Reset planner for new game."""
        self.current_plan = []
        self.plan_step = 0

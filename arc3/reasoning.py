"""
Rule Inference Engine — Hypothesize and test game rules from observations.

Builds a world model by observing how actions affect the game state,
identifying causal relationships, and refining hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Hypothesis:
    """A hypothesis about a game rule."""
    rule_type: str  # 'movement', 'interaction', 'goal', 'obstacle', 'sequence'
    description: str
    confidence: float  # 0.0 to 1.0
    evidence_for: int = 0
    evidence_against: int = 0
    action: Optional[int] = None
    details: dict = field(default_factory=dict)

    def update(self, supports: bool):
        if supports:
            self.evidence_for += 1
        else:
            self.evidence_against += 1
        total = self.evidence_for + self.evidence_against
        self.confidence = self.evidence_for / total if total > 0 else 0.0


class RuleInferenceEngine:
    """Infers game rules from observed action→effect patterns."""

    def __init__(self):
        self.hypotheses: list[Hypothesis] = []
        self.action_effects: dict[int, list[dict]] = {}  # action -> list of observed effects
        self.movement_map: dict[int, Optional[tuple[int, int]]] = {}  # action -> (dr, dc)
        self.goal_color: Optional[int] = None
        self.obstacle_colors: set[int] = set()
        self.player_color: Optional[int] = None
        self.level_advance_conditions: list[dict] = []
        self.frame_history: list[np.ndarray] = []

    def observe(self, action: int, prev_frame: np.ndarray, curr_frame: np.ndarray,
                levels_before: int, levels_after: int, game_state: str):
        """Process an observation to update hypotheses."""
        diff = curr_frame.astype(int) - prev_frame.astype(int)
        changed_mask = curr_frame != prev_frame
        pixels_changed = int(np.sum(changed_mask))

        # Store observation
        if action not in self.action_effects:
            self.action_effects[action] = []

        effect = {
            'pixels_changed': pixels_changed,
            'level_advance': levels_after > levels_before,
            'game_over': game_state == 'GAME_OVER',
            'changed_mask': changed_mask,
        }
        self.action_effects[action].append(effect)

        # Keep limited frame history
        if len(self.frame_history) < 100:
            self.frame_history.append(curr_frame.copy())

        # Infer movement
        if pixels_changed > 0 and pixels_changed < 500:
            self._infer_movement(action, prev_frame, curr_frame, changed_mask)

        # Infer player identity
        if self.player_color is None and pixels_changed > 0:
            self._infer_player(prev_frame, curr_frame, changed_mask)

        # Infer goal conditions
        if levels_after > levels_before:
            self._infer_goal(prev_frame, curr_frame, action, levels_before, levels_after)

        # Infer obstacles
        if game_state == 'GAME_OVER':
            self._infer_obstacles(prev_frame, curr_frame, action)

    def _infer_movement(self, action: int, prev_frame: np.ndarray,
                        curr_frame: np.ndarray, changed_mask: np.ndarray):
        """Infer which direction an action moves the player."""
        # Find regions that disappeared and appeared
        changed_rows, changed_cols = np.where(changed_mask)
        if len(changed_rows) == 0:
            return

        # Find pixels that appeared (new non-background) vs disappeared
        bg = self._get_background(prev_frame)
        appeared = []
        disappeared = []

        for r, c in zip(changed_rows, changed_cols):
            prev_val = int(prev_frame[r, c])
            curr_val = int(curr_frame[r, c])
            if prev_val == bg and curr_val != bg:
                appeared.append((r, c))
            elif prev_val != bg and curr_val == bg:
                disappeared.append((r, c))

        if appeared and disappeared:
            # Movement = center of appeared - center of disappeared
            app_r = np.mean([p[0] for p in appeared])
            app_c = np.mean([p[1] for p in appeared])
            dis_r = np.mean([p[0] for p in disappeared])
            dis_c = np.mean([p[1] for p in disappeared])

            dr = int(np.sign(app_r - dis_r))
            dc = int(np.sign(app_c - dis_c))

            # Update movement map with consistency check
            if action not in self.movement_map:
                self.movement_map[action] = (dr, dc)
            elif self.movement_map[action] != (dr, dc):
                # Movement might be context-dependent
                pass

            # Update/create hypothesis
            h = self._find_hypothesis('movement', action=action)
            if h is None:
                name_map = {(0, -1): 'left', (0, 1): 'right', (-1, 0): 'up', (1, 0): 'down'}
                direction = name_map.get((dr, dc), f'({dr},{dc})')
                h = Hypothesis(
                    rule_type='movement',
                    description=f'ACTION{action} moves player {direction}',
                    confidence=0.5,
                    action=action,
                    details={'direction': (dr, dc)},
                )
                self.hypotheses.append(h)
            h.update(True)

    def _infer_player(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                      changed_mask: np.ndarray):
        """Identify the player object by what moves consistently."""
        bg = self._get_background(prev_frame)
        changed_rows, changed_cols = np.where(changed_mask)

        # Find the most common non-background color in changed pixels
        colors = {}
        for r, c in zip(changed_rows, changed_cols):
            for val in [int(prev_frame[r, c]), int(curr_frame[r, c])]:
                if val != bg:
                    colors[val] = colors.get(val, 0) + 1

        if colors:
            self.player_color = max(colors, key=colors.get)

    def _infer_goal(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                    action: int, level_before: int, level_after: int):
        """Infer what caused a level advance."""
        condition = {
            'action': action,
            'from_level': level_before,
            'to_level': level_after,
        }

        # Check what colors were near the player before advance
        if self.player_color is not None:
            player_pixels = list(zip(*np.where(prev_frame == self.player_color)))
            if player_pixels:
                pr, pc = player_pixels[0]
                # Check neighboring colors
                neighbors = set()
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < prev_frame.shape[0] and 0 <= nc < prev_frame.shape[1]:
                            val = int(prev_frame[nr, nc])
                            if val != self.player_color and val != self._get_background(prev_frame):
                                neighbors.add(val)
                condition['nearby_colors'] = list(neighbors)

                # Check what color disappeared (goal object reached)
                for color in neighbors:
                    count_before = int(np.sum(prev_frame == color))
                    count_after = int(np.sum(curr_frame == color))
                    if count_after < count_before:
                        self.goal_color = color
                        condition['goal_color'] = color

        self.level_advance_conditions.append(condition)

        h = Hypothesis(
            rule_type='goal',
            description=f'Reach goal object (color={self.goal_color}) to advance level',
            confidence=0.7,
            details=condition,
        )
        self.hypotheses.append(h)

    def _infer_obstacles(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                         action: int):
        """Infer what caused game over."""
        if self.player_color is not None:
            player_pixels = list(zip(*np.where(prev_frame == self.player_color)))
            if player_pixels:
                pr, pc = player_pixels[0]
                nearby = set()
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < prev_frame.shape[0] and 0 <= nc < prev_frame.shape[1]:
                            val = int(prev_frame[nr, nc])
                            bg = self._get_background(prev_frame)
                            if val != self.player_color and val != bg:
                                nearby.add(val)

                for color in nearby:
                    if color != self.goal_color:
                        self.obstacle_colors.add(color)

        h = Hypothesis(
            rule_type='obstacle',
            description=f'Obstacle colors: {self.obstacle_colors}',
            confidence=0.5,
            details={'obstacle_colors': list(self.obstacle_colors)},
        )
        self.hypotheses.append(h)

    def _get_background(self, frame: np.ndarray) -> int:
        """Get the background color (most common)."""
        values, counts = np.unique(frame, return_counts=True)
        return int(values[np.argmax(counts)])

    def _find_hypothesis(self, rule_type: str, action: Optional[int] = None) -> Optional[Hypothesis]:
        """Find existing hypothesis matching criteria."""
        for h in self.hypotheses:
            if h.rule_type == rule_type and (action is None or h.action == action):
                return h
        return None

    def get_movement_for_action(self, action: int) -> Optional[tuple[int, int]]:
        """Get the inferred movement direction for an action."""
        return self.movement_map.get(action)

    def get_confident_rules(self, min_confidence: float = 0.5) -> list[Hypothesis]:
        """Get rules we're confident about."""
        return [h for h in self.hypotheses if h.confidence >= min_confidence]

    def get_goal_info(self) -> dict:
        """Get current understanding of the goal."""
        return {
            'goal_color': self.goal_color,
            'obstacle_colors': list(self.obstacle_colors),
            'player_color': self.player_color,
            'level_conditions': self.level_advance_conditions,
        }

    def get_world_model(self) -> dict:
        """Get full inferred world model."""
        return {
            'movement_map': {k: v for k, v in self.movement_map.items()},
            'player_color': self.player_color,
            'goal_color': self.goal_color,
            'obstacle_colors': list(self.obstacle_colors),
            'confident_rules': [
                {'type': h.rule_type, 'desc': h.description, 'conf': round(h.confidence, 2)}
                for h in self.get_confident_rules()
            ],
        }

    def reset(self):
        """Reset for new game."""
        self.hypotheses.clear()
        self.action_effects.clear()
        self.movement_map.clear()
        self.goal_color = None
        self.obstacle_colors.clear()
        self.player_color = None
        self.level_advance_conditions.clear()
        self.frame_history.clear()

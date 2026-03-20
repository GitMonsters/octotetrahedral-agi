"""
Episode Memory — Track action→observation transitions and detect patterns.

Maintains short-term (current episode) and long-term (cross-episode) memory
to learn from experience and avoid repeating failed strategies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class Transition:
    """A single action→observation transition."""
    step: int
    action: int  # GameAction value
    state_hash: int  # Hash of frame before action
    next_state_hash: int  # Hash of frame after action
    pixels_changed: int
    levels_before: int
    levels_after: int
    game_state: str  # NOT_FINISHED, WIN, GAME_OVER
    frame_diff_summary: dict = field(default_factory=dict)


@dataclass
class ActionEffect:
    """Aggregated statistics for an action's effects."""
    action: int
    times_used: int = 0
    total_pixel_change: int = 0
    level_advances: int = 0
    game_overs: int = 0
    unique_states_reached: int = 0
    avg_pixel_change: float = 0.0
    states_seen: set = field(default_factory=set)


@dataclass
class Pattern:
    """A detected action sequence pattern."""
    actions: tuple[int, ...]
    occurrences: int
    success_rate: float  # How often this leads to level advance
    avg_efficiency: float  # Average actions to advance


class EpisodeMemory:
    """Tracks transitions and learns patterns across episodes."""

    def __init__(self, max_history: int = 5000):
        self.transitions: deque[Transition] = deque(maxlen=max_history)
        self.action_effects: dict[int, ActionEffect] = {}
        self.episode_count: int = 0
        self.current_episode: list[Transition] = []
        self.state_visit_counts: dict[int, int] = {}
        self.successful_sequences: list[list[int]] = []
        self.failed_sequences: list[list[int]] = []
        self.state_action_values: dict[tuple[int, int], float] = {}
        self.level_transitions: list[dict] = []  # When levels advance

    def record(self, step: int, action: int, state_hash: int,
               next_state_hash: int, pixels_changed: int,
               levels_before: int, levels_after: int,
               game_state: str) -> Transition:
        """Record a transition."""
        t = Transition(
            step=step,
            action=action,
            state_hash=state_hash,
            next_state_hash=next_state_hash,
            pixels_changed=pixels_changed,
            levels_before=levels_before,
            levels_after=levels_after,
            game_state=game_state,
        )

        self.transitions.append(t)
        self.current_episode.append(t)

        # Track state visits
        self.state_visit_counts[next_state_hash] = \
            self.state_visit_counts.get(next_state_hash, 0) + 1

        # Update action effects
        if action not in self.action_effects:
            self.action_effects[action] = ActionEffect(action=action)
        ae = self.action_effects[action]
        ae.times_used += 1
        ae.total_pixel_change += pixels_changed
        ae.avg_pixel_change = ae.total_pixel_change / ae.times_used
        ae.states_seen.add(next_state_hash)
        ae.unique_states_reached = len(ae.states_seen)

        if levels_after > levels_before:
            ae.level_advances += 1
            self.level_transitions.append({
                'step': step,
                'action': action,
                'from_level': levels_before,
                'to_level': levels_after,
                'episode': self.episode_count,
                'actions_in_episode': len(self.current_episode),
            })

        if game_state == 'GAME_OVER':
            ae.game_overs += 1

        # Update Q-like state-action values
        reward = 0.0
        if levels_after > levels_before:
            reward = 1.0
        elif game_state == 'GAME_OVER':
            reward = -0.5
        elif pixels_changed > 0:
            reward = 0.01  # Small reward for making something happen

        key = (state_hash, action)
        alpha = 0.1
        old_val = self.state_action_values.get(key, 0.0)
        self.state_action_values[key] = old_val + alpha * (reward - old_val)

        return t

    def end_episode(self, success: bool):
        """Mark end of an episode (game over or win)."""
        actions = [t.action for t in self.current_episode]
        if success:
            self.successful_sequences.append(actions)
        else:
            self.failed_sequences.append(actions)

        self.current_episode = []
        self.episode_count += 1

    def get_action_ranking(self, state_hash: int,
                           available_actions: list[int]) -> list[tuple[int, float]]:
        """Rank available actions by expected value for given state."""
        rankings = []
        for action in available_actions:
            key = (state_hash, action)
            value = self.state_action_values.get(key, 0.0)

            # Bonus for exploration (less-visited state-action pairs)
            ae = self.action_effects.get(action)
            if ae:
                exploration_bonus = 0.1 / (1 + ae.times_used ** 0.5)
                value += exploration_bonus

            rankings.append((action, value))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def is_stuck(self, lookback: int = 20) -> bool:
        """Check if we're in a loop (repeating states)."""
        if len(self.current_episode) < lookback:
            return False

        recent = self.current_episode[-lookback:]
        state_hashes = [t.next_state_hash for t in recent]
        unique_states = len(set(state_hashes))
        return unique_states < lookback * 0.3  # Less than 30% unique states

    def get_unexplored_actions(self, state_hash: int,
                                available_actions: list[int]) -> list[int]:
        """Get actions not yet tried from this state."""
        tried = set()
        for t in self.transitions:
            if t.state_hash == state_hash:
                tried.add(t.action)
        return [a for a in available_actions if a not in tried]

    def get_effective_actions(self) -> list[int]:
        """Get actions that have historically caused level advances."""
        effective = []
        for action, ae in self.action_effects.items():
            if ae.level_advances > 0:
                effective.append(action)
        return effective

    def get_dangerous_actions(self, state_hash: int) -> set[int]:
        """Get actions that led to GAME_OVER from similar states."""
        dangerous = set()
        for t in self.transitions:
            if t.game_state == 'GAME_OVER':
                dangerous.add(t.action)
        return dangerous

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'total_transitions': len(self.transitions),
            'episodes': self.episode_count,
            'unique_states': len(self.state_visit_counts),
            'successful_episodes': len(self.successful_sequences),
            'failed_episodes': len(self.failed_sequences),
            'level_transitions': len(self.level_transitions),
            'action_effects': {
                a: {
                    'used': ae.times_used,
                    'avg_change': round(ae.avg_pixel_change, 1),
                    'advances': ae.level_advances,
                    'game_overs': ae.game_overs,
                }
                for a, ae in self.action_effects.items()
            },
        }

    def reset(self):
        """Full reset for new game."""
        self.transitions.clear()
        self.action_effects.clear()
        self.current_episode.clear()
        self.state_visit_counts.clear()
        self.successful_sequences.clear()
        self.failed_sequences.clear()
        self.state_action_values.clear()
        self.level_transitions.clear()
        self.episode_count = 0

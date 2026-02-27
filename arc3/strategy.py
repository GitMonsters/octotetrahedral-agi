"""
MetaCognition / Strategy Module — Manages exploration vs exploitation,
strategy switching, and self-evaluation.

Decides when to explore (try new actions), exploit (follow known strategies),
or switch strategy entirely when stuck.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Optional

from arc3.memory import EpisodeMemory
from arc3.reasoning import RuleInferenceEngine


class Strategy(Enum):
    EXPLORE_SYSTEMATIC = "explore_systematic"  # Try each action methodically
    EXPLORE_RANDOM = "explore_random"          # Random exploration
    EXPLOIT_PLAN = "exploit_plan"              # Follow planner's path
    EXPLOIT_REPEAT = "exploit_repeat"          # Repeat successful sequence
    RECOVER = "recover"                        # Recovering from game over


class MetaCognitionModule:
    """Manages high-level strategy selection and switching."""

    def __init__(self, memory: EpisodeMemory, reasoning: RuleInferenceEngine):
        self.memory = memory
        self.reasoning = reasoning
        self.current_strategy = Strategy.EXPLORE_SYSTEMATIC
        self.strategy_history: list[Strategy] = []
        self.explore_index: int = 0
        self.exploit_sequence: list[int] = []
        self.exploit_step: int = 0
        self.steps_in_strategy: int = 0
        self.max_steps_before_switch: int = 50
        self.exploration_phase_complete: bool = False

    def select_action(self, available_actions: list[int],
                      state_hash: int,
                      planned_action: Optional[int] = None) -> int:
        """Select action based on current strategy."""
        self.steps_in_strategy += 1

        # Check if we should switch strategy
        self._maybe_switch_strategy(available_actions, state_hash)

        if self.current_strategy == Strategy.EXPLORE_SYSTEMATIC:
            return self._systematic_explore(available_actions, state_hash)

        elif self.current_strategy == Strategy.EXPLORE_RANDOM:
            return self._random_explore(available_actions)

        elif self.current_strategy == Strategy.EXPLOIT_PLAN:
            if planned_action is not None and planned_action in available_actions:
                return planned_action
            return self._fallback_action(available_actions, state_hash)

        elif self.current_strategy == Strategy.EXPLOIT_REPEAT:
            return self._repeat_successful(available_actions)

        elif self.current_strategy == Strategy.RECOVER:
            return self._recover_action(available_actions, state_hash)

        return random.choice(available_actions)

    def _maybe_switch_strategy(self, available_actions: list[int], state_hash: int):
        """Decide if we should switch strategy."""
        # If stuck in a loop, switch strategy
        if self.memory.is_stuck():
            self._switch_to(Strategy.EXPLORE_RANDOM)
            return

        # If we've explored enough and have a world model, switch to exploit
        if (self.current_strategy in (Strategy.EXPLORE_SYSTEMATIC, Strategy.EXPLORE_RANDOM)
                and self.steps_in_strategy > self.max_steps_before_switch):
            # Do we have enough info to plan?
            if self.reasoning.movement_map and (self.reasoning.goal_color or self.reasoning.player_color):
                self._switch_to(Strategy.EXPLOIT_PLAN)
                self.exploration_phase_complete = True
                return
            else:
                # Haven't learned enough, try random exploration
                if self.current_strategy == Strategy.EXPLORE_SYSTEMATIC:
                    self._switch_to(Strategy.EXPLORE_RANDOM)
                else:
                    self._switch_to(Strategy.EXPLORE_SYSTEMATIC)
                return

        # If exploiting and not making progress, go back to explore
        if (self.current_strategy == Strategy.EXPLOIT_PLAN
                and self.steps_in_strategy > 30
                and not self._making_progress()):
            self._switch_to(Strategy.EXPLORE_RANDOM)
            return

        # If we have a successful sequence, try repeating it
        if (self.memory.successful_sequences
                and self.current_strategy != Strategy.EXPLOIT_REPEAT
                and self.steps_in_strategy > 20):
            self._switch_to(Strategy.EXPLOIT_REPEAT)

    def _switch_to(self, strategy: Strategy):
        """Switch to a new strategy."""
        if strategy == self.current_strategy:
            return
        self.strategy_history.append(self.current_strategy)
        self.current_strategy = strategy
        self.steps_in_strategy = 0
        self.explore_index = 0

        if strategy == Strategy.EXPLOIT_REPEAT and self.memory.successful_sequences:
            self.exploit_sequence = self.memory.successful_sequences[-1]
            self.exploit_step = 0

    def _systematic_explore(self, available_actions: list[int], state_hash: int) -> int:
        """Try each action systematically to learn effects."""
        # First try unexplored actions from this state
        unexplored = self.memory.get_unexplored_actions(state_hash, available_actions)
        if unexplored:
            return unexplored[0]

        # Otherwise cycle through actions
        action = available_actions[self.explore_index % len(available_actions)]
        self.explore_index += 1
        return action

    def _random_explore(self, available_actions: list[int]) -> int:
        """Random action selection for exploration."""
        return random.choice(available_actions)

    def _repeat_successful(self, available_actions: list[int]) -> int:
        """Repeat a previously successful action sequence."""
        if self.exploit_step < len(self.exploit_sequence):
            action = self.exploit_sequence[self.exploit_step]
            self.exploit_step += 1
            if action in available_actions:
                return action

        # Sequence exhausted or action unavailable
        self._switch_to(Strategy.EXPLOIT_PLAN)
        return random.choice(available_actions)

    def _recover_action(self, available_actions: list[int], state_hash: int) -> int:
        """Choose action during recovery from game over."""
        # Avoid actions that caused game over
        dangerous = self.memory.get_dangerous_actions(state_hash)
        safe = [a for a in available_actions if a not in dangerous]
        if safe:
            return random.choice(safe)
        return random.choice(available_actions)

    def _fallback_action(self, available_actions: list[int], state_hash: int) -> int:
        """Fallback when planned action isn't available."""
        # Use memory-based ranking
        rankings = self.memory.get_action_ranking(state_hash, available_actions)
        if rankings:
            return rankings[0][0]
        return random.choice(available_actions)

    def _making_progress(self) -> bool:
        """Check if current strategy is making progress."""
        if len(self.memory.current_episode) < 10:
            return True

        recent = self.memory.current_episode[-10:]
        unique_states = len(set(t.next_state_hash for t in recent))
        any_progress = any(t.levels_after > t.levels_before for t in recent)
        return any_progress or unique_states > 5

    def notify_game_over(self):
        """Called when game ends (game over or win)."""
        self._switch_to(Strategy.RECOVER)

    def notify_level_advance(self):
        """Called when a level is completed."""
        # Good progress, keep current strategy but reset counter
        self.steps_in_strategy = 0

    def get_status(self) -> dict:
        """Get current strategy status."""
        return {
            'strategy': self.current_strategy.value,
            'steps_in_strategy': self.steps_in_strategy,
            'exploration_complete': self.exploration_phase_complete,
            'strategy_switches': len(self.strategy_history),
        }

    def reset(self):
        """Reset for new game."""
        self.current_strategy = Strategy.EXPLORE_SYSTEMATIC
        self.strategy_history.clear()
        self.explore_index = 0
        self.exploit_sequence = []
        self.exploit_step = 0
        self.steps_in_strategy = 0
        self.exploration_phase_complete = False

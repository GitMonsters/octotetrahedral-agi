"""
Planning Limb - Goal-directed planning and decision making
Inspired by octopus problem-solving capabilities

Biological insight:
- Octopuses show remarkable problem-solving abilities
- Can plan multi-step actions (opening jars, navigating mazes)
- Demonstrate tool use and forward thinking
- Prefrontal-like planning without a cortex

Our implementation:
- Hierarchical goal representation
- Action sequence planning
- Tree search for plan generation
- Reward prediction for plan evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from collections import deque
import heapq

from .base_limb import BaseLimb


class Goal:
    """Represents a goal with hierarchical structure."""
    
    def __init__(
        self,
        name: str,
        target_state: torch.Tensor,
        priority: float = 1.0,
        parent: Optional['Goal'] = None
    ):
        self.name = name
        self.target_state = target_state
        self.priority = priority
        self.parent = parent
        self.subgoals: List['Goal'] = []
        self.progress = 0.0
        self.completed = False
    
    def add_subgoal(self, subgoal: 'Goal'):
        """Add a subgoal."""
        subgoal.parent = self
        self.subgoals.append(subgoal)
    
    def update_progress(self, current_state: torch.Tensor) -> float:
        """Update and return progress toward goal."""
        if self.target_state is not None:
            distance = (current_state - self.target_state).norm().item()
            max_distance = self.target_state.norm().item() + 1e-6
            self.progress = max(0, 1 - distance / max_distance)
        
        if self.progress > 0.95:
            self.completed = True
        
        return self.progress


class ActionSequence:
    """Represents a planned sequence of actions."""
    
    def __init__(self):
        self.actions: List[int] = []
        self.expected_rewards: List[float] = []
        self.expected_states: List[torch.Tensor] = []
        self.confidence: float = 1.0
    
    def add_step(
        self,
        action: int,
        expected_reward: float,
        expected_state: torch.Tensor
    ):
        """Add a step to the plan."""
        self.actions.append(action)
        self.expected_rewards.append(expected_reward)
        self.expected_states.append(expected_state)
    
    def total_expected_reward(self, gamma: float = 0.99) -> float:
        """Compute discounted total expected reward."""
        total = 0.0
        for i, r in enumerate(self.expected_rewards):
            total += (gamma ** i) * r
        return total
    
    def __len__(self) -> int:
        return len(self.actions)


class WorldModel(nn.Module):
    """
    Simple learned world model for planning.
    Predicts next state and reward given current state and action.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_actions: int = 10
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        
        # State-action encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Next state predictor
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.
        
        Args:
            state: Current state [batch, hidden_dim]
            action: Action indices [batch]
            
        Returns:
            (predicted_next_state, predicted_reward)
        """
        action_emb = self.action_embed(action)
        combined = torch.cat([state, action_emb], dim=-1)
        encoded = self.encoder(combined)
        
        next_state = state + self.state_predictor(encoded)  # Residual
        reward = self.reward_predictor(encoded)
        
        return next_state, reward


class PlanningLimb(BaseLimb):
    """
    Planning Limb for goal-directed behavior and decision making.
    
    Capabilities:
    1. Hierarchical goal management
    2. Action sequence planning
    3. World model for simulation
    4. Plan evaluation and selection
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_actions: int = 10,
        planning_horizon: int = 10,
        num_plans: int = 5,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100
    ):
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="planning"
        )
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        self.num_plans = num_plans
        
        # World model for planning
        self.world_model = WorldModel(hidden_dim, num_actions)
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Plan encoder (processes action sequences)
        self.plan_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Policy head (action probabilities given state and goal)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value head (expected return from state)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Goal hierarchy
        self.active_goals: List[Goal] = []
        self.max_goals = 10
        
        # Plan storage
        self.current_plans: List[ActionSequence] = []
        
        # Stats
        self._plans_generated = 0
        self._goals_completed = 0
    
    def process(
        self,
        x: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        do_planning: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input with planning.
        
        Args:
            x: Input state [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            goal_state: Optional goal state [batch, hidden_dim]
            do_planning: Whether to generate plans
            
        Returns:
            Planning-enhanced state [batch, seq_len, hidden_dim]
        """
        # Handle sequence input
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            state = x.mean(dim=1)  # Pool to single state
        else:
            batch_size = x.size(0)
            seq_len = 1
            state = x
        
        # Encode goal if provided
        if goal_state is not None:
            goal_encoded = self.goal_encoder(goal_state)
        else:
            goal_encoded = torch.zeros_like(state)
        
        # Generate policy logits
        combined = torch.cat([state, goal_encoded], dim=-1)
        policy_logits = self.policy_head(combined)
        
        # Generate plans if requested
        if do_planning and self.training is False:
            self._generate_plans(state, goal_encoded)
        
        # Value estimate
        value = self.value_head(state)
        
        # Compute planning-enhanced representation
        # Blend state with goal direction
        enhanced = state + 0.1 * goal_encoded
        
        # Expand back to sequence if needed
        if x.dim() == 3:
            enhanced = enhanced.unsqueeze(1).expand(-1, seq_len, -1)
        
        return enhanced
    
    def forward(
        self,
        x: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
        return_policy: bool = False,
        return_value: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[Dict]]:
        """
        Forward pass through planning limb.
        """
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Planning processing
        output = self.process(adapted, goal_state=goal_state, **kwargs)
        
        # Confidence
        confidence = None
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
        
        # Additional outputs
        extras = {}
        if return_policy or return_value:
            state = adapted.mean(dim=1) if adapted.dim() == 3 else adapted
            goal = goal_state if goal_state is not None else torch.zeros_like(state)
            
            if return_policy:
                combined = torch.cat([state, goal], dim=-1)
                extras['policy_logits'] = self.policy_head(combined)
            
            if return_value:
                extras['value'] = self.value_head(state)
        
        return output, confidence, extras if extras else None
    
    def _generate_plans(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        num_plans: Optional[int] = None
    ):
        """Generate candidate plans via tree search."""
        num_plans = num_plans or self.num_plans
        self.current_plans = []
        
        with torch.no_grad():
            for _ in range(num_plans):
                plan = ActionSequence()
                current_state = state[0].clone()  # Single batch element
                
                for step in range(self.planning_horizon):
                    # Sample action from policy
                    combined = torch.cat([current_state, goal[0]], dim=-1)
                    logits = self.policy_head(combined.unsqueeze(0))
                    probs = F.softmax(logits, dim=-1)
                    
                    # Add exploration noise
                    probs = 0.8 * probs + 0.2 * torch.ones_like(probs) / self.num_actions
                    action = torch.multinomial(probs[0], 1).item()
                    
                    # Predict next state and reward
                    action_tensor = torch.tensor([action], device=state.device)
                    next_state, reward = self.world_model(
                        current_state.unsqueeze(0),
                        action_tensor
                    )
                    
                    plan.add_step(action, reward.item(), next_state[0])
                    current_state = next_state[0]
                
                # Score plan
                plan.confidence = self._evaluate_plan(plan, goal[0])
                self.current_plans.append(plan)
        
        # Sort by expected value
        self.current_plans.sort(
            key=lambda p: p.total_expected_reward() * p.confidence,
            reverse=True
        )
        
        self._plans_generated += num_plans
    
    def _evaluate_plan(self, plan: ActionSequence, goal: torch.Tensor) -> float:
        """Evaluate a plan's quality."""
        if len(plan.expected_states) == 0:
            return 0.0
        
        # Distance to goal at end of plan
        final_state = plan.expected_states[-1]
        distance = (final_state - goal).norm().item()
        proximity_score = max(0, 1 - distance / (goal.norm().item() + 1e-6))
        
        return proximity_score
    
    def get_best_action(self) -> Optional[int]:
        """Get the best action from current plans."""
        if not self.current_plans:
            return None
        return self.current_plans[0].actions[0] if self.current_plans[0].actions else None
    
    def add_goal(
        self,
        name: str,
        target_state: torch.Tensor,
        priority: float = 1.0
    ) -> Goal:
        """Add a new goal."""
        goal = Goal(name, target_state, priority)
        
        # Insert maintaining priority order
        inserted = False
        for i, g in enumerate(self.active_goals):
            if priority > g.priority:
                self.active_goals.insert(i, goal)
                inserted = True
                break
        
        if not inserted:
            self.active_goals.append(goal)
        
        # Limit number of active goals
        if len(self.active_goals) > self.max_goals:
            self.active_goals = self.active_goals[:self.max_goals]
        
        return goal
    
    def update_goals(self, current_state: torch.Tensor):
        """Update progress on all active goals."""
        state = current_state[0] if current_state.dim() > 1 else current_state
        
        completed = []
        for goal in self.active_goals:
            goal.update_progress(state)
            if goal.completed:
                completed.append(goal)
                self._goals_completed += 1
        
        # Remove completed goals
        for goal in completed:
            self.active_goals.remove(goal)
    
    def get_highest_priority_goal(self) -> Optional[Goal]:
        """Get the highest priority active goal."""
        if self.active_goals:
            return self.active_goals[0]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planning limb statistics."""
        stats = super().get_stats()
        stats.update({
            'active_goals': len(self.active_goals),
            'current_plans': len(self.current_plans),
            'plans_generated': self._plans_generated,
            'goals_completed': self._goals_completed,
            'best_plan_value': (
                self.current_plans[0].total_expected_reward()
                if self.current_plans else 0.0
            )
        })
        return stats


if __name__ == "__main__":
    print("Testing PlanningLimb...")
    
    # Create limb
    limb = PlanningLimb(
        hidden_dim=256,
        num_actions=10,
        planning_horizon=5,
        num_plans=3
    )
    limb.eval()  # Enable planning
    
    # Test input
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 256)
    goal = torch.randn(batch_size, 256)
    
    # Forward pass
    output, confidence, extras = limb(
        x,
        goal_state=goal,
        return_confidence=True,
        return_policy=True,
        return_value=True
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Policy logits shape: {extras['policy_logits'].shape}")
    print(f"Value shape: {extras['value'].shape}")
    
    # Test goal management
    goal_tensor = torch.randn(256)
    g = limb.add_goal("test_goal", goal_tensor, priority=0.8)
    print(f"\nActive goals: {len(limb.active_goals)}")
    
    # Update goals
    current = torch.randn(256)
    limb.update_goals(current)
    
    # Get best action
    best_action = limb.get_best_action()
    print(f"Best action: {best_action}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nPlanning Limb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nPlanningLimb tests passed!")

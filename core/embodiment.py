"""Embodiment interface for TranscendPlexity multi-modal AGI.

Abstract embodiment layer providing:
- Observation space (proprioception, vision, touch, audio)
- Action space (continuous motor commands + discrete decisions)
- Reward/feedback signal for learning
- Compatible with MuJoCo, PyBullet, Isaac Gym, or custom simulators

Extends the existing sensorimotor_loop.py with multi-modal grounding.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionType(Enum):
    CONTINUOUS = "continuous"   # Motor commands (joint angles, velocities)
    DISCRETE = "discrete"      # Categorical choices (grasp/release, left/right)
    HYBRID = "hybrid"          # Both continuous and discrete


@dataclass
class ObservationSpace:
    """Defines what the agent can perceive."""
    proprioception_dim: int = 32   # Joint angles, velocities, forces
    touch_dim: int = 64            # Tactile sensor array
    vision_enabled: bool = True    # Uses VisionEncoder
    audio_enabled: bool = True     # Uses AudioEncoder
    goal_dim: int = 64             # Task goal embedding


@dataclass
class ActionSpace:
    """Defines what the agent can do."""
    continuous_dim: int = 32       # Motor commands (e.g., 7-DOF arm + gripper)
    discrete_choices: int = 16     # Categorical actions
    action_type: ActionType = ActionType.HYBRID
    max_force: float = 1.0         # Action magnitude clamp


@dataclass
class EmbodimentConfig:
    """Full embodiment configuration."""
    hidden_dim: int = 256
    observation: ObservationSpace = field(default_factory=ObservationSpace)
    action: ActionSpace = field(default_factory=ActionSpace)
    num_layers: int = 3
    num_heads: int = 8
    history_length: int = 32       # Frames of proprioceptive history
    prediction_horizon: int = 8    # Steps of action prediction
    dropout: float = 0.1


class ProprioceptionEncoder(nn.Module):
    """Encode proprioceptive state (joint angles, velocities, forces)."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TouchEncoder(nn.Module):
    """Encode tactile sensor data."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, touch: torch.Tensor) -> torch.Tensor:
        return self.net(touch)


class ActionDecoder(nn.Module):
    """Decode hidden state into actions.

    Supports continuous (motor), discrete (choice), or hybrid actions.
    """

    def __init__(self, config: EmbodimentConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim

        # Continuous action head (motor commands)
        if config.action.action_type in (ActionType.CONTINUOUS, ActionType.HYBRID):
            self.continuous_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, config.action.continuous_dim),
                nn.Tanh(),  # Bounded actions [-1, 1]
            )

        # Discrete action head (categorical choices)
        if config.action.action_type in (ActionType.DISCRETE, ActionType.HYBRID):
            self.discrete_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, config.action.discrete_choices),
            )

        # Action value estimate (for RL)
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """Decode actions from hidden state.

        Args:
            hidden: [B, D] hidden state from embodiment transformer

        Returns:
            dict with 'continuous', 'discrete_logits', 'value'
        """
        result = {}

        if hasattr(self, 'continuous_head'):
            cont = self.continuous_head(hidden)
            result['continuous'] = cont * self.config.action.max_force

        if hasattr(self, 'discrete_head'):
            result['discrete_logits'] = self.discrete_head(hidden)
            result['discrete_action'] = result['discrete_logits'].argmax(dim=-1)

        result['value'] = self.value_head(hidden).squeeze(-1)

        return result


class EmbodimentTransformer(nn.Module):
    """Transformer that processes multi-modal embodied observations."""

    def __init__(self, dim: int = 256, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EmbodimentInterface(nn.Module):
    """Full embodiment module for TranscendPlexity.

    Processes multi-modal observations (proprioception, touch, vision, audio)
    and produces actions. Designed to plug into the compound braid alongside
    the 8 cognitive limbs.

    The embodiment layer maintains a proprioceptive history buffer for
    temporal reasoning about body state.
    """

    def __init__(self, config: Optional[EmbodimentConfig] = None):
        super().__init__()
        self.config = config or EmbodimentConfig()
        dim = self.config.hidden_dim

        # Sensory encoders
        self.proprio_encoder = ProprioceptionEncoder(
            self.config.observation.proprioception_dim, dim
        )
        self.touch_encoder = TouchEncoder(
            self.config.observation.touch_dim, dim
        )
        self.goal_encoder = nn.Linear(self.config.observation.goal_dim, dim)

        # Modality fusion tokens
        self.proprio_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.touch_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.goal_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.action_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Temporal positional encoding for history
        self.register_buffer(
            'temporal_pos',
            self._sinusoidal_pos(self.config.history_length + 16, dim)
        )

        # Cross-modal transformer
        self.transformer = EmbodimentTransformer(
            dim, self.config.num_layers, self.config.num_heads, self.config.dropout
        )

        # Action decoder
        self.action_decoder = ActionDecoder(self.config)

        # Projection for compound braid input
        self.braid_proj = nn.Linear(dim, dim)

        # World model head (predict next observation)
        self.world_model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.config.observation.proprioception_dim),
        )

        # History buffer (registered as non-persistent buffer)
        self.register_buffer('_history', torch.zeros(1, 0, dim), persistent=False)

    @staticmethod
    def _sinusoidal_pos(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def reset_history(self, batch_size: int = 1):
        """Reset the proprioceptive history buffer."""
        self._history = torch.zeros(
            batch_size, 0, self.config.hidden_dim,
            device=self.proprio_token.device
        )

    def forward(
        self,
        proprioception: Optional[torch.Tensor] = None,
        touch: Optional[torch.Tensor] = None,
        vision_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        cognitive_state: Optional[torch.Tensor] = None,
        return_actions: bool = True,
    ) -> dict:
        """Process embodied observations and produce actions.

        Args:
            proprioception: [B, proprio_dim] current body state
            touch: [B, touch_dim] tactile sensor data
            vision_embeddings: [B, N_v, D] from VisionEncoder
            audio_embeddings: [B, N_a, D] from AudioEncoder
            goal: [B, goal_dim] task goal embedding
            cognitive_state: [B, N_c, D] from cognitive limbs (compound braid output)
            return_actions: If True, decode actions

        Returns:
            dict with:
                'embeddings': [B, seq_len, D] for compound braid fusion
                'actions': dict with continuous/discrete/value (if return_actions)
                'world_prediction': [B, proprio_dim] predicted next state
                'body_state': [B, D] current body representation
        """
        B = proprioception.shape[0] if proprioception is not None else 1
        device = self.proprio_token.device
        tokens = []

        # Encode proprioception
        if proprioception is not None:
            proprio_emb = self.proprio_encoder(proprioception).unsqueeze(1)  # [B, 1, D]
            proprio_emb = proprio_emb + self.proprio_token
            tokens.append(proprio_emb)

            # Update history
            if self._history.shape[0] != B:
                self.reset_history(B)
            self._history = torch.cat([self._history, proprio_emb], dim=1)
            if self._history.shape[1] > self.config.history_length:
                self._history = self._history[:, -self.config.history_length:]
            # Add history with temporal position
            hist = self._history + self.temporal_pos[:, :self._history.shape[1]]
            tokens.append(hist)

        # Encode touch
        if touch is not None:
            touch_emb = self.touch_encoder(touch).unsqueeze(1)  # [B, 1, D]
            touch_emb = touch_emb + self.touch_token
            tokens.append(touch_emb)

        # Vision (pre-encoded by VisionEncoder)
        if vision_embeddings is not None:
            tokens.append(vision_embeddings)

        # Audio (pre-encoded by AudioEncoder)
        if audio_embeddings is not None:
            tokens.append(audio_embeddings)

        # Goal
        if goal is not None:
            goal_emb = self.goal_encoder(goal).unsqueeze(1)  # [B, 1, D]
            goal_emb = goal_emb + self.goal_token
            tokens.append(goal_emb)

        # Cognitive state from other limbs
        if cognitive_state is not None:
            tokens.append(cognitive_state)

        # Action query token (learns to aggregate info for action selection)
        action_q = self.action_query.expand(B, -1, -1)
        tokens.append(action_q)

        # Concatenate all tokens
        if not tokens:
            # Dummy input if nothing provided
            x = torch.zeros(B, 1, self.config.hidden_dim, device=device)
        else:
            x = torch.cat(tokens, dim=1)  # [B, total_tokens, D]

        # Transformer processing
        x = self.transformer(x)

        # Extract outputs
        action_hidden = x[:, -1]  # Last token = action query
        body_state = x[:, 0]  # First token = current state

        # Embeddings for compound braid (exclude action query)
        braid_embeddings = self.braid_proj(x[:, :-1])

        result = {
            'embeddings': braid_embeddings,
            'body_state': body_state,
            'world_prediction': self.world_model(action_hidden),
        }

        if return_actions:
            result['actions'] = self.action_decoder(action_hidden)

        return result

    def step(
        self,
        proprioception: torch.Tensor,
        touch: Optional[torch.Tensor] = None,
        vision_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> dict:
        """Single embodied step: observe → think → act.

        Convenience method for the sensorimotor loop.
        """
        return self.forward(
            proprioception=proprioception,
            touch=touch,
            vision_embeddings=vision_embeddings,
            audio_embeddings=audio_embeddings,
            goal=goal,
            return_actions=True,
        )


class SimulatorBridge(nn.Module):
    """Abstract bridge to physics simulators.

    Subclass this to connect to MuJoCo, PyBullet, Isaac Gym, etc.
    Provides the standard interface for the EmbodimentInterface.
    """

    def __init__(self, config: EmbodimentConfig):
        super().__init__()
        self.config = config

    def reset(self) -> dict:
        """Reset environment, return initial observation."""
        raise NotImplementedError

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Execute action, return (observation, reward, done, info)."""
        raise NotImplementedError

    def render(self) -> Optional[torch.Tensor]:
        """Return visual observation as [H, W, C] tensor."""
        raise NotImplementedError


class DummySimulator(SimulatorBridge):
    """Simple physics-free simulator for testing the embodiment pipeline."""

    def __init__(self, config: Optional[EmbodimentConfig] = None):
        super().__init__(config or EmbodimentConfig())
        self.position = None
        self.velocity = None
        self.step_count = 0

    def reset(self) -> dict:
        self.position = torch.zeros(self.config.observation.proprioception_dim // 2)
        self.velocity = torch.zeros(self.config.observation.proprioception_dim // 2)
        self.step_count = 0
        return self._get_obs()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        # Apply continuous action as force
        force = action.get('continuous', torch.zeros(self.config.action.continuous_dim))
        # Simple physics: F = ma → a = F, v += a*dt, x += v*dt
        dt = 0.01
        acc = force[:self.velocity.shape[0]]
        self.velocity = self.velocity + acc * dt
        self.velocity = self.velocity * 0.99  # damping
        self.position = self.position + self.velocity * dt

        self.step_count += 1

        # Reward: minimize distance to origin
        reward = -self.position.norm().item()
        done = self.step_count >= 1000

        return self._get_obs(), reward, done, {'step': self.step_count}

    def _get_obs(self) -> dict:
        proprio = torch.cat([self.position, self.velocity])
        touch = torch.randn(self.config.observation.touch_dim) * 0.1
        return {
            'proprioception': proprio.unsqueeze(0),
            'touch': touch.unsqueeze(0),
        }

"""
OctoTetrahedral AGI Configuration
Central configuration for all hyperparameters

Combines:
- Tetrahedral geometry (64-point structure)
- Octopus-inspired RNA editing (dynamic adaptation)
- Distributed limb architecture (semi-autonomous processing)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch


@dataclass
class GeometryConfig:
    """Tetrahedral geometry configuration"""
    num_points: int = 64  # Total points in tetrahedral structure
    num_vertices: int = 4  # Primary vertices
    num_edge_midpoints: int = 6  # Edge midpoints
    num_face_centers: int = 4  # Face centers
    num_edge_subdivisions: int = 24  # 4 per edge × 6 edges
    num_face_subdivisions: int = 12  # 3 per face × 4 faces
    num_internal_points: int = 14  # Internal distribution
    
    # Transformation parameters
    rotation_angle: float = 0.5236  # 30 degrees in radians
    scale_factor: float = 1.2
    shear_factor: float = 0.2


@dataclass
class ModelConfig:
    """Core model architecture configuration"""
    # Vocabulary (tiktoken cl100k_base)
    vocab_size: int = 100277
    
    # Dimensions
    hidden_dim: int = 256
    ffn_dim: int = 1024  # 4x hidden_dim
    
    # Transformer
    num_layers: int = 3
    num_heads: int = 8
    head_dim: int = 32  # hidden_dim // num_heads
    
    # Sequence
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    
    # Memory
    memory_slots: int = 4
    
    # Geometry
    num_points: int = 64


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration"""
    rank: int = 4
    alpha: float = 1.0  # Scaling factor
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        'query', 'key', 'value', 'output'
    ])


@dataclass
class RNAEditingConfig:
    """RNA editing layer configuration (octopus-inspired)"""
    # Temperature controls editing intensity
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    temperature_max: float = 5.0
    
    # Number of pathways (meta-limbs)
    num_pathways: int = 3  # Perception, Reasoning, Action
    
    # Attention head gating
    num_gated_heads: int = 8
    
    # Editing site density (fraction of parameters affected)
    editing_density: float = 0.1


@dataclass
class LimbConfig:
    """Limb (distributed processing unit) configuration"""
    # Buffer for local learning
    buffer_size: int = 100
    
    # Local learning rate (for PPO updates)
    local_lr: float = 1e-4
    
    # Confidence threshold for escalation
    confidence_threshold: float = 0.5


@dataclass
class SyncConfig:
    """Hub synchronization configuration"""
    # Sync frequency (every N gradient updates)
    sync_frequency: int = 10
    
    # Safety: maximum weight drift per limb per sync
    max_drift: float = 0.1
    
    # Rollback buffer size
    rollback_buffer_size: int = 10
    
    # Performance weighting for FedAvg
    use_performance_weighting: bool = True


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    
    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Loss weights
    prediction_loss_weight: float = 1.0
    information_gain_weight: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass 
class Config:
    """Master configuration combining all sub-configs"""
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rna_editing: RNAEditingConfig = field(default_factory=RNAEditingConfig)
    limb: LimbConfig = field(default_factory=LimbConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Device
    device: str = field(default_factory=lambda: (
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    ))
    
    # Seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration consistency"""
        assert self.model.hidden_dim % self.model.num_heads == 0, \
            f"hidden_dim ({self.model.hidden_dim}) must be divisible by num_heads ({self.model.num_heads})"
        assert self.model.head_dim == self.model.hidden_dim // self.model.num_heads, \
            f"head_dim mismatch"
        assert self.geometry.num_points == (
            self.geometry.num_vertices + 
            self.geometry.num_edge_midpoints + 
            self.geometry.num_face_centers +
            self.geometry.num_edge_subdivisions +
            self.geometry.num_face_subdivisions +
            self.geometry.num_internal_points
        ), "Point distribution must sum to num_points"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'geometry': self.geometry.__dict__,
            'model': self.model.__dict__,
            'lora': {**self.lora.__dict__, 'target_modules': self.lora.target_modules},
            'rna_editing': self.rna_editing.__dict__,
            'limb': self.limb.__dict__,
            'sync': self.sync.__dict__,
            'training': {**self.training.__dict__, 'betas': self.training.betas},
            'device': self.device,
            'seed': self.seed
        }


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config(**overrides) -> Config:
    """
    Get configuration with optional overrides
    
    Example:
        config = get_config(model={'hidden_dim': 512}, training={'batch_size': 16})
    """
    config = Config()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            sub_config = getattr(config, key)
            if isinstance(value, dict) and hasattr(sub_config, '__dict__'):
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            else:
                setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("OctoTetrahedral AGI Configuration")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Vocabulary size: {config.model.vocab_size:,}")
    print(f"Hidden dimension: {config.model.hidden_dim}")
    print(f"Number of layers: {config.model.num_layers}")
    print(f"Number of heads: {config.model.num_heads}")
    print(f"Tetrahedral points: {config.geometry.num_points}")
    print(f"Memory slots: {config.model.memory_slots}")
    print(f"LoRA rank: {config.lora.rank}")
    print(f"Sync frequency: {config.sync.sync_frequency}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Estimate parameter count
    vocab_params = config.model.vocab_size * config.model.hidden_dim * 2  # embed + output
    transformer_params = config.model.num_layers * (
        4 * config.model.hidden_dim ** 2 +  # attention
        2 * config.model.hidden_dim * config.model.ffn_dim  # FFN
    )
    memory_params = config.model.memory_slots * config.model.hidden_dim
    total_params = vocab_params + transformer_params + memory_params
    
    print(f"\nEstimated parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

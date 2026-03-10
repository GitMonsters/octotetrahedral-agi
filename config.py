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
import os
import torch


def _select_device() -> str:
    """Select a usable device; fall back if accelerators are broken."""
    forced = os.getenv("OCTO_DEVICE") or os.getenv("OCTOTETRAHEDRAL_DEVICE")
    if forced:
        return forced

    # Respect explicit CUDA disable.
    if os.getenv("CUDA_VISIBLE_DEVICES", None) == "":
        cuda_available = False
    else:
        cuda_available = torch.cuda.is_available()

    if cuda_available:
        # Some environments report CUDA available but fail at runtime
        # (e.g. invalid device function on embedding kernels). Smoke-test.
        try:
            emb = torch.nn.Embedding(16, 8).to("cuda")
            ids = torch.tensor([1, 2, 3, 4], device="cuda")
            _ = emb(ids).sum().item()
            return "cuda"
        except Exception:
            pass

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


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
    max_seq_len: int = 4096
    
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
    
    # Quantum-enhanced sync
    use_quantum_sync: bool = True
    quantum_coupling_strength: float = 0.1
    use_tpms_routing: bool = True


@dataclass
class GeometricPhysicsConfig:
    """Configuration for the unified Geometric Physics Layer"""
    # Module enables
    enable_fuller: bool = True  # Fuller Synergetics (tensegrity, VE, geodesics)
    enable_lloyd: bool = True  # Lloyd Computational Universe (limits, Landauer)
    enable_morphogenesis: bool = True  # Turing patterns, Ricci flow, catastrophe
    enable_tpms: bool = True  # TPMS-guided attention (gyroid, Schwarz)
    enable_qbit_nexus: bool = True  # Icosahedral quantum network (D-Wave inspired)
    enable_parallel_universe: bool = True  # Multiverse parallel computation
    
    # Combination mode:
    # - 'learnable': Parallel ensemble with learned gating (default)
    # - 'compound': Sequential chaining - each module feeds into next (true integration)
    # - 'sequential': Simple sequential pass through modules
    # - 'parallel': Average all module outputs
    # - 'residual': Sum all module outputs with input
    combination_mode: str = 'compound'  # Changed default to compound for true integration
    
    # Fuller Synergetics parameters
    fuller_ve_vertices: int = 12  # Vector Equilibrium vertices
    fuller_tensegrity_struts: int = 6  # Number of compression struts
    fuller_geodesic_frequency: int = 2  # Geodesic subdivision frequency
    
    # Lloyd Computational Universe parameters
    lloyd_energy_budget: float = 1.0  # Total energy budget for computation
    lloyd_temperature: float = 1.0  # Thermodynamic temperature
    lloyd_reversible_fraction: float = 0.5  # Fraction of reversible computation
    
    # Morphogenesis parameters
    morpho_diffusion_steps: int = 3  # Reaction-diffusion iterations
    morpho_activator_diffusion: float = 0.1  # Activator diffusion rate (Du)
    morpho_inhibitor_diffusion: float = 0.4  # Inhibitor diffusion rate (Dv > Du)
    
    # TPMS parameters
    tpms_surface_type: str = 'gyroid'  # gyroid, schwarz_p, schwarz_d, neovius
    tpms_num_heads: int = 8  # Number of attention heads
    tpms_threshold: float = 0.1  # Surface threshold for mask
    
    # QbitNexus parameters (icosahedral quantum network)
    qbit_num_vertices: int = 12  # Icosahedron has 12 vertices (golden ratio)
    qbit_num_layers: int = 2  # Number of quantum propagation layers
    qbit_dropout: float = 0.1
    
    # ParallelUniverse parameters (multiverse computation)
    parallel_num_universes: int = 4  # Number of parallel computational universes
    parallel_num_dimensions: int = 8  # Dimensions per universe
    parallel_overlap_dims: int = 4  # Overlap dimensions for interference
    parallel_collapse_mode: str = 'soft'  # 'soft', 'hard', 'superposition'
    
    # Physics-informed loss weights
    loss_weight_tensegrity: float = 0.01  # Tension-compression balance
    loss_weight_lloyd: float = 0.01  # Computational efficiency (Landauer)
    loss_weight_turing: float = 0.01  # Pattern emergence
    loss_weight_equilibrium: float = 0.01  # Vector equilibrium stability
    loss_weight_entanglement: float = 0.01  # Quantum entanglement coherence
    loss_weight_universe_overlap: float = 0.01  # Universe interference structure


@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration for scaled models"""
    enabled: bool = False  # When False, use standard dense FFN
    num_experts: int = 64
    top_k: int = 8
    expert_ffn_dim: int = 28672  # Per-expert FFN intermediate dim
    jitter_noise: float = 0.01  # Router exploration noise during training
    load_balance_weight: float = 0.001  # Reduced: pruning replaces forced balancing (Yuan LAEP)
    expert_prune_threshold: float = 0.25  # Prune experts below this fraction of mean load
    expert_prune_min: int = 8  # Never prune below this many experts
    # Compounding integration settings
    compound_enabled: bool = False  # Use CompoundMoELayer instead of base MoELayer
    compound_bias_scale: float = 0.1  # Scale for adaptive routing bias
    enable_cross_transfer: bool = True  # Cross-expert knowledge transfer


@dataclass
class QuantumCouplingConfig:
    """Configuration for quantum coupling between limbs"""
    # Coupling parameters (z = z³ + 7 → zero-point energy)
    coupling_strength: float = 0.1
    zero_point_energy: float = 7.0  # From z = z³ + 7
    
    # Oscillator parameters
    num_oscillators: int = 8  # One per limb
    frequency_scale: float = 1.0
    
    # Quantum parameters
    coherence_threshold: float = 0.5
    decoherence_rate: float = 0.01
    
    # Training
    learn_coupling_matrix: bool = True
    learn_frequencies: bool = True


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
class CompoundLoopConfig:
    """Compound looping — adaptive-depth reasoning (Ouro-style + Yuan RIRM)"""
    enabled: bool = False         # Off by default for backward compat
    max_loops: int = 4            # Maximum reasoning loop iterations
    exit_threshold: float = 0.5   # CDF threshold for early exit at inference
    entropy_beta: float = 0.1     # KL(exit_dist || uniform) regularization weight
    warmup_loops: int = 0         # Minimum loops before exit gate activates
    # RIRM: Reflection Inhibition Reward (Yuan 3.0 Ultra inspired)
    conciseness_reward: float = 0.05  # Penalizes late-loop probability mass
    max_cheap_loops: int = 2          # Loops beyond this are "expensive"


@dataclass
class CognitiveGeometryConfigDC:
    """Cognitive Geometry Engine — compound integration of ML vocabulary concepts."""
    enabled: bool = True
    # SVD Activation Decomposer
    svd_enabled: bool = True
    svd_top_k: int = 8
    svd_loss_weight: float = 0.01
    # Concept Alignment Matrix
    alignment_enabled: bool = True
    alignment_loss_weight: float = 0.01
    # Entropy Flow Monitor
    entropy_monitor_enabled: bool = True
    entropy_target: float = 2.0
    entropy_loss_weight: float = 0.005
    # Semantic Drift Detector
    drift_enabled: bool = True
    drift_max_rotation: float = 0.5
    drift_loss_weight: float = 0.01
    # Anchor Vector System
    anchor_enabled: bool = True
    num_anchors: int = 4
    anchor_decay: float = 0.95
    anchor_strength: float = 0.1
    # Repetition Dampener
    repetition_dampen_enabled: bool = True
    repetition_penalty: float = 1.2
    repetition_window: int = 32
    # Branch Scorer
    branch_scorer_enabled: bool = True
    branch_prune_threshold: float = 0.1
    # Manifold Partitioner
    manifold_enabled: bool = True
    manifold_loss_weight: float = 0.02
    # Goal Vector System
    goal_vector_enabled: bool = True
    goal_strength: float = 0.15
    # Attention Plane Reconstructor
    attention_plane_enabled: bool = True
    plane_dim: int = 16
    # Vector Field Tracker
    vector_field_enabled: bool = True


@dataclass 
class Config:
    """Master configuration combining all sub-configs"""
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rna_editing: RNAEditingConfig = field(default_factory=RNAEditingConfig)
    limb: LimbConfig = field(default_factory=LimbConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    geometric_physics: GeometricPhysicsConfig = field(default_factory=GeometricPhysicsConfig)
    quantum_coupling: QuantumCouplingConfig = field(default_factory=QuantumCouplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compound_loop: CompoundLoopConfig = field(default_factory=CompoundLoopConfig)
    cognitive_geometry: CognitiveGeometryConfigDC = field(default_factory=CognitiveGeometryConfigDC)
    
    # Device
    device: str = field(default_factory=_select_device)
    
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
            'moe': self.moe.__dict__,
            'geometric_physics': self.geometric_physics.__dict__,
            'quantum_coupling': self.quantum_coupling.__dict__,
            'training': {**self.training.__dict__, 'betas': self.training.betas},
            'cognitive_geometry': self.cognitive_geometry.__dict__,
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

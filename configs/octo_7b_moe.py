"""
OctoTetrahedral AGI — 7B MoE Configuration (Testable on single GPU)

Architecture: MoE (Mixture-of-Experts)
  - ~6.8 billion total parameters
  - ~2.6 billion active parameters per token
  - 8 experts with top-2 routing per layer
  - 22 transformer layers, hidden_dim=2048

Usage:
    from configs.octo_7b_moe import get_7b_moe_config
    config = get_7b_moe_config()
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    Config,
    GeometryConfig,
    ModelConfig,
    LoRAConfig,
    RNAEditingConfig,
    LimbConfig,
    SyncConfig,
    MoEConfig,
    GeometricPhysicsConfig,
    QuantumCouplingConfig,
    TrainingConfig,
)


def get_7b_moe_config(device: str = "cpu") -> Config:
    """
    Returns the 7B parameter OctoTetrahedral-MoE configuration.

    Total params:  ~6.8B
    Active params: ~2.6B per token (top-2 of 8 experts)
    Fits on a single 80GB GPU with bf16.
    """
    config = Config(
        geometry=GeometryConfig(
            num_points=64,
            num_vertices=4,
            num_edge_midpoints=6,
            num_face_centers=4,
            num_edge_subdivisions=24,
            num_face_subdivisions=12,
            num_internal_points=14,
        ),
        model=ModelConfig(
            vocab_size=100277,
            hidden_dim=2048,
            ffn_dim=5120,               # 2.5× hidden_dim (SwiGLU)
            num_layers=22,
            num_heads=32,               # 2048 / 64
            head_dim=64,
            max_seq_len=4096,
            dropout=0.0,
            memory_slots=16,
            num_points=64,
        ),
        lora=LoRAConfig(rank=16, alpha=32.0, dropout=0.0),
        rna_editing=RNAEditingConfig(
            temperature_init=1.0,
            num_pathways=8,
            num_gated_heads=32,
            editing_density=0.05,
        ),
        limb=LimbConfig(buffer_size=500, local_lr=3e-5, confidence_threshold=0.5),
        sync=SyncConfig(
            sync_frequency=10,
            max_drift=0.05,
            rollback_buffer_size=10,
            use_performance_weighting=True,
            use_quantum_sync=True,
            quantum_coupling_strength=0.05,
            use_tpms_routing=True,
        ),
        moe=MoEConfig(
            enabled=True,
            num_experts=8,
            top_k=2,
            expert_ffn_dim=5120,
            jitter_noise=0.01,
            load_balance_weight=0.01,
        ),
        geometric_physics=GeometricPhysicsConfig(
            enable_fuller=True,
            enable_lloyd=True,
            enable_morphogenesis=True,
            enable_tpms=True,
            enable_qbit_nexus=True,
            enable_parallel_universe=True,
            combination_mode="compound",
            tpms_num_heads=32,
        ),
        quantum_coupling=QuantumCouplingConfig(coupling_strength=0.05, num_oscillators=8),
        training=TrainingConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            batch_size=32,
            gradient_accumulation_steps=4,
            warmup_steps=1000,
            max_steps=100_000,
            log_interval=10,
            eval_interval=500,
            save_interval=2000,
            max_grad_norm=1.0,
        ),
        device=device,
        seed=42,
    )
    return config


if __name__ == "__main__":
    from configs.octo_1_72t import estimate_params

    config = get_7b_moe_config()
    stats = estimate_params(config)

    print("OctoTetrahedral-7B-MoE Configuration")
    print("=" * 50)
    print(f"Hidden dim:        {config.model.hidden_dim}")
    print(f"Layers:            {config.model.num_layers}")
    print(f"Heads:             {config.model.num_heads} (head_dim={config.model.head_dim})")
    print(f"Experts:           {config.moe.num_experts} (top-{config.moe.top_k})")
    print(f"Expert FFN dim:    {config.moe.expert_ffn_dim}")
    print(f"Max seq len:       {config.model.max_seq_len}")
    print()
    print(f"Total parameters:  {stats['total_T']*1000:.2f}B ({stats['total_params']:,})")
    print(f"Active per token:  {stats['active_B']:.2f}B ({stats['active_params']:,})")

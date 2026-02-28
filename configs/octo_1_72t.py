"""
OctoTetrahedral AGI — 1.72T Parameter Configuration

Architecture: MoE (Mixture-of-Experts)
  - 1.727 trillion total parameters
  - ~226 billion active parameters per token
  - 64 experts with top-8 routing per layer
  - 63 transformer layers, hidden_dim=7168
  - Full OctoTetrahedral: 8 limbs, RNA editing, tetrahedral geometry

Usage:
    from configs.octo_1_72t import get_1_72t_config
    config = get_1_72t_config()
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


def get_1_72t_config(device: str = "cpu") -> Config:
    """
    Returns the 1.72T parameter OctoTetrahedral-MoE configuration.

    Total params:  ~1.727T
    Active params: ~226B per token (top-8 of 64 experts)
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
            vocab_size=100277,          # tiktoken cl100k_base
            hidden_dim=7168,            # d_model
            ffn_dim=17920,              # 2.5× hidden_dim (SwiGLU; used for limbs too)
            num_layers=67,              # transformer depth
            num_heads=56,               # 7168 / 128
            head_dim=128,               # standard for large models
            max_seq_len=8192,           # context length
            dropout=0.0,               # large models typically use 0 dropout
            memory_slots=64,            # scaled working memory
            num_points=64,
        ),
        lora=LoRAConfig(
            rank=64,                    # scaled LoRA rank
            alpha=128.0,
            dropout=0.0,
        ),
        rna_editing=RNAEditingConfig(
            temperature_init=1.0,
            num_pathways=8,             # one per limb
            num_gated_heads=56,         # match num_heads
            editing_density=0.05,       # lower density at scale
        ),
        limb=LimbConfig(
            buffer_size=1000,           # larger replay buffer
            local_lr=1e-5,              # lower LR for stability
            confidence_threshold=0.5,
        ),
        sync=SyncConfig(
            sync_frequency=20,
            max_drift=0.05,             # tighter drift at scale
            rollback_buffer_size=20,
            use_performance_weighting=True,
            use_quantum_sync=True,
            quantum_coupling_strength=0.05,
            use_tpms_routing=True,
        ),
        moe=MoEConfig(
            enabled=True,
            num_experts=64,
            top_k=8,
            expert_ffn_dim=17920,       # 2.5× d_model per expert (SwiGLU)
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
            tpms_num_heads=56,
        ),
        quantum_coupling=QuantumCouplingConfig(
            coupling_strength=0.05,
            num_oscillators=8,
        ),
        training=TrainingConfig(
            learning_rate=1.5e-4,       # cosine decay from here
            weight_decay=0.1,
            betas=(0.9, 0.95),          # GPT-style betas
            batch_size=2048,            # tokens per micro-batch
            gradient_accumulation_steps=64,  # effective batch = 131K tokens
            warmup_steps=2000,
            max_steps=500_000,
            log_interval=10,
            eval_interval=500,
            save_interval=5000,
            max_grad_norm=1.0,
        ),
        device=device,
        seed=42,
    )
    return config


def estimate_params(config: Config) -> dict:
    """Estimate total and active parameter counts for the configuration."""
    d = config.model.hidden_dim
    L = config.model.num_layers
    V = config.model.vocab_size
    E = config.moe.num_experts
    K = config.moe.top_k
    ffn = config.moe.expert_ffn_dim if config.moe.enabled else config.model.ffn_dim

    # Embedding + output head
    embed_params = 2 * V * d

    # Per-layer attention: Q, K, V, O projections
    attn_per_layer = 4 * d * d

    if config.moe.enabled:
        # MoE FFN: each expert has gate + up + down (SwiGLU: 3 matrices)
        expert_params = 3 * d * ffn
        router_params = d * E
        ffn_per_layer_total = E * expert_params + router_params
        ffn_per_layer_active = K * expert_params + router_params
    else:
        ffn_per_layer_total = 2 * d * ffn
        ffn_per_layer_active = ffn_per_layer_total

    # Layer norms: 2 per layer × 2 params (weight + bias) × d
    norm_per_layer = 4 * d

    total_per_layer = attn_per_layer + ffn_per_layer_total + norm_per_layer
    active_per_layer = attn_per_layer + ffn_per_layer_active + norm_per_layer

    # 8 limbs: each has attention + FFN at full dim (2 sub-layers)
    limb_params = 8 * (attn_per_layer + 2 * d * config.model.ffn_dim)

    # Misc: RNA editing, geometric physics, hub sync (~3% of transformer)
    misc_overhead = int(0.03 * L * total_per_layer)

    total = embed_params + L * total_per_layer + limb_params + misc_overhead
    active = embed_params + L * active_per_layer + limb_params + misc_overhead

    return {
        "total_params": total,
        "active_params": active,
        "total_T": total / 1e12,
        "active_B": active / 1e9,
        "embed_params": embed_params,
        "transformer_total": L * total_per_layer,
        "transformer_active": L * active_per_layer,
        "limb_params": limb_params,
        "misc_overhead": misc_overhead,
    }


if __name__ == "__main__":
    config = get_1_72t_config()
    stats = estimate_params(config)

    print("OctoTetrahedral-1.72T Configuration")
    print("=" * 55)
    print(f"Hidden dim:        {config.model.hidden_dim}")
    print(f"Layers:            {config.model.num_layers}")
    print(f"Heads:             {config.model.num_heads} (head_dim={config.model.head_dim})")
    print(f"Experts:           {config.moe.num_experts} (top-{config.moe.top_k})")
    print(f"Expert FFN dim:    {config.moe.expert_ffn_dim}")
    print(f"Max seq len:       {config.model.max_seq_len}")
    print(f"Vocab size:        {config.model.vocab_size:,}")
    print()
    print(f"Total parameters:  {stats['total_T']:.3f}T ({stats['total_params']:,})")
    print(f"Active per token:  {stats['active_B']:.1f}B ({stats['active_params']:,})")
    print(f"  Embeddings:      {stats['embed_params']/1e9:.2f}B")
    print(f"  Transformer:     {stats['transformer_total']/1e12:.3f}T total, {stats['transformer_active']/1e9:.1f}B active")
    print(f"  Limbs:           {stats['limb_params']/1e9:.2f}B")
    print(f"  Overhead:        {stats['misc_overhead']/1e9:.2f}B")

"""
Distributed scaling infrastructure for OctoTetrahedral AGI.

Genus-13 Architecture at Scale:
  14 information streams form a genus-13 topological manifold.
  At 1.7T parameters, each "handle" of the genus surface carries
  billions of cross-stream information pathways through the
  compound braid.

Scale Presets:
  tiny  → 204M  (CPU prototyping, current)
  base  → ~7B   (single GPU / small cluster)
  large → ~110B (multi-GPU node)
  ultra → ~1.7T (distributed cluster, 512+ H100s)

Usage:
  from core.distributed_scale import apply_scale_preset, get_deepspeed_config
  config = apply_scale_preset('ultra')
  ds_config = get_deepspeed_config(config.scale)
"""

import json
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

try:
    import torch.distributed as dist
except ImportError:
    dist = None


def apply_scale_preset(preset: str, base_config=None):
    """Apply a scaling preset to the master config.
    
    Args:
        preset: One of 'tiny', 'base', 'large', 'ultra'
        base_config: Existing Config to modify (creates new if None)
        
    Returns:
        Modified Config with scaled parameters
    """
    from config import Config, ScaleConfig
    
    config = base_config if base_config is not None else Config()
    config.scale.preset = preset
    
    presets = ScaleConfig.PRESETS
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    
    h, layers, heads, experts, ffn, top_k, _ = presets[preset]
    
    # Core model dimensions
    config.model.hidden_dim = h
    config.model.num_layers = layers
    config.model.num_heads = heads
    config.model.head_dim = h // heads
    config.model.ffn_dim = h * 4  # Dense FFN (used alongside MoE)
    
    # MoE configuration
    config.moe.enabled = True
    config.moe.compound_enabled = True
    config.moe.num_experts = experts
    config.moe.expert_ffn_dim = ffn
    config.moe.top_k = top_k
    
    # Scale-dependent training config
    if preset == 'ultra':
        config.scale.tensor_parallel = 8
        config.scale.pipeline_parallel = 16
        config.scale.expert_parallel = 8
        config.scale.data_parallel = 4  # Total: 8×16×4 = 512 GPUs minimum
        config.scale.zero_stage = 3
        config.scale.offload_optimizer = True
        config.scale.gradient_checkpointing = True
        config.scale.mixed_precision = 'bf16'
        config.scale.global_batch_size = 2048
        config.scale.micro_batch_size = 1
        config.scale.sequence_length = 8192
        config.training.learning_rate = 1.5e-4
        config.training.weight_decay = 0.1
    elif preset == 'large':
        config.scale.tensor_parallel = 4
        config.scale.pipeline_parallel = 4
        config.scale.expert_parallel = 4
        config.scale.data_parallel = 2  # Total: 4×4×2 = 32 GPUs
        config.scale.zero_stage = 3
        config.scale.gradient_checkpointing = True
        config.scale.mixed_precision = 'bf16'
        config.scale.global_batch_size = 512
        config.scale.micro_batch_size = 2
        config.scale.sequence_length = 4096
        config.training.learning_rate = 3e-4
    elif preset == 'base':
        config.scale.tensor_parallel = 2
        config.scale.pipeline_parallel = 1
        config.scale.expert_parallel = 2
        config.scale.data_parallel = 2  # Total: 2×1×2 = 4 GPUs
        config.scale.zero_stage = 2
        config.scale.gradient_checkpointing = True
        config.scale.mixed_precision = 'bf16'
        config.scale.global_batch_size = 64
        config.scale.micro_batch_size = 4
        config.scale.sequence_length = 2048
    # tiny: keep defaults
    
    return config


def get_deepspeed_config(scale_config) -> Dict[str, Any]:
    """Generate DeepSpeed JSON config for the given scale preset.
    
    Returns a dict suitable for `deepspeed.initialize(config=...)`.
    """
    grad_accum = max(1, scale_config.global_batch_size // (
        scale_config.micro_batch_size * scale_config.data_parallel
    ))
    
    ds_config = {
        "train_batch_size": scale_config.global_batch_size,
        "train_micro_batch_size_per_gpu": scale_config.micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": scale_config.zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        
        "activation_checkpointing": {
            "partition_activations": scale_config.gradient_checkpointing,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False,
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": 50,
    }
    
    # FP16/BF16
    if scale_config.mixed_precision == 'bf16':
        ds_config["bf16"] = {"enabled": True}
    elif scale_config.mixed_precision == 'fp16':
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
        }
    
    # ZeRO Stage 3 offloading
    if scale_config.zero_stage == 3:
        ds_config["zero_optimization"]["sub_group_size"] = 1e9
        ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6
        
        if scale_config.offload_param:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        if scale_config.offload_optimizer:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": True,
            }
    
    # Optimizer
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 1.5e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
    }
    
    # Scheduler
    ds_config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1.5e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 500000,
        }
    }
    
    return ds_config


def get_fsdp_config(scale_config) -> Dict[str, Any]:
    """Generate PyTorch FSDP configuration."""
    return {
        "sharding_strategy": "FULL_SHARD",  # ZeRO-3 equivalent
        "cpu_offload": scale_config.offload_param,
        "mixed_precision": scale_config.mixed_precision,
        "auto_wrap_policy": "size_based",
        "min_num_params": 1e6,
        "backward_prefetch": "BACKWARD_PRE",
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "use_orig_params": True,
    }


class ModelParallelismManager:
    """Manages tensor, pipeline, and expert parallelism for the OctoTetrahedral model.
    
    At ultra scale (1.7T params), the model is distributed as:
      - Tensor parallel: attention heads + dense FFN split across TP GPUs
      - Pipeline parallel: layers split into PP stages
      - Expert parallel: MoE experts distributed across EP GPUs
      - Data parallel: remaining GPUs replicate for data parallelism
    
    Total GPUs = TP × PP × EP × DP
    
    Genus-13 topology ensures that the 14 compound braid streams
    maintain full connectivity even when distributed — each stream
    is replicated across all tensor-parallel ranks.
    """
    
    def __init__(self, scale_config):
        self.config = scale_config
        self.tp = scale_config.tensor_parallel
        self.pp = scale_config.pipeline_parallel
        self.ep = scale_config.expert_parallel
        self.dp = scale_config.data_parallel
        self.total_gpus = self.tp * self.pp * self.ep * self.dp
    
    def get_layer_assignment(self, num_layers: int) -> Dict[int, int]:
        """Assign layers to pipeline stages."""
        layers_per_stage = math.ceil(num_layers / self.pp)
        assignment = {}
        for layer_idx in range(num_layers):
            stage = min(layer_idx // layers_per_stage, self.pp - 1)
            assignment[layer_idx] = stage
        return assignment
    
    def get_expert_assignment(self, num_experts: int) -> Dict[int, int]:
        """Assign MoE experts to expert-parallel ranks."""
        experts_per_rank = math.ceil(num_experts / self.ep)
        assignment = {}
        for expert_idx in range(num_experts):
            rank = min(expert_idx // experts_per_rank, self.ep - 1)
            assignment[expert_idx] = rank
        return assignment
    
    def get_head_assignment(self, num_heads: int) -> Dict[int, int]:
        """Assign attention heads to tensor-parallel ranks."""
        heads_per_rank = math.ceil(num_heads / self.tp)
        assignment = {}
        for head_idx in range(num_heads):
            rank = min(head_idx // heads_per_rank, self.tp - 1)
            assignment[head_idx] = rank
        return assignment
    
    def summary(self) -> str:
        """Print parallelism summary."""
        hw = self.config.get_hardware_estimate()
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  OctoTetrahedral AGI — {hw['preset'].upper()} Scale    ║",
            f"║  Genus-13 Topology × {hw['total_params_str']} Parameters  ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Total params:     {hw['total_params_str']:>10}                  ║",
            f"║  Active per pass:  {hw['active_params_str']:>10}                  ║",
            f"║  Hidden dim:       {hw['hidden_dim']:>10}                  ║",
            f"║  Layers:           {hw['num_layers']:>10}                  ║",
            f"║  Heads:            {hw['num_heads']:>10}                  ║",
            f"║  MoE experts:      {hw['num_experts']:>10}                  ║",
            f"║  Expert FFN dim:   {hw['expert_ffn_dim']:>10}                  ║",
            f"║  Top-K routing:    {hw['top_k']:>10}                  ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Parallelism:                                    ║",
            f"║    Tensor parallel:  {self.tp:>4} GPUs                   ║",
            f"║    Pipeline parallel:{self.pp:>4} stages                  ║",
            f"║    Expert parallel:  {self.ep:>4} GPUs                   ║",
            f"║    Data parallel:    {self.dp:>4} replicas                ║",
            f"║    Total GPUs:       {self.total_gpus:>4}                       ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Memory (BF16 training):                         ║",
            f"║    Model:    {hw['model_memory_gb']:>8.0f} GB                     ║",
            f"║    Optimizer:{hw['optimizer_memory_gb']:>8.0f} GB                     ║",
            f"║    Total:    {hw['total_training_memory_gb']:>8.0f} GB                     ║",
            f"║    Min H100s:{hw['min_h100_gpus']:>8}                        ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Compound Braid: 14 streams (genus=13)           ║",
            f"║  Modalities: text + vision + audio + embodiment  ║",
            f"╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def print_all_presets():
    """Print hardware estimates for all scale presets."""
    from config import ScaleConfig
    
    for preset_name in ['tiny', 'base', 'large', 'ultra']:
        sc = ScaleConfig(preset=preset_name)
        hw = sc.get_hardware_estimate()
        mpm = ModelParallelismManager(sc)
        
        # Apply default parallelism for each preset
        if preset_name == 'ultra':
            sc.tensor_parallel = 8
            sc.pipeline_parallel = 16
            sc.expert_parallel = 8
            sc.data_parallel = 4
        elif preset_name == 'large':
            sc.tensor_parallel = 4
            sc.pipeline_parallel = 4
            sc.expert_parallel = 4
            sc.data_parallel = 2
        elif preset_name == 'base':
            sc.tensor_parallel = 2
            sc.pipeline_parallel = 1
            sc.expert_parallel = 2
            sc.data_parallel = 2
        
        mpm = ModelParallelismManager(sc)
        print(mpm.summary())
        print()


if __name__ == '__main__':
    print_all_presets()
    
    # Show DeepSpeed config for ultra
    from config import Config
    config = apply_scale_preset('ultra')
    ds = get_deepspeed_config(config.scale)
    print("\n=== DeepSpeed Config (Ultra / 1.7T) ===")
    print(json.dumps(ds, indent=2))

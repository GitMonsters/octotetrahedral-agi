"""
OctoTetrahedral AGI — Distributed Training Launcher

Supports:
  - PyTorch FSDP (Fully Sharded Data Parallel) for model parallelism
  - Expert parallelism for MoE layers
  - Mixed precision (bf16/fp16)
  - Gradient checkpointing

Usage:
  # Single node, multi-GPU (FSDP)
  torchrun --nproc_per_node=4 train_distributed.py --config 7b

  # Multi-node (2 nodes × 4 GPUs)
  torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \\
      --rdzv_endpoint=master:29500 train_distributed.py --config 70b

  # Full 1.72T model (8 nodes × 8 GPUs)
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d \\
      --rdzv_endpoint=master:29500 train_distributed.py --config 1.72t
"""

import argparse
import os
import logging
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from config import Config
from model import OctoTetrahedralModel
from core.tetrahedral_core import TetrahedralTransformerLayer
from core.moe import MoELayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(name: str) -> Config:
    """Load a named configuration preset."""
    if name in ("7b", "7B"):
        from configs.octo_7b_moe import get_7b_moe_config
        return get_7b_moe_config()
    elif name in ("70b", "70B"):
        from configs.octo_70b_moe import get_70b_moe_config
        return get_70b_moe_config()
    elif name in ("1.72t", "1.72T", "1_72t"):
        from configs.octo_1_72t import get_1_72t_config
        return get_1_72t_config()
    else:
        from config import get_config
        return get_config()


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def get_fsdp_config(config: Config) -> Dict[str, Any]:
    """Build FSDP kwargs based on model config."""
    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy: wrap each transformer layer and MoE layer independently
    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TetrahedralTransformerLayer, MoELayer},
    )

    return dict(
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,  # Required for AdamW with weight decay groups
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_optimizer(model: FSDP, config: Config) -> AdamW:
    """Create AdamW with proper weight-decay groups for FSDP."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    params_decay = []
    params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    return AdamW(
        [
            {"params": params_decay, "weight_decay": config.training.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=config.training.betas,
    )


def create_scheduler(optimizer: AdamW, config: Config) -> LambdaLR:
    """Cosine decay with linear warmup."""
    warmup = config.training.warmup_steps
    total = config.training.max_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def create_dummy_dataloader(config: Config, rank: int, world_size: int) -> DataLoader:
    """Create a dummy dataloader for smoke-testing distributed setup."""
    from torch.utils.data import TensorDataset

    n_samples = config.training.batch_size * 10
    seq_len = min(config.model.max_seq_len, 512)
    data = torch.randint(0, config.model.vocab_size, (n_samples, seq_len))
    dataset = TensorDataset(data, data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    def collate(batch):
        inputs, labels = zip(*batch)
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labels)}

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )


def train(
    rank: int,
    world_size: int,
    config: Config,
    max_steps: Optional[int] = None,
):
    """Main distributed training loop."""
    max_steps = max_steps or config.training.max_steps
    is_main = rank == 0

    # Build model
    if is_main:
        logger.info(f"Building OctoTetrahedral model ({config.model.num_layers}L, "
                     f"d={config.model.hidden_dim}, MoE={config.moe.enabled})")

    model = OctoTetrahedralModel(config, use_geometric_physics=False)

    if is_main:
        total = model.get_num_params()
        active = model.get_active_params()
        logger.info(f"Total params: {total:,} ({total/1e9:.1f}B)")
        if config.moe.enabled:
            logger.info(f"Active params/token: {active:,} ({active/1e9:.1f}B)")
            logger.info(f"Experts: {config.moe.num_experts}, top-{config.moe.top_k}")

    # Wrap with FSDP
    fsdp_kwargs = get_fsdp_config(config)
    model = FSDP(model, **fsdp_kwargs)

    if is_main:
        logger.info(f"FSDP wrapped, sharding across {world_size} GPUs")

    # Optimizer + scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Data
    dataloader = create_dummy_dataloader(config, rank, world_size)

    # Training loop
    model.train()
    global_step = 0
    accum_steps = config.training.gradient_accumulation_steps
    start_time = time.time()

    if is_main:
        logger.info(f"Starting training: {max_steps} steps, "
                     f"grad_accum={accum_steps}, world_size={world_size}")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(rank)
            labels = batch["labels"].to(rank)

            output = model(input_ids=input_ids, labels=labels)
            loss = output["loss"] / accum_steps
            loss.backward()

            if (global_step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_main and global_step % config.training.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = max(1, global_step) / elapsed
                moe_loss = output.get("moe_aux_loss")
                moe_str = f" | MoE aux: {moe_loss.item():.3f}" if moe_loss is not None else ""
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {loss.item() * accum_steps:.4f}{moe_str} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            global_step += 1

    # Save checkpoint (rank 0 only)
    if is_main:
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        # FSDP full state dict
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state = model.state_dict()
            torch.save(
                {"model_state_dict": state, "config": config.to_dict(), "step": global_step},
                ckpt_dir / "distributed_final.pt",
            )
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/60:.1f}min. Saved checkpoint.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OctoTetrahedral Distributed Training")
    parser.add_argument("--config", type=str, default="7b",
                        choices=["default", "7b", "70b", "1.72t"],
                        help="Config preset to use")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps")
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"Distributed training: {world_size} GPUs, config={args.config}")

    config = load_config(args.config)
    config.device = f"cuda:{rank}"

    try:
        train(rank, world_size, config, max_steps=args.max_steps)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

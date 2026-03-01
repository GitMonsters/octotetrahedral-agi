#!/usr/bin/env python3
"""
Lean ARC training script for GPU cloud instances.
Bypasses the full Trainer class to minimize memory overhead.
Uses bfloat16 + SGD for memory-efficient 7B training.

Usage:
    python3 train_gpu.py --config 7b --max-steps 50000 --device cuda:0
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import argparse
import time
import math
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_config(name: str):
    if name == '7b':
        from configs.octo_7b_moe import get_7b_moe_config
        return get_7b_moe_config()
    elif name == '70b':
        from configs.octo_70b_moe import get_70b_moe_config
        return get_70b_moe_config()
    else:
        from config import Config
        return Config()


def build_data(tokenizer, max_seq_len: int = 512):
    """Load ARC + synthetic training data."""
    from data.arc_dataset import ARCDataset
    from train_arc_moe import SyntheticARCDataset, collate_fn

    datasets = []

    # Real ARC data
    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"
    if (arc_dir / "training").exists():
        arc_train = ARCDataset(data_dir=str(arc_dir), split="training",
                               tokenizer=tokenizer, max_seq_len=max_seq_len)
        datasets.append(arc_train)
        logger.info(f"ARC training: {len(arc_train)} samples")

    # Synthetic data (check both home dir and cwd)
    for search_dir in [Path.home(), Path.cwd()]:
        for synth_file in sorted(search_dir.glob("synthetic_arc_dataset_*.json")):
            synth = SyntheticARCDataset(str(synth_file), tokenizer, max_seq_len)
            datasets.append(synth)
            logger.info(f"Synthetic: {len(synth)} samples from {synth_file.name}")

    if not datasets:
        raise RuntimeError("No training data found!")

    # Validation
    val_ds = None
    if (arc_dir / "evaluation").exists():
        val_ds = ARCDataset(data_dir=str(arc_dir), split="evaluation",
                            tokenizer=tokenizer, max_seq_len=max_seq_len)
        logger.info(f"Validation: {len(val_ds)} samples")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    logger.info(f"Total training: {len(combined)} samples")
    return combined, val_ds, collate_fn


def train(args):
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # Tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Data
    train_ds, val_ds, collate_fn = build_data(tokenizer, args.max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, collate_fn=lambda b: collate_fn(b), drop_last=True)
    val_dl = None
    if val_ds:
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=lambda b: collate_fn(b))

    # Model
    logger.info(f"Building {args.config} model...")
    config = get_config(args.config)
    from model import OctoTetrahedralModel
    model = OctoTetrahedralModel(config, use_geometric_physics=False)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params/1e9:.2f}B params")

    # Resume checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get('model_state_dict', ckpt)
        # Filter mismatched shapes
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded {len(filtered)}/{len(model_state)} params")

    # BFloat16 for memory efficiency
    use_bf16 = device.type == 'cuda'
    if use_bf16:
        model = model.to(torch.bfloat16)
        logger.info("Model cast to bfloat16")

    model.to(device)
    if device.type == 'cuda':
        logger.info(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Gradient checkpointing
    for layer in model.core.layers:
        orig = layer.forward
        def make_ckpt(fn):
            def wrapper(*a, **kw):
                return checkpoint(fn, *a, use_reentrant=False, **kw)
            return wrapper
        layer.forward = make_ckpt(orig)
    logger.info(f"Gradient checkpointing on {len(model.core.layers)} layers")

    # Optimizer — SGD for memory efficiency (1 state vs Adam's 2)
    lr = args.lr or (config.training.learning_rate * 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=0.01)
    logger.info(f"SGD optimizer, lr={lr:.2e}")

    # LR scheduler: linear warmup + cosine decay
    warmup = args.warmup_steps
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Starting training: {args.max_steps} steps, batch {args.batch_size}")
    model.train()
    global_step = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    start_time = time.time()
    epoch = 0

    while global_step < args.max_steps:
        epoch += 1
        for batch in train_dl:
            if global_step >= args.max_steps:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attn_mask = batch.get('attention_mask')
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            # Forward
            output = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = output['loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                sps = global_step / elapsed if elapsed > 0 else 0
                lr_now = scheduler.get_last_lr()[0]
                mem = torch.cuda.memory_allocated()/1024**3 if device.type == 'cuda' else 0
                logger.info(
                    f"Step {global_step}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | "
                    f"Speed: {sps:.2f} steps/s | GPU: {mem:.1f}GB"
                )
                running_loss = 0.0

            # Save checkpoint
            if global_step % args.save_interval == 0:
                ckpt_path = ckpt_dir / f"arc_step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                    'loss': loss.item(),
                    'config': args.config,
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # Validation
            if val_dl and global_step % args.eval_interval == 0:
                model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for vb in val_dl:
                        vi = vb['input_ids'].to(device)
                        vl = vb['labels'].to(device)
                        va = vb.get('attention_mask')
                        if va is not None:
                            va = va.to(device)
                        vo = model(input_ids=vi, attention_mask=va, labels=vl)
                        val_loss += vo['loss'].item()
                        val_count += 1
                        if val_count >= 100:
                            break
                avg_val = val_loss / max(1, val_count)
                logger.info(f"Validation loss: {avg_val:.4f} (n={val_count})")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_path = ckpt_dir / "arc_best.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'step': global_step,
                        'val_loss': avg_val,
                        'config': args.config,
                    }, best_path)
                    logger.info(f"New best model saved! val_loss={avg_val:.4f}")
                model.train()

    # Final save
    final_path = ckpt_dir / "arc_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'config': args.config,
    }, final_path)
    elapsed = time.time() - start_time
    logger.info(f"Training complete! {global_step} steps in {elapsed/3600:.1f}h")
    logger.info(f"Final checkpoint: {final_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OctoTetrahedral GPU Training")
    parser.add_argument("--config", choices=["default", "7b", "70b"], default="7b")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=2500)
    args = parser.parse_args()
    train(args)

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


def init_compound_learning(model):
    """Wire compound learning into the 8-limb architecture.
    
    Registers each limb as a model in the CompoundLearningEngine so
    cross-limb interactions are tracked and optimized over training.
    """
    from ngvt_compound_learning import CompoundLearningEngine, CompoundIntegrationEngine

    engine = CompoundLearningEngine(max_patterns=10000, learning_rate=0.01)
    integration = CompoundIntegrationEngine(learning_engine=engine)

    limb_capabilities = {
        'perception':    ['input_encoding', 'tokenization', 'embedding'],
        'memory':        ['episodic_storage', 'semantic_storage', 'retrieval'],
        'planning':      ['goal_sequencing', 'action_planning', 'lookahead'],
        'language':       ['nlu', 'nlg', 'grounding'],
        'spatial':       ['grid_reasoning', 'geometric_transform', 'pattern_detection'],
        'reasoning':     ['abstract_patterns', 'logical_inference', 'analogy'],
        'metacognition': ['self_monitoring', 'confidence_estimation', 'adaptation'],
        'action':        ['output_generation', 'decision_making', 'response_formation'],
    }

    for limb_name, caps in limb_capabilities.items():
        engine.register_model(limb_name, caps)
        integration.register_model(limb_name, 'limb', {
            'capabilities': caps,
            'hidden_dim': model.hidden_dim,
        })

    # Define the default integration path (perception → reasoning → action)
    integration.define_integration_path(
        'arc_solve',
        ['perception', 'spatial', 'reasoning', 'memory', 'planning', 'language', 'action']
    )

    logger.info("Compound learning initialized: 8 limbs registered, arc_solve path defined")
    return engine, integration


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


def build_data(tokenizer, max_seq_len: int = 4096, curriculum: bool = False):
    """Load ARC-AGI-2 training data (1000 tasks, superset of AGI-1).
    
    IMPORTANT: Only includes samples that have actual target outputs.
    Synthetic datasets with no test outputs are skipped.
    """
    from data.arc_dataset import ARCDataset
    from train_arc_moe import SyntheticARCDataset, collate_fn

    datasets = []

    # Prefer ARC-AGI-2 (1000 tasks, superset of AGI-1's 400)
    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI-2" / "data"
    if not (arc_dir / "training").exists():
        # Fall back to AGI-1
        arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"

    if (arc_dir / "training").exists():
        arc_train = ARCDataset(data_dir=str(arc_dir), split="training",
                               tokenizer=tokenizer, max_seq_len=max_seq_len,
                               curriculum=curriculum)
        datasets.append(arc_train)
        logger.info(f"ARC training: {len(arc_train)} samples from {arc_dir}"
                     + (" [curriculum: easy→hard]" if curriculum else ""))

    # Synthetic data — only if it has test outputs
    for search_dir in [Path.home(), Path.cwd()]:
        for synth_file in sorted(search_dir.glob("synthetic_arc_dataset_*.json")):
            import json as _json
            with open(synth_file) as _f:
                _raw = _json.load(_f)
            has_outputs = any(
                t.get('output') for td in _raw.values() for t in td.get('test', [])
            )
            if has_outputs:
                synth = SyntheticARCDataset(str(synth_file), tokenizer, max_seq_len)
                datasets.append(synth)
                logger.info(f"Synthetic: {len(synth)} samples from {synth_file.name}")
            else:
                logger.warning(f"SKIPPING {synth_file.name} — no test outputs!")

    if not datasets:
        raise RuntimeError("No training data found!")

    # Validation — combine AGI-1 + AGI-2 eval for broader coverage
    val_datasets = []
    for data_subdir in ["ARC-AGI-2/data", "ARC-AGI/data"]:
        eval_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / data_subdir
        if (eval_dir / "evaluation").exists():
            vds = ARCDataset(data_dir=str(eval_dir), split="evaluation",
                             tokenizer=tokenizer, max_seq_len=max_seq_len)
            val_datasets.append(vds)
            logger.info(f"Validation: {len(vds)} samples from {eval_dir}")

    val_ds = None
    if val_datasets:
        from torch.utils.data import ConcatDataset as CD
        val_ds = CD(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        logger.info(f"Total validation: {len(val_ds)} samples")

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
    train_ds, val_ds, collate_fn = build_data(tokenizer, args.max_seq_len, curriculum=args.curriculum)
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

    # Optimizer — prefer Adam8bit > AdamW > SGD
    lr = args.lr or config.training.learning_rate
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr, weight_decay=0.01)
        opt_name = "Adam8bit"
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                      weight_decay=0.01, fused=device.type == 'cuda')
        opt_name = "AdamW"
    logger.info(f"{opt_name} optimizer, lr={lr:.2e}")

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
    grad_accum = args.grad_accum
    logger.info(f"Starting training: {args.max_steps} steps, batch {args.batch_size}, "
                f"grad_accum={grad_accum} (effective batch={args.batch_size * grad_accum})")

    # Compound learning — tracks cross-limb interactions
    compound_engine, compound_integration = init_compound_learning(model)
    compound_cycle_interval = 500  # Run learning cycle every N steps

    model.train()
    global_step = 0
    running_loss = 0.0
    best_val_loss = float('inf')
    start_time = time.time()
    epoch = 0
    micro_step = 0

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
            loss = output['loss'] / grad_accum  # Scale loss for accumulation

            # Backward
            loss.backward()
            micro_step += 1

            # Only step optimizer every grad_accum micro-steps
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item() * grad_accum
                global_step += 1

                # Compound learning: record experience + sync limbs
                from ngvt_compound_learning import LearningExperience
                from datetime import datetime
                step_loss = loss.item() * grad_accum
                compound_engine.record_experience(LearningExperience(
                    query=f"arc_train_step_{global_step}",
                    response=f"loss={step_loss:.4f}",
                    latency_ms=(time.time() - start_time) * 1000 / max(1, global_step),
                    success=step_loss < 3.0,
                    timestamp=datetime.now().isoformat(),
                    confidence=max(0.0, 1.0 - step_loss / 5.0),
                    metadata={'step': global_step, 'lr': scheduler.get_last_lr()[0]},
                ))

                # Sync limbs based on training performance
                sync_result = model.sync_limbs(performance=max(0.0, 1.0 - step_loss / 5.0))
                if sync_result:
                    logger.info(f"Hub sync at step {global_step}: {sync_result.get('action', 'sync')}")

                # Periodic compound learning cycle — extract patterns, optimize
                if global_step % compound_cycle_interval == 0:
                    cycle_stats = compound_engine.compound_learning_cycle()
                    n_patterns = cycle_stats.get('total_patterns', 0)
                    efficiency = cycle_stats.get('transfer_efficiency', 0)
                    logger.info(
                        f"Compound learning cycle: {n_patterns} patterns, "
                        f"transfer_efficiency={efficiency:.3f}"
                    )

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

    # Final save (include compound learning state)
    compound_state = {
        'learning_stats': compound_engine.get_learning_stats(),
        'patterns': [vars(p) for p in compound_engine.extract_patterns(min_frequency=1)],
    }
    final_path = ckpt_dir / "arc_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'config': args.config,
        'compound_learning': compound_state,
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
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--curriculum", action="store_true", help="Sort tasks easy→hard (curriculum learning)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=2500)
    args = parser.parse_args()
    train(args)

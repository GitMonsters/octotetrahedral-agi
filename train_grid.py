"""
OctoTetrahedral AGI — Grid-Level ARC Training

Instead of next-token prediction (which memorizes token patterns but
doesn't learn spatial reasoning), this trains the model to directly
predict output grids:
    - Each cell → classification over 10 ARC colors
    - Each grid → predicted (height, width) dimensions
    - Loss: per-cell cross-entropy + dimension prediction loss

The core transformer still processes tokenized context via in-context
learning, but the output is decoded by a GridPredictionHead instead
of an autoregressive language model head.
"""

import argparse
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime

from model import OctoTetrahedralModel
from config import Config
from grid_head import GridPredictionHead, grid_loss, predict_grid, MAX_GRID_SIZE
from data.arc_grid_dataset import ARCGridDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_data(tokenizer, max_seq_len: int, curriculum: bool = False):
    """Build grid-level ARC datasets."""
    datasets = []

    # Primary: ARC-AGI-2
    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI-2" / "data"
    if not (arc_dir / "training").exists():
        arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"

    if (arc_dir / "training").exists():
        ds = ARCGridDataset(
            data_dir=str(arc_dir), split="training",
            tokenizer=tokenizer, max_seq_len=max_seq_len,
            curriculum=curriculum
        )
        datasets.append(ds)
        logger.info(f"ARC training: {len(ds)} grid samples from {arc_dir}")

    # Also try ARC-AGI-1 for more data
    arc1_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"
    if arc1_dir != arc_dir and (arc1_dir / "training").exists():
        ds1 = ARCGridDataset(
            data_dir=str(arc1_dir), split="training",
            tokenizer=tokenizer, max_seq_len=max_seq_len,
            curriculum=curriculum
        )
        datasets.append(ds1)
        logger.info(f"ARC-AGI-1 training: {len(ds1)} grid samples")

    if not datasets:
        raise RuntimeError("No ARC data found!")

    combined = torch.utils.data.ConcatDataset(datasets)
    logger.info(f"Total grid training samples: {len(combined)}")

    # Validation
    val_datasets = []
    for d in [arc_dir, arc1_dir]:
        if d.exists() and (d / "evaluation").exists():
            vds = ARCGridDataset(
                data_dir=str(d), split="evaluation",
                tokenizer=tokenizer, max_seq_len=max_seq_len
            )
            val_datasets.append(vds)
            logger.info(f"Validation: {len(vds)} grid samples from {d}")

    val_ds = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None
    return combined, val_ds


def evaluate_grid_accuracy(model, grid_head, val_dl, device, use_bf16=True, max_batches=50):
    """Evaluate grid prediction accuracy on validation set."""
    model.eval()
    grid_head.eval()

    total_cells = 0
    correct_cells = 0
    total_dims = 0
    correct_dims = 0
    total_loss = 0
    n_batches = 0

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for batch in val_dl:
            if n_batches >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            target_grid = batch['target_grid'].to(device)
            grid_mask = batch['grid_mask'].to(device)
            target_h = batch['target_h'].to(device)
            target_w = batch['target_w'].to(device)

            # Forward through backbone
            output = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = output['hidden_states']

            # Grid prediction
            pred = grid_head(hidden)
            losses = grid_loss(pred, target_grid, target_h, target_w, grid_mask)
            total_loss += losses['total'].item()

            # Cell accuracy
            pred_colors = pred['grid_logits'].argmax(dim=-1)  # [batch, H, W]
            correct = (pred_colors == target_grid) & grid_mask
            correct_cells += correct.sum().item()
            total_cells += grid_mask.sum().item()

            # Dimension accuracy
            pred_h = pred['dim_logits'][:, 0, :].argmax(dim=-1) + 1
            pred_w = pred['dim_logits'][:, 1, :].argmax(dim=-1) + 1
            correct_dims += ((pred_h == target_h) & (pred_w == target_w)).sum().item()
            total_dims += target_h.shape[0]

            n_batches += 1

    cell_acc = correct_cells / max(1, total_cells)
    dim_acc = correct_dims / max(1, total_dims)
    avg_loss = total_loss / max(1, n_batches)

    model.train()
    grid_head.train()
    return avg_loss, cell_acc, dim_acc


def main():
    parser = argparse.ArgumentParser(description="OctoTetrahedral Grid-Level ARC Training")
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grid-lr', type=float, default=1e-3,
                        help="Learning rate for grid head (higher since training from scratch)")
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from checkpoint (backbone weights)")
    parser.add_argument('--freeze-backbone', action='store_true',
                        help="Freeze transformer backbone, only train grid head")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/grid')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=2000)
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Build model
    cfg = Config()
    model = OctoTetrahedralModel(cfg)
    model.to(device)

    # Load backbone weights if resuming
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        loaded = 0
        for k, v in state.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                model.state_dict()[k].copy_(v)
                loaded += 1
        logger.info(f"Loaded {loaded}/{len(state)} backbone params from {args.resume}")

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen — only training grid head")

    # Grid prediction head
    grid_head = GridPredictionHead(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=3,
        num_heads=4,
        max_grid=MAX_GRID_SIZE
    ).to(device)
    logger.info(f"Grid head: {sum(p.numel() for p in grid_head.parameters()):,} params")

    # Tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Data
    use_bf16 = device.type == 'cuda'
    train_ds, val_ds = build_data(tokenizer, args.max_seq_len, args.curriculum)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.curriculum,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_dl = None
    if val_ds:
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

    logger.info(f"GPU memory after model load: "
                f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if device.type == 'cuda' else "")

    # Optimizer: separate LR for backbone vs grid head
    param_groups = []
    if not args.freeze_backbone:
        param_groups.append({
            'params': [p for p in model.parameters() if p.requires_grad],
            'lr': args.lr
        })
    param_groups.append({
        'params': grid_head.parameters(),
        'lr': args.grid_lr
    })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = args.max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    grad_accum = args.grad_accum
    logger.info(f"Starting grid training: {args.max_steps} steps, batch {args.batch_size}, "
                f"grad_accum={grad_accum} (effective batch={args.batch_size * grad_accum})")

    model.train()
    grid_head.train()

    # Use GradScaler for mixed precision with unfrozen backbone
    scaler = torch.amp.GradScaler('cuda', enabled=(use_bf16 and not args.freeze_backbone))
    global_step = 0
    running_cell_loss = 0.0
    running_dim_loss = 0.0
    best_val_cell_acc = 0.0
    start_time = time.time()
    micro_step = 0

    while global_step < args.max_steps:
        for batch in train_dl:
            if global_step >= args.max_steps:
                break

            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            target_grid = batch['target_grid'].to(device)
            grid_mask = batch['grid_mask'].to(device)
            target_h = batch['target_h'].to(device)
            target_w = batch['target_w'].to(device)

            # Forward through backbone + grid head
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                if args.freeze_backbone:
                    with torch.no_grad():
                        output = model(input_ids=input_ids, attention_mask=attn_mask)
                    hidden = output['hidden_states'].detach()
                else:
                    output = model(input_ids=input_ids, attention_mask=attn_mask)
                    # Detach hidden states to break backbone graph when not needed
                    # Only keep gradient path through hidden_states
                    hidden = output['hidden_states']
                pred = grid_head(hidden)
                losses = grid_loss(pred, target_grid, target_h, target_w, grid_mask)
                loss = losses['total'] / grad_accum

            # Cache loss values before backward frees the graph
            cell_loss_val = losses['cell_loss'].item()
            dim_loss_val = losses['dim_loss'].item()

            loss.backward()
            # Free computation graph references immediately
            del output, hidden, pred, losses, loss
            if device.type == 'cuda' and micro_step % (grad_accum * 2) == 0:
                torch.cuda.empty_cache()
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            # NaN gradient check — skip step if any NaN
            all_params = list(model.parameters()) + list(grid_head.parameters())
            has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in all_params
            )
            if has_nan:
                logger.warning(f"NaN gradient detected at step {global_step}, skipping")
                optimizer.zero_grad()
                micro_step = 0
                continue

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_cell_loss += cell_loss_val
            running_dim_loss += dim_loss_val
            global_step += 1

            # Periodic GPU cache cleanup
            if global_step % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

            # Logging
            if global_step % args.log_interval == 0:
                avg_cell = running_cell_loss / args.log_interval
                avg_dim = running_dim_loss / args.log_interval
                elapsed = time.time() - start_time
                sps = global_step / elapsed if elapsed > 0 else 0
                lr_now = optimizer.param_groups[-1]['lr']
                mem = torch.cuda.memory_allocated()/1024**3 if device.type == 'cuda' else 0
                logger.info(
                    f"Step {global_step}/{args.max_steps} | "
                    f"CellLoss: {avg_cell:.4f} | DimLoss: {avg_dim:.4f} | "
                    f"LR: {lr_now:.2e} | Speed: {sps:.2f} steps/s | GPU: {mem:.1f}GB"
                )
                running_cell_loss = 0.0
                running_dim_loss = 0.0

            # Save checkpoint
            if global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = ckpt_dir / f"grid_step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'grid_head_state_dict': grid_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                    'config': 'default',
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # Validation
            if val_dl and global_step > 0 and global_step % args.eval_interval == 0:
                val_loss, cell_acc, dim_acc = evaluate_grid_accuracy(
                    model, grid_head, val_dl, device, use_bf16
                )
                logger.info(
                    f"Validation | Loss: {val_loss:.4f} | "
                    f"Cell Acc: {cell_acc:.4f} ({cell_acc*100:.1f}%) | "
                    f"Dim Acc: {dim_acc:.4f} ({dim_acc*100:.1f}%)"
                )
                if cell_acc > best_val_cell_acc:
                    best_val_cell_acc = cell_acc
                    best_path = ckpt_dir / "grid_best.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'grid_head_state_dict': grid_head.state_dict(),
                        'step': global_step,
                        'cell_acc': cell_acc,
                        'dim_acc': dim_acc,
                        'val_loss': val_loss,
                    }, best_path)
                    logger.info(f"New best! cell_acc={cell_acc:.4f}")
                model.train()
                grid_head.train()

    # Final save
    final_path = ckpt_dir / "grid_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'grid_head_state_dict': grid_head.state_dict(),
        'step': global_step,
    }, final_path)
    logger.info(f"Training complete. Final checkpoint: {final_path}")
    logger.info(f"Best validation cell accuracy: {best_val_cell_acc:.4f}")


if __name__ == "__main__":
    main()

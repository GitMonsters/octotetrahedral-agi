"""
OctoTetrahedral AGI — Direct Grid-to-Grid ARC Training (v2)

Key fixes over train_arc.py:
1. Direct grid encoding (10-color embeddings, not text tokens)
2. Per-cell cross-entropy loss (not next-token prediction)
3. Proper data augmentation (rotation, flip, color permutation)
4. Curriculum learning (easy → hard)
5. Consistent train/val pipeline (same loss computation)

The model sees: [example_in_1, example_out_1, ..., test_in] → predicts test_out
Each grid cell is an integer 0-9 (10 ARC colors).
"""

import os
import json
import math
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ─── Grid utilities ────────────────────────────────────────────────

NUM_COLORS = 11  # 0-9 = ARC colors, 10 = PAD
PAD = 10
MAX_GRID = 15  # Most ARC grids fit in 15x15; larger ones are cropped
MAX_EXAMPLES = 3  # max training examples per task (keeps sequence manageable)


def pad_grid(grid: List[List[int]], h: int, w: int) -> torch.Tensor:
    """Pad grid to (h, w) with PAD token. Returns LongTensor."""
    t = torch.full((h, w), PAD, dtype=torch.long)
    gh, gw = len(grid), len(grid[0]) if grid else 0
    for r in range(min(gh, h)):
        for c in range(min(gw, w)):
            t[r, c] = grid[r][c]
    return t


def augment_grid_pair(
    inp: List[List[int]], out: List[List[int]], rng: random.Random
) -> Tuple[List[List[int]], List[List[int]]]:
    """Random augmentation: rotation, flip, color permutation."""
    import numpy as np
    inp_arr = np.array(inp)
    out_arr = np.array(out)

    # Random rotation (0, 90, 180, 270)
    k = rng.randint(0, 3)
    if k > 0:
        inp_arr = np.rot90(inp_arr, k)
        out_arr = np.rot90(out_arr, k)

    # Random flip
    if rng.random() < 0.5:
        inp_arr = np.fliplr(inp_arr)
        out_arr = np.fliplr(out_arr)
    if rng.random() < 0.5:
        inp_arr = np.flipud(inp_arr)
        out_arr = np.flipud(out_arr)

    # Color permutation (preserving 0 = background)
    if rng.random() < 0.5:
        perm = list(range(10))
        non_zero = perm[1:]
        rng.shuffle(non_zero)
        perm[1:] = non_zero
        inp_arr = np.vectorize(lambda x: perm[x])(inp_arr)
        out_arr = np.vectorize(lambda x: perm[x])(out_arr)

    return inp_arr.tolist(), out_arr.tolist()


def task_difficulty(data: dict) -> float:
    """Score task difficulty for curriculum learning."""
    total_cells = 0
    colors = set()
    for ex in data['train']:
        for grid in [ex['input'], ex['output']]:
            h, w = len(grid), len(grid[0]) if grid else 0
            total_cells += h * w
            for row in grid:
                colors.update(row)
    return total_cells + len(colors) * 10


# ─── Dataset ───────────────────────────────────────────────────────

class ARCGridDataset(Dataset):
    """
    Direct grid-to-grid ARC dataset.
    
    Each sample encodes:
    - N training examples (input+output grids) as context
    - 1 test input grid
    - Target: test output grid
    
    All grids padded to MAX_GRID x MAX_GRID.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'training',
        max_examples: int = MAX_EXAMPLES,
        augment: bool = True,
        curriculum: bool = False,
        seed: int = 42,
    ):
        self.max_examples = max_examples
        self.augment = augment
        self.rng = random.Random(seed)

        # Load tasks
        task_dir = Path(data_dir) / split
        self.tasks = []
        for f in sorted(task_dir.glob('*.json')):
            with open(f) as fh:
                data = json.load(fh)
            self.tasks.append((f.stem, data))

        if curriculum:
            self.tasks.sort(key=lambda t: task_difficulty(t[1]))

        # Flatten to (task_idx, test_idx) samples
        self.samples = []
        for i, (tid, data) in enumerate(self.tasks):
            for j in range(len(data['test'])):
                self.samples.append((i, j))

        log.info(f"Loaded {len(self.tasks)} tasks, {len(self.samples)} samples ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task_idx, test_idx = self.samples[idx]
        tid, data = self.tasks[task_idx]

        train_exs = data['train'][:self.max_examples]
        test_pair = data['test'][test_idx]

        # Augment
        if self.augment:
            augmented_train = []
            for ex in train_exs:
                ai, ao = augment_grid_pair(ex['input'], ex['output'], self.rng)
                augmented_train.append({'input': ai, 'output': ao})
            ti, to = augment_grid_pair(
                test_pair['input'], test_pair.get('output', test_pair['input']),
                self.rng
            )
            train_exs = augmented_train
            test_input = ti
            test_output = to
        else:
            test_input = test_pair['input']
            test_output = test_pair.get('output', test_pair['input'])

        # Encode all grids as padded tensors
        context_grids = []  # list of (h, w, grid_tensor)
        for ex in train_exs:
            context_grids.append(pad_grid(ex['input'], MAX_GRID, MAX_GRID))
            context_grids.append(pad_grid(ex['output'], MAX_GRID, MAX_GRID))

        # Pad to max_examples * 2 context grids
        while len(context_grids) < self.max_examples * 2:
            context_grids.append(torch.full((MAX_GRID, MAX_GRID), PAD, dtype=torch.long))

        # Stack context: (num_context_grids, H, W)
        context = torch.stack(context_grids)  # (2*max_examples, 30, 30)

        # Test input & output
        test_in = pad_grid(test_input, MAX_GRID, MAX_GRID)
        test_out = pad_grid(test_output, MAX_GRID, MAX_GRID)

        # Actual output dimensions (for masking loss)
        out_h = len(test_output)
        out_w = len(test_output[0]) if test_output else 0

        return {
            'task_id': tid,
            'context': context,       # (2*max_ex, 30, 30)
            'test_input': test_in,     # (30, 30)
            'test_output': test_out,   # (30, 30)
            'out_h': out_h,
            'out_w': out_w,
        }


def collate_fn(batch):
    return {
        'task_id': [b['task_id'] for b in batch],
        'context': torch.stack([b['context'] for b in batch]),
        'test_input': torch.stack([b['test_input'] for b in batch]),
        'test_output': torch.stack([b['test_output'] for b in batch]),
        'out_h': torch.tensor([b['out_h'] for b in batch]),
        'out_w': torch.tensor([b['out_w'] for b in batch]),
    }


# ─── Model: ARC Grid Transformer ──────────────────────────────────

class ARCGridModel(nn.Module):
    """
    Transformer that processes ARC grids directly.
    
    Architecture:
    1. Color embedding (11 → d_model)
    2. Flatten all context grids + test input into token sequence
    3. Add positional embeddings (grid position + which-grid)
    4. Transformer encoder
    5. Extract test output region → per-cell classification (10 colors)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_grids: int = MAX_EXAMPLES * 2 + 1,  # context + test_input
        grid_size: int = MAX_GRID,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.max_grids = max_grids

        # Color embedding
        self.color_embed = nn.Embedding(NUM_COLORS, d_model)

        # Position embeddings
        self.row_embed = nn.Embedding(grid_size, d_model)
        self.col_embed = nn.Embedding(grid_size, d_model)
        self.grid_embed = nn.Embedding(max_grids + 1, d_model)  # which grid
        self.role_embed = nn.Embedding(4, d_model)  # 0=ctx_in, 1=ctx_out, 2=test_in, 3=test_out_query

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predict color for each cell
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 10),  # 10 ARC colors (no PAD in output)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode_grid(self, grid: torch.Tensor, grid_idx: int, role: int) -> torch.Tensor:
        """
        Encode a single grid into token embeddings.
        grid: (B, H, W) LongTensor
        Returns: (B, H*W, d_model)
        """
        B, H, W = grid.shape
        device = grid.device

        # Color embeddings
        tokens = self.color_embed(grid)  # (B, H, W, d)

        # Position embeddings
        rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
        tokens = tokens + self.row_embed(rows) + self.col_embed(cols)

        # Grid index embedding
        gi = torch.tensor(grid_idx, device=device)
        tokens = tokens + self.grid_embed(gi)

        # Role embedding
        ri = torch.tensor(role, device=device)
        tokens = tokens + self.role_embed(ri)

        # Flatten spatial dims
        return tokens.view(B, H * W, self.d_model)

    def forward(
        self,
        context: torch.Tensor,      # (B, num_ctx, H, W)
        test_input: torch.Tensor,    # (B, H, W)
        test_output: torch.Tensor = None,  # (B, H, W) — targets
        out_h: torch.Tensor = None,  # (B,) actual output heights
        out_w: torch.Tensor = None,  # (B,) actual output widths
    ):
        B = context.shape[0]
        num_ctx = context.shape[1]
        device = context.device

        # Encode context grids
        all_tokens = []
        for i in range(num_ctx):
            role = 0 if i % 2 == 0 else 1  # alternating input/output
            grid_tokens = self.encode_grid(context[:, i], grid_idx=i, role=role)
            all_tokens.append(grid_tokens)

        # Encode test input
        test_in_tokens = self.encode_grid(test_input, grid_idx=num_ctx, role=2)
        all_tokens.append(test_in_tokens)

        # Query tokens for test output (learnable based on position)
        query_grid = torch.full((B, MAX_GRID, MAX_GRID), PAD, dtype=torch.long, device=device)
        query_tokens = self.encode_grid(query_grid, grid_idx=num_ctx + 1, role=3)
        all_tokens.append(query_tokens)

        # Concatenate all tokens
        x = torch.cat(all_tokens, dim=1)  # (B, total_tokens, d_model)

        # Create attention mask: don't attend to PAD tokens
        # For efficiency, we skip masking and rely on the model learning to ignore PAD
        # (PAD has its own embedding that the model learns to treat as empty)

        # Transformer
        x = self.transformer(x)  # (B, total_tokens, d_model)

        # Extract query region (last MAX_GRID*MAX_GRID tokens)
        query_out = x[:, -MAX_GRID * MAX_GRID:, :]  # (B, 900, d_model)
        query_out = query_out.view(B, MAX_GRID, MAX_GRID, self.d_model)

        # Predict colors
        logits = self.output_head(query_out)  # (B, 30, 30, 10)

        result = {'logits': logits}

        if test_output is not None:
            # Compute loss only on actual output cells (not PAD)
            mask = test_output != PAD  # (B, 30, 30)

            # Flatten for cross-entropy
            logits_flat = logits.view(-1, 10)       # (B*30*30, 10)
            targets_flat = test_output.view(-1)      # (B*30*30,)
            mask_flat = mask.view(-1)                # (B*30*30,)

            # Clamp targets to valid range for non-masked positions
            targets_clamped = targets_flat.clamp(0, 9)

            # Masked loss
            if mask_flat.any():
                loss = F.cross_entropy(
                    logits_flat[mask_flat],
                    targets_clamped[mask_flat]
                )
            else:
                loss = torch.tensor(0.0, device=device)

            # Cell accuracy
            preds = logits.argmax(dim=-1)  # (B, 30, 30)
            correct = ((preds == test_output) & mask).float().sum()
            total = mask.float().sum()
            cell_acc = correct / total.clamp(min=1)

            # Exact match (per sample)
            exact = 0
            for b in range(B):
                if out_h is not None and out_w is not None:
                    h, w = out_h[b].item(), out_w[b].item()
                    pred_crop = preds[b, :h, :w]
                    target_crop = test_output[b, :h, :w]
                    if (pred_crop == target_crop).all():
                        exact += 1
                else:
                    b_mask = mask[b]
                    if b_mask.any() and ((preds[b] == test_output[b]) | ~b_mask).all():
                        exact += 1

            result['loss'] = loss
            result['cell_accuracy'] = cell_acc.item()
            result['exact_match'] = exact / B

        return result


# ─── Trainer ───────────────────────────────────────────────────────

class GridTrainer:
    def __init__(
        self,
        model: ARCGridModel,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        device: str = None,
        checkpoint_dir: str = 'checkpoints/arc_grid',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=lr * 0.01)

        self.step = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accs = []

    def train(self):
        log.info(f"Training on {self.device} | {self.model.get_num_params():,} params")
        log.info(f"Train: {len(self.train_loader.dataset)} samples | "
                 f"Val: {len(self.val_loader.dataset) if self.val_loader else 0} samples")

        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_exact = 0.0
        log_interval = 10

        start = time.time()

        while self.step < self.max_steps:
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break

                # Warmup LR
                if self.step < self.warmup_steps:
                    lr_scale = (self.step + 1) / self.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = pg['lr'] * lr_scale / max(lr_scale, 1e-8) if self.step > 0 else pg['lr']

                # Move to device
                ctx = batch['context'].to(self.device)
                t_in = batch['test_input'].to(self.device)
                t_out = batch['test_output'].to(self.device)
                o_h = batch['out_h'].to(self.device)
                o_w = batch['out_w'].to(self.device)

                # Forward
                result = self.model(ctx, t_in, t_out, o_h, o_w)
                loss = result['loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    log.warning(f"Step {self.step}: NaN/Inf loss, skipping")
                    self.optimizer.zero_grad()
                    self.step += 1
                    continue

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                running_acc += result['cell_accuracy']
                running_exact += result['exact_match']
                self.step += 1

                # Log
                if self.step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    avg_acc = running_acc / log_interval
                    avg_exact = running_exact / log_interval
                    elapsed = time.time() - start
                    sps = self.step / elapsed

                    log.info(
                        f"Step {self.step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Cell Acc: {avg_acc:.3f} | "
                        f"Exact: {avg_exact:.3f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"{sps:.1f} step/s"
                    )
                    self.train_losses.append(avg_loss)
                    running_loss = 0.0
                    running_acc = 0.0
                    running_exact = 0.0

                # Validate
                if self.step % 100 == 0 and self.val_loader:
                    val_result = self.validate()
                    log.info(
                        f"  VAL | Loss: {val_result['loss']:.4f} | "
                        f"Cell Acc: {val_result['cell_acc']:.3f} | "
                        f"Exact: {val_result['exact']:.3f}"
                    )
                    self.val_accs.append(val_result['cell_acc'])

                    if val_result['cell_acc'] > self.best_val_acc:
                        self.best_val_acc = val_result['cell_acc']
                        self.save('best_grid.pt')
                        log.info(f"  New best! Cell Acc: {self.best_val_acc:.3f}")

                # Checkpoint
                if self.step % 500 == 0:
                    self.save(f'grid_step_{self.step}.pt')

        # Final
        self.save('grid_final.pt')
        if self.val_loader:
            final = self.validate()
            log.info(f"FINAL | Cell Acc: {final['cell_acc']:.3f} | Exact: {final['exact']:.3f}")
        log.info(f"Best val cell accuracy: {self.best_val_acc:.3f}")

    @torch.no_grad()
    def validate(self, max_batches: int = 50):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_exact = 0.0
        n = 0
        for batch in self.val_loader:
            if n >= max_batches:
                break
            ctx = batch['context'].to(self.device)
            t_in = batch['test_input'].to(self.device)
            t_out = batch['test_output'].to(self.device)
            o_h = batch['out_h'].to(self.device)
            o_w = batch['out_w'].to(self.device)

            result = self.model(ctx, t_in, t_out, o_h, o_w)
            total_loss += result['loss'].item()
            total_acc += result['cell_accuracy']
            total_exact += result['exact_match']
            n += 1

        self.model.train()
        return {
            'loss': total_loss / max(n, 1),
            'cell_acc': total_acc / max(n, 1),
            'exact': total_exact / max(n, 1),
        }

    def save(self, name: str):
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_accs': self.val_accs,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.step = ckpt.get('step', 0)
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        log.info(f"Resumed from {path} (step {self.step})")


# ─── Entry point ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train ARC Grid Transformer (v2)')
    parser.add_argument('--data-dir', default='ARC_AMD_TRANSFER/data/ARC-AGI/data')
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ARC Grid Transformer v2 — Direct Grid-to-Grid Training")
    log.info("=" * 60)

    # Data
    train_ds = ARCGridDataset(
        args.data_dir, split='training',
        augment=not args.no_augment, curriculum=args.curriculum,
    )
    val_ds = ARCGridDataset(
        args.data_dir, split='evaluation',
        augment=False, curriculum=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = ARCGridModel(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
    )
    log.info(f"Model: {model.get_num_params():,} parameters")
    log.info(f"Config: d={args.d_model}, layers={args.num_layers}, heads={args.nhead}")

    # Trainer
    trainer = GridTrainer(
        model, train_loader, val_loader,
        lr=args.lr, max_steps=args.max_steps,
    )

    if args.resume:
        trainer.load(args.resume)

    if args.eval_only:
        result = trainer.validate(max_batches=200)
        log.info(f"Eval: Loss={result['loss']:.4f} CellAcc={result['cell_acc']:.3f} Exact={result['exact']:.3f}")
    else:
        trainer.train()


if __name__ == '__main__':
    main()

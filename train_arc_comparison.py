#!/usr/bin/env python3
"""Focused ARC training + evaluation script for MoE+SOA comparison."""

import json
import os
import glob
import time
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import OctoTetrahedralModel
from config import get_config

# ARC token encoding
# 0-9: grid colors, 10: row sep, 11: input/output sep, 12: example sep
VOCAB_SIZE = 13
ROW_SEP = 10
IO_SEP = 11
EX_SEP = 12


def grid_to_tokens(grid: list[list[int]]) -> list[int]:
    tokens = []
    for row in grid:
        tokens.extend(row)
        tokens.append(ROW_SEP)
    return tokens


def task_to_sequence(task: dict) -> list[int]:
    """Convert an ARC task to a flat token sequence: train pairs then test."""
    seq = []
    for pair in task['train']:
        seq.extend(grid_to_tokens(pair['input']))
        seq.append(IO_SEP)
        seq.extend(grid_to_tokens(pair['output']))
        seq.append(EX_SEP)
    # Add test input
    seq.extend(grid_to_tokens(task['test'][0]['input']))
    seq.append(IO_SEP)
    # Add test output (target)
    seq.extend(grid_to_tokens(task['test'][0]['output']))
    return seq


class ARCTokenDataset(Dataset):
    """Tokenizes ARC tasks as sequences for next-token prediction."""

    def __init__(self, data_dir: str, seq_len: int = 128):
        self.seq_len = seq_len
        self.sequences = []
        for path in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
            with open(path) as f:
                task = json.load(f)
            seq = task_to_sequence(task)
            # Chunk into seq_len windows with stride
            stride = seq_len // 2
            for i in range(0, max(1, len(seq) - seq_len), stride):
                chunk = seq[i:i + seq_len + 1]  # +1 for target shift
                if len(chunk) >= 4:
                    self.sequences.append(chunk)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Pad if needed
        if len(seq) < self.seq_len + 1:
            seq = seq + [0] * (self.seq_len + 1 - len(seq))
        x = torch.tensor(seq[:self.seq_len], dtype=torch.long)
        y = torch.tensor(seq[1:self.seq_len + 1], dtype=torch.long)
        return x, y


def create_model(mode: str = 'moe_soa', compound_loops: int = 2) -> OctoTetrahedralModel:
    """Create model with specified config."""
    config = get_config()
    config.compound_loop.enabled = True
    config.compound_loop.max_loops = compound_loops

    if mode == 'moe_soa':
        config.moe.enabled = True
        config.moe.compound_enabled = True
        config.moe.expert_ffn_dim = 512
    else:
        config.moe.enabled = False
        config.moe.compound_enabled = False

    return OctoTetrahedralModel(config, use_geometric_physics=False)


def train(
    mode: str = 'moe_soa',
    steps: int = 500,
    batch_size: int = 2,
    seq_len: int = 128,
    lr: float = 1.5e-4,
    warmup_steps: int = 50,
    save_path: str = None,
    log_interval: int = 25,
):
    """Train model and return final metrics."""
    data_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/training/'
    dataset = ARCTokenDataset(data_dir, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = create_model(mode, compound_loops=2)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f'\n{"═" * 55}')
    print(f'  TRAINING: {mode.upper()} | {total_params/1e6:.1f}M params')
    print(f'  {steps} steps | batch={batch_size} | seq={seq_len} | lr={lr}')
    print(f'  Dataset: {len(dataset)} chunks from 400 ARC tasks')
    print(f'{"═" * 55}')

    data_iter = iter(loader)
    losses = []
    start_time = time.time()

    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        result = model(x, labels=y)
        loss = result['loss']
        if mode == 'moe_soa' and result.get('moe_aux_loss') is not None:
            loss = loss + 0.001 * result['moe_aux_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(result['loss'].item())

        if step % log_interval == 0 or step == 1:
            avg_loss = sum(losses[-log_interval:]) / len(losses[-log_interval:])
            elapsed = time.time() - start_time
            eta = elapsed / step * (steps - step)
            # Token accuracy on this batch
            with torch.no_grad():
                preds = result['logits'].argmax(dim=-1)
                acc = (preds == y).float().mean().item() * 100
            cur_lr = scheduler.get_last_lr()[0]
            print(f'  step {step:>4}/{steps}  loss={avg_loss:.3f}  acc={acc:.1f}%  lr={cur_lr:.2e}  [{elapsed:.0f}s, ETA {eta:.0f}s]')

    # Save checkpoint
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'mode': mode,
            'steps': steps,
            'final_loss': losses[-1],
            'losses': losses,
        }, save_path)
        print(f'  Saved: {save_path} ({os.path.getsize(save_path)/1e6:.1f}MB)')

    return model, losses


def evaluate(model: OctoTetrahedralModel, data_dir: str, num_tasks: int = 20, label: str = '') -> dict:
    """Evaluate on ARC tasks. Returns metrics dict."""
    model.eval()
    task_files = sorted(glob.glob(os.path.join(data_dir, '*.json')))[:num_tasks]

    total_tokens = 0
    correct_tokens = 0
    total_cells = 0
    correct_cells = 0
    total_loss = 0.0

    with torch.no_grad():
        for path in task_files:
            with open(path) as f:
                task = json.load(f)

            seq = task_to_sequence(task)
            # Find where test output starts (after last IO_SEP)
            last_sep = len(seq)
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] == IO_SEP:
                    last_sep = i + 1
                    break

            target_tokens = seq[last_sep:]
            if len(target_tokens) < 2:
                continue

            # Use everything up to test output as context
            context = seq[:last_sep]
            full = context + target_tokens

            # Truncate to fit
            max_len = 256
            if len(full) > max_len + 1:
                full = full[-(max_len + 1):]
                context_len = len(full) - len(target_tokens)
            else:
                context_len = len(context)

            x = torch.tensor([full[:len(full)-1]], dtype=torch.long)
            y = torch.tensor([full[1:]], dtype=torch.long)

            result = model(x, labels=y)
            total_loss += result['loss'].item()

            # Accuracy only on the target portion
            preds = result['logits'].argmax(dim=-1)[0]
            target_start = max(0, context_len - 1)
            pred_target = preds[target_start:].tolist()
            true_target = target_tokens[:len(pred_target)]

            for p, t in zip(pred_target, true_target):
                total_tokens += 1
                if p == t:
                    correct_tokens += 1
                # Cell accuracy (only color tokens 0-9)
                if t <= 9:
                    total_cells += 1
                    if p == t:
                        correct_cells += 1

    metrics = {
        'token_acc': correct_tokens / max(1, total_tokens) * 100,
        'cell_acc': correct_cells / max(1, total_cells) * 100,
        'avg_loss': total_loss / max(1, len(task_files)),
        'num_tasks': len(task_files),
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
    }

    if label:
        print(f'\n  {label}:')
        print(f'    Loss:      {metrics["avg_loss"]:.3f}')
        print(f'    Token acc: {metrics["correct_tokens"]}/{metrics["total_tokens"]} ({metrics["token_acc"]:.1f}%)')
        print(f'    Cell acc:  {correct_cells}/{total_cells} ({metrics["cell_acc"]:.1f}%)')

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['soa', 'moe_soa', 'both'], default='both')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-tasks', type=int, default=20)
    args = parser.parse_args()

    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/'

    if args.eval_only:
        for mode in (['soa', 'moe_soa'] if args.mode == 'both' else [args.mode]):
            model = create_model(mode, compound_loops=3)
            ckpt_path = f'checkpoints/{mode}_{args.steps}.pt'
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, weights_only=False)
                model.load_state_dict(sd['model_state'])
                print(f'Loaded {ckpt_path}')
            evaluate(model, eval_dir, args.eval_tasks, f'{mode.upper()} (trained)')
    else:
        modes = ['soa', 'moe_soa'] if args.mode == 'both' else [args.mode]
        for mode in modes:
            model, losses = train(
                mode=mode,
                steps=args.steps,
                save_path=f'checkpoints/{mode}_{args.steps}.pt',
            )
            evaluate(model, eval_dir, args.eval_tasks, f'{mode.upper()} (after {args.steps} steps)')
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

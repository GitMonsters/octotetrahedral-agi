"""
OctoTetrahedral AGI — ARC Training Script

Trains the 7B/70B MoE model on ARC-AGI + synthetic ARC tasks.
Combines real ARC training data (400 tasks) with synthetic data (up to 10K tasks).

Usage:
    # Local smoke test (MPS/CPU, small batch)
    python train_arc_moe.py --config 7b --max-steps 100 --batch-size 1 --device mps

    # Full training on GPU
    python train_arc_moe.py --config 7b --max-steps 50000 --batch-size 32 --device cuda:0

    # Resume from checkpoint
    python train_arc_moe.py --config 7b --resume checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Synthetic JSON dataset adapter
# ────────────────────────────────────────────────────────────────

class SyntheticARCDataset(Dataset):
    """Load synthetic ARC tasks from a flat JSON dict (synthetic_arc_dataset_*.json)."""

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_seq_len: int = 512,
        max_tasks: Optional[int] = None,
    ):
        from data.arc_dataset import ARCTask, grid_to_tokens

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(json_path) as f:
            raw = json.load(f)

        self.samples: List[Dict] = []
        for i, (task_id, task_data) in enumerate(raw.items()):
            if max_tasks and i >= max_tasks:
                break
            task = ARCTask(task_id, task_data)
            for test_idx in range(task.num_test):
                input_text, target_text = task.format_compact(test_idx, include_answer=True)
                self.samples.append({
                    'task_id': task_id,
                    'input_text': input_text,
                    'target_text': target_text,
                })

        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        input_tokens = self.tokenizer.encode(s['input_text'])
        target_tokens = self.tokenizer.encode(s['target_text'])

        if len(input_tokens) + len(target_tokens) > self.max_seq_len:
            max_input = self.max_seq_len - len(target_tokens)
            if max_input <= 0:
                full_tokens = target_tokens[-self.max_seq_len:]
                input_length = 0
            else:
                input_tokens = input_tokens[-max_input:]
                full_tokens = input_tokens + target_tokens
                input_length = len(input_tokens)
        else:
            full_tokens = input_tokens + target_tokens
            input_length = len(input_tokens)

        input_ids = torch.tensor(full_tokens[:-1])
        labels = torch.tensor(full_tokens[1:])
        prompt_label_len = max(0, min(input_length - 1, len(full_tokens) - 1))
        if prompt_label_len > 0:
            labels[:prompt_label_len] = -100
        if (labels != -100).sum().item() == 0:
            labels = torch.tensor(full_tokens[1:])

        return {
            'task_id': s['task_id'],
            'input_ids': input_ids,
            'labels': labels,
            'input_length': min(input_length, len(full_tokens) - 1),
        }


# ────────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        length = len(item['input_ids'])
        padding = max_len - length
        input_ids.append(torch.cat([item['input_ids'], torch.full((padding,), pad_token_id)]))
        labels.append(torch.cat([item['labels'], torch.full((padding,), -100)]))
        attention_mask.append(torch.cat([torch.ones(length), torch.zeros(padding)]))
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask).long(),
        'task_ids': [item['task_id'] for item in batch],
    }


# ────────────────────────────────────────────────────────────────
# Build datasets
# ────────────────────────────────────────────────────────────────

def build_datasets(tokenizer, max_seq_len: int = 512):
    """Build combined ARC + synthetic training dataset."""
    from data.arc_dataset import ARCDataset

    datasets = []

    # Real ARC training data
    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"
    if (arc_dir / "training").exists():
        arc_train = ARCDataset(
            data_dir=str(arc_dir),
            split='training',
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            augment=True,
        )
        logger.info(f"ARC training: {len(arc_train)} samples")
        datasets.append(arc_train)
    else:
        logger.warning(f"ARC data not found at {arc_dir}")

    # Synthetic ARC datasets
    root = Path.home()
    for synth_file in sorted(root.glob("synthetic_arc_dataset_*.json")):
        synth = SyntheticARCDataset(str(synth_file), tokenizer, max_seq_len)
        datasets.append(synth)

    if not datasets:
        raise RuntimeError("No training data found!")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    logger.info(f"Total training samples: {len(combined)}")
    return combined


def build_val_dataset(tokenizer, max_seq_len: int = 512):
    """Build validation dataset from ARC evaluation split."""
    from data.arc_dataset import ARCDataset

    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"
    if (arc_dir / "evaluation").exists():
        val_ds = ARCDataset(
            data_dir=str(arc_dir),
            split='evaluation',
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            augment=False,
        )
        logger.info(f"ARC validation: {len(val_ds)} samples")
        return val_ds
    else:
        logger.warning("No ARC evaluation data found; skipping validation")
        return None


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train OctoTetrahedral MoE on ARC")
    p.add_argument("--config", type=str, default="7b", choices=["default", "7b", "70b", "1.72t"])
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    from train_distributed import load_config
    from model import OctoTetrahedralModel
    from train import Trainer

    config = load_config(args.config)

    # Override training params
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.log_interval:
        config.training.log_interval = args.log_interval
    if args.eval_interval:
        config.training.eval_interval = args.eval_interval
    if args.save_interval:
        config.training.save_interval = args.save_interval
    if args.device:
        config.device = args.device

    # Tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Datasets
    train_ds = build_datasets(tokenizer, args.max_seq_len)
    val_ds = build_val_dataset(tokenizer, args.max_seq_len)

    batch_size = config.training.batch_size
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b),
        drop_last=True,
    )
    val_dl = None
    if val_ds:
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b),
        )

    # Model
    logger.info(f"Building {args.config} model...")
    model = OctoTetrahedralModel(config, use_geometric_physics=False)
    total = model.get_num_params()
    active = model.get_active_params()
    logger.info(f"Model: {total/1e9:.2f}B total, {active/1e9:.2f}B active")

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Train
    trainer = Trainer(
        model, config, train_dl, val_dl,
        gradient_checkpointing=True,
        mixed_precision=(args.device.startswith('cuda')),
    )

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    logger.info(f"Starting ARC training: {args.config} config, {args.max_steps} steps, batch {batch_size}")
    logger.info(f"Training data: {len(train_ds)} samples")
    if val_ds:
        logger.info(f"Validation data: {len(val_ds)} samples")

    trainer.train(max_steps=args.max_steps)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

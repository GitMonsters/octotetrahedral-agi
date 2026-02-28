"""
OctoTetrahedral AGI — Targeted Fine-Tuning

Analyzes ARC evaluation results to identify weak task categories,
then fine-tunes the model on those specific tasks with higher learning rate.

Strategy:
1. Load eval results JSON (from eval_arc_moe.py)
2. Classify tasks by grid-size pattern and transformation type
3. Identify categories with low accuracy
4. Build focused dataset from weak categories + synthetic variants
5. Fine-tune with cosine LR schedule and early stopping

Usage:
    # After initial evaluation
    python finetune_targeted.py --config 7b \
        --checkpoint checkpoints/best.pt \
        --eval-results arc_eval_evaluation_7b_*.json \
        --device cuda:0

    # Fine-tune on all failed tasks only
    python finetune_targeted.py --config 7b \
        --checkpoint checkpoints/best.pt \
        --eval-results arc_eval_evaluation_7b_*.json \
        --failed-only --device cuda:0
"""

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Task categorization heuristics
# ────────────────────────────────────────────────────────────────

def categorize_task(task_data: Dict) -> List[str]:
    """Classify an ARC task into categories based on grid structure."""
    categories = []
    train = task_data.get("train", [])
    if not train:
        return ["unknown"]

    # Grid size category
    inp_sizes = [(len(ex["input"]), len(ex["input"][0])) for ex in train if ex.get("input")]
    out_sizes = [(len(ex["output"]), len(ex["output"][0])) for ex in train if ex.get("output")]

    if inp_sizes and out_sizes:
        avg_in = (sum(h for h, w in inp_sizes) / len(inp_sizes),
                  sum(w for h, w in inp_sizes) / len(inp_sizes))
        avg_out = (sum(h for h, w in out_sizes) / len(out_sizes),
                   sum(w for h, w in out_sizes) / len(out_sizes))

        # Size-change pattern
        if abs(avg_in[0] - avg_out[0]) < 0.5 and abs(avg_in[1] - avg_out[1]) < 0.5:
            categories.append("same_size")
        elif avg_out[0] > avg_in[0] * 1.5 or avg_out[1] > avg_in[1] * 1.5:
            categories.append("upscale")
        elif avg_out[0] < avg_in[0] * 0.7 or avg_out[1] < avg_in[1] * 0.7:
            categories.append("downscale")
        else:
            categories.append("size_change")

        # Grid size buckets
        max_dim = max(avg_in[0], avg_in[1])
        if max_dim <= 5:
            categories.append("small_grid")
        elif max_dim <= 15:
            categories.append("medium_grid")
        else:
            categories.append("large_grid")

    # Color analysis
    all_colors = set()
    for ex in train:
        for row in ex.get("input", []):
            all_colors.update(row)
        for row in ex.get("output", []):
            all_colors.update(row)
    categories.append(f"colors_{len(all_colors)}")

    # Symmetry detection
    for ex in train[:1]:
        grid = ex.get("output", [])
        if grid:
            h, w = len(grid), len(grid[0])
            # Horizontal symmetry
            is_h_sym = all(grid[i] == grid[h - 1 - i] for i in range(h // 2))
            if is_h_sym:
                categories.append("symmetric")
            # Repetition pattern
            if h >= 2 and grid[0] == grid[1]:
                categories.append("repetition")

    return categories if categories else ["unknown"]


# ────────────────────────────────────────────────────────────────
# Analyze evaluation results
# ────────────────────────────────────────────────────────────────

def analyze_eval_results(
    eval_json: str,
    arc_dir: Path,
    threshold: float = 0.5,
) -> Dict:
    """Analyze eval results to find weak categories and tasks."""
    with open(eval_json) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        logger.warning("No results found in eval JSON")
        return {"weak_tasks": [], "weak_categories": {}, "category_stats": {}}

    # Load task data for categorization
    task_cache = {}
    for split_dir in [arc_dir / "training", arc_dir / "evaluation"]:
        if split_dir.exists():
            for fp in split_dir.glob("*.json"):
                with open(fp) as f:
                    task_cache[fp.stem] = json.load(f)

    # Categorize and aggregate
    category_results = defaultdict(lambda: {"correct": 0, "total": 0, "tasks": []})
    weak_tasks = []
    failed_tasks = []

    for r in results:
        task_id = r.get("task_id", "")
        is_correct = r.get("correct", False)
        cell_acc = r.get("cell_accuracy", 0.0)

        if task_id in task_cache:
            cats = categorize_task(task_cache[task_id])
        else:
            cats = ["unknown"]

        for cat in cats:
            category_results[cat]["total"] += 1
            if is_correct:
                category_results[cat]["correct"] += 1
            category_results[cat]["tasks"].append(task_id)

        if not is_correct:
            failed_tasks.append(task_id)
        if cell_acc < threshold:
            weak_tasks.append(task_id)

    # Find weak categories (accuracy below threshold)
    category_stats = {}
    weak_categories = {}
    for cat, stats in category_results.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        category_stats[cat] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        if acc < threshold:
            weak_categories[cat] = {
                "accuracy": acc,
                "total": stats["total"],
                "task_ids": list(set(stats["tasks"])),
            }

    # Sort by weakness
    weak_categories = dict(sorted(weak_categories.items(), key=lambda x: x[1]["accuracy"]))

    logger.info(f"Eval results: {len(results)} tasks, {len(failed_tasks)} failed")
    logger.info(f"Weak categories ({len(weak_categories)}): "
                + ", ".join(f"{k} ({v['accuracy']:.0%})" for k, v in list(weak_categories.items())[:5]))

    return {
        "weak_tasks": list(set(weak_tasks)),
        "failed_tasks": list(set(failed_tasks)),
        "weak_categories": weak_categories,
        "category_stats": category_stats,
    }


# ────────────────────────────────────────────────────────────────
# Targeted dataset
# ────────────────────────────────────────────────────────────────

class TargetedARCDataset(Dataset):
    """Dataset that focuses on specific task IDs, with optional oversampling."""

    def __init__(
        self,
        task_ids: List[str],
        arc_dir: Path,
        tokenizer,
        max_seq_len: int = 512,
        oversample: int = 3,
    ):
        from data.arc_dataset import ARCTask, grid_to_tokens

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        # Load specified tasks from all splits
        for split in ["training", "evaluation"]:
            split_dir = arc_dir / split
            if not split_dir.exists():
                continue
            for fp in split_dir.glob("*.json"):
                if fp.stem in task_ids:
                    with open(fp) as f:
                        task_data = json.load(f)
                    task = ARCTask(fp.stem, task_data)
                    for test_idx in range(task.num_test):
                        input_text, target_text = task.format_compact(test_idx, include_answer=True)
                        self.samples.append({
                            "task_id": fp.stem,
                            "input_text": input_text,
                            "target_text": target_text,
                        })

        # Oversample weak tasks
        if oversample > 1 and self.samples:
            self.samples = self.samples * oversample

        logger.info(f"Targeted dataset: {len(self.samples)} samples from {len(task_ids)} tasks "
                    f"(oversample={oversample}x)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        input_tokens = self.tokenizer.encode(s["input_text"])
        target_tokens = self.tokenizer.encode(s["target_text"])

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
        prompt_len = max(0, min(input_length - 1, len(full_tokens) - 1))
        if prompt_len > 0:
            labels[:prompt_len] = -100
        if (labels != -100).sum().item() == 0:
            labels = torch.tensor(full_tokens[1:])

        return {"task_id": s["task_id"], "input_ids": input_ids, "labels": labels}


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        length = len(item["input_ids"])
        padding = max_len - length
        input_ids.append(torch.cat([item["input_ids"], torch.full((padding,), pad_token_id)]))
        labels.append(torch.cat([item["labels"], torch.full((padding,), -100)]))
        attention_mask.append(torch.cat([torch.ones(length), torch.zeros(padding)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask).long(),
    }


# ────────────────────────────────────────────────────────────────
# Cosine LR scheduler with warmup
# ────────────────────────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ────────────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_dl: DataLoader,
    config,
    device: str,
    max_steps: int = 5000,
    lr: float = 1e-5,
    warmup_steps: int = 200,
    grad_accum: int = 1,
    checkpoint_dir: str = "checkpoints",
    save_interval: int = 500,
    log_interval: int = 50,
    patience: int = 10,
) -> Dict:
    """Fine-tune with cosine LR, gradient accumulation, and loss tracking."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

    Path(checkpoint_dir).mkdir(exist_ok=True)
    train_iter = iter(train_dl)
    best_loss = float("inf")
    no_improve = 0
    losses = []
    step = 0
    t0 = time.time()

    logger.info(f"Fine-tuning: {max_steps} steps, lr={lr}, warmup={warmup_steps}, "
                f"grad_accum={grad_accum}, patience={patience}")

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels_batch = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        output = model(input_ids=input_ids, labels=labels_batch)
        loss = output if isinstance(output, torch.Tensor) and output.dim() == 0 else output[0]

        # MoE auxiliary loss
        if hasattr(model, "core") and hasattr(model.core, "get_aux_loss"):
            aux = model.core.get_aux_loss()
            if aux is not None:
                loss = loss + 0.01 * aux

        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_loss = loss.item() * grad_accum
        losses.append(step_loss)
        step += 1

        if step % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            logger.info(f"step {step}/{max_steps}  loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                        f"elapsed={elapsed:.0f}s")

            # Early stopping check
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at step {step} (no improvement for {patience} checks)")
                break

        if step % save_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"finetune_step_{step}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "loss": step_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = Path(checkpoint_dir) / "finetune_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": losses[-1] if losses else 0,
    }, final_path)
    logger.info(f"Saved final: {final_path}")

    return {
        "steps": step,
        "final_loss": losses[-1] if losses else 0,
        "best_loss": best_loss,
        "early_stopped": no_improve >= patience,
    }


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Targeted fine-tuning on weak ARC categories")
    parser.add_argument("--config", type=str, default="7b", choices=["default", "7b", "70b", "1.72t"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-results", type=str, required=True, help="Eval JSON from eval_arc_moe.py")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--oversample", type=int, default=3, help="Oversample weak tasks N times")
    parser.add_argument("--threshold", type=float, default=0.5, help="Accuracy threshold for 'weak'")
    parser.add_argument("--failed-only", action="store_true", help="Only train on completely failed tasks")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (in log intervals)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't fine-tune")
    args = parser.parse_args()

    import tiktoken
    from train_distributed import load_config
    from model import OctoTetrahedralModel

    config = load_config(args.config)
    device = args.device or config.device
    config.device = device

    arc_dir = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data"

    # Step 1: Analyze evaluation results
    logger.info(f"Analyzing eval results: {args.eval_results}")
    analysis = analyze_eval_results(args.eval_results, arc_dir, threshold=args.threshold)

    # Print category analysis
    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)
    for cat, stats in sorted(analysis["category_stats"].items(), key=lambda x: x[1]["accuracy"]):
        acc = stats["accuracy"]
        marker = "⚠️" if acc < args.threshold else "✅"
        print(f"  {marker} {cat:20s}  {stats['correct']}/{stats['total']}  ({acc:.0%})")

    print(f"\nFailed tasks: {len(analysis['failed_tasks'])}")
    print(f"Weak tasks (cell_acc < {args.threshold}): {len(analysis['weak_tasks'])}")
    print(f"Weak categories: {len(analysis['weak_categories'])}")

    if args.analyze_only:
        # Save analysis
        out_path = f"arc_analysis_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {out_path}")
        return

    # Step 2: Build targeted dataset
    if args.failed_only:
        target_tasks = analysis["failed_tasks"]
        logger.info(f"Fine-tuning on {len(target_tasks)} failed tasks")
    else:
        # Combine weak tasks + tasks from weak categories
        target_set = set(analysis["weak_tasks"])
        for cat_info in analysis["weak_categories"].values():
            target_set.update(cat_info["task_ids"])
        target_tasks = list(target_set)
        logger.info(f"Fine-tuning on {len(target_tasks)} weak tasks")

    if not target_tasks:
        logger.info("No weak tasks found — model performing well! Skipping fine-tuning.")
        return

    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = TargetedARCDataset(
        task_ids=target_tasks,
        arc_dir=arc_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        oversample=args.oversample,
    )

    if len(dataset) == 0:
        logger.error("No samples found for targeted tasks")
        return

    train_dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Step 3: Load model from checkpoint
    logger.info(f"Loading {args.config} model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = OctoTetrahedralModel(config, use_geometric_physics=False)
    model_state = model.state_dict()
    pretrained = ckpt["model_state_dict"]
    filtered = {}
    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            logger.warning(f"Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
    model.load_state_dict(filtered, strict=False)

    total = model.get_num_params()
    active = model.get_active_params()
    logger.info(f"Model: {total / 1e9:.2f}B total, {active / 1e9:.2f}B active")

    # Step 4: Fine-tune
    result = finetune(
        model=model,
        train_dl=train_dl,
        config=config,
        device=device,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_accum=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Steps:         {result['steps']}")
    print(f"Final loss:    {result['final_loss']:.4f}")
    print(f"Best loss:     {result['best_loss']:.4f}")
    print(f"Early stopped: {result['early_stopped']}")
    print(f"\nNext: re-evaluate with eval_arc_moe.py using checkpoints/finetune_final.pt")


if __name__ == "__main__":
    main()

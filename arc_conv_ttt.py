#!/usr/bin/env python3
"""Per-task ConvNet Test-Time Training ARC solver.

For each ARC task:
1. Generate augmented training pairs (dihedral transforms + color permutations)
2. Train a small ConvNet from scratch on augmented examples
3. Predict test output(s) via majority voting

ConvNets have natural inductive bias for spatial patterns (translation invariance)
which helps generalize from few examples without pretraining.
"""
import os
import sys
import json
import time
import hashlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ─── Grid Encoding ───────────────────────────────────────────────────────────

MAX_GRID = 30
NUM_COLORS = 10  # ARC uses 0-9


def grid_to_tensor(grid: np.ndarray, max_h: int, max_w: int) -> torch.Tensor:
    """Encode grid as one-hot tensor: (10, max_h, max_w)."""
    h, w = grid.shape
    t = torch.zeros(NUM_COLORS, max_h, max_w)
    for c in range(NUM_COLORS):
        t[c, :h, :w] = torch.tensor((grid == c).astype(np.float32))
    return t


def tensor_to_grid(t: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Decode one-hot tensor to grid."""
    return t[:, :h, :w].argmax(dim=0).numpy().astype(np.uint8)


# ─── Data Augmentation ───────────────────────────────────────────────────────

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0: return arr
    elif tid == 1: return np.rot90(arr, 1)
    elif tid == 2: return np.rot90(arr, 2)
    elif tid == 3: return np.rot90(arr, 3)
    elif tid == 4: return np.fliplr(arr)
    elif tid == 5: return np.flipud(arr)
    elif tid == 6: return arr.T
    elif tid == 7: return np.fliplr(np.rot90(arr, 1))
    return arr

DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

def inverse_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def augment_pairs(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    num_aug: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """Generate augmented (inp, out, transform_id, color_perm) tuples."""
    results = []
    seen = set()
    
    # Original pairs
    for inp, out in pairs:
        results.append((inp.copy(), out.copy(), 0, np.arange(10, dtype=np.uint8)))
    
    for _ in range(num_aug * 5):
        if len(results) >= num_aug + len(pairs):
            break
        
        # Random pair
        idx = np.random.randint(len(pairs))
        inp, out = pairs[idx]
        
        # Random dihedral
        tid = np.random.randint(0, 8)
        # Random color perm (keep 0 fixed)
        perm = np.arange(10, dtype=np.uint8)
        perm[1:] = np.random.permutation(perm[1:])
        
        a_inp = dihedral_transform(perm[inp], tid)
        a_out = dihedral_transform(perm[out], tid)
        
        if a_inp.shape[0] > MAX_GRID or a_inp.shape[1] > MAX_GRID:
            continue
        if a_out.shape[0] > MAX_GRID or a_out.shape[1] > MAX_GRID:
            continue
        
        h = hashlib.sha256(a_inp.tobytes() + a_out.tobytes()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            results.append((a_inp, a_out, tid, perm))
    
    return results


# ─── Model ───────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class ArcConvNet(nn.Module):
    """Small ConvNet for ARC grid transformation.
    
    Architecture:
    - Input: one-hot encoded grid (10, H, W)
    - Several residual conv blocks with varying receptive fields
    - Output: per-pixel color logits (10, H, W)
    """
    def __init__(self, hidden: int = 128, num_blocks: int = 6):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv2d(NUM_COLORS, hidden, 1)
        
        # Multi-scale feature extraction
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlock(hidden))
        
        # Large receptive field path (dilated convolutions)
        self.dilated = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        
        # Global context (pool + broadcast)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        
        # Output head (combine local + dilated + global)
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, NUM_COLORS, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 10, H, W) -> (B, 10, H, W) logits."""
        h = self.input_proj(x)
        
        # Local features
        local = h
        for block in self.blocks:
            local = block(local)
        
        # Dilated features
        dilated = self.dilated(h)
        
        # Global features
        glob = self.global_pool(h)
        glob = glob.expand_as(h)
        
        # Combine
        combined = torch.cat([local, dilated, glob], dim=1)
        return self.output_head(combined)


class SameSizeArcNet(nn.Module):
    """For tasks where output has same dimensions as input."""
    def __init__(self, hidden: int = 128, num_blocks: int = 6):
        super().__init__()
        self.net = ArcConvNet(hidden, num_blocks)
    
    def forward(self, x):
        return self.net(x)
    
    def predict(self, inp_grid: np.ndarray, device: torch.device) -> np.ndarray:
        h, w = inp_grid.shape
        x = grid_to_tensor(inp_grid, h, w).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = self.forward(x)
        return tensor_to_grid(logits[0].cpu(), h, w)


class DiffSizeArcNet(nn.Module):
    """For tasks where output may differ in dimensions from input.
    
    Uses an encoder-decoder with a learned size predictor.
    """
    def __init__(self, hidden: int = 128, max_out: int = 30):
        super().__init__()
        self.max_out = max_out
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_COLORS, hidden, 3, padding=1),
            nn.GELU(),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        
        # Pool to fixed size
        self.pool = nn.AdaptiveAvgPool2d(8)
        
        # Size predictor
        self.size_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 64, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # (out_h, out_w)
        )
        
        # Decoder (generates output from encoded features)
        self.decoder_fc = nn.Linear(hidden * 64, hidden * max_out * max_out)
        self.decoder_conv = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            nn.Conv2d(hidden, NUM_COLORS, 1),
        )
    
    def forward(self, x, out_h=None, out_w=None):
        B = x.shape[0]
        enc = self.encoder(x)
        pooled = self.pool(enc)
        
        # Size prediction
        size_logits = self.size_head(pooled)
        
        # Decode
        flat = pooled.flatten(1)
        dec = self.decoder_fc(flat).reshape(B, -1, self.max_out, self.max_out)
        logits = self.decoder_conv(dec)
        
        return logits, size_logits


# ─── Per-Task Training ───────────────────────────────────────────────────────

def analyze_task(task: dict) -> dict:
    """Analyze task to determine properties."""
    same_size = True
    sizes = []
    out_sizes = []
    
    for pair in task["train"]:
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        sizes.append(inp.shape)
        out_sizes.append(out.shape)
        if inp.shape != out.shape:
            same_size = False
    
    # Check if output size is constant
    const_out_size = len(set(out_sizes)) == 1
    
    return {
        "same_size": same_size,
        "const_out_size": const_out_size,
        "out_sizes": out_sizes,
        "max_h": max(max(s[0] for s in sizes), max(s[0] for s in out_sizes)),
        "max_w": max(max(s[1] for s in sizes), max(s[1] for s in out_sizes)),
    }


def train_same_size(
    task: dict,
    device: torch.device,
    num_aug: int = 100,
    num_steps: int = 1000,
    lr: float = 1e-3,
    hidden: int = 128,
    verbose: bool = False,
) -> nn.Module:
    """Train a ConvNet for same-size transformation tasks."""
    # Prepare pairs
    pairs = []
    for p in task["train"]:
        inp = np.array(p["input"], dtype=np.uint8)
        out = np.array(p["output"], dtype=np.uint8)
        pairs.append((inp, out))
    
    aug_pairs = augment_pairs(pairs, num_aug=num_aug)
    
    # Find max dimensions
    max_h = max(a[0].shape[0] for a in aug_pairs)
    max_w = max(a[0].shape[1] for a in aug_pairs)
    
    # Build tensors
    inputs = torch.stack([grid_to_tensor(a[0], max_h, max_w) for a in aug_pairs]).to(device)
    targets = torch.stack([
        torch.tensor(
            np.pad(a[1], ((0, max_h - a[1].shape[0]), (0, max_w - a[1].shape[1])),
                   constant_values=-1).astype(np.int64)
        ) for a in aug_pairs
    ]).to(device)
    
    n = len(aug_pairs)
    if verbose:
        print(f"  Training ConvNet on {n} augmented pairs, grid={max_h}x{max_w}")
    
    model = SameSizeArcNet(hidden=hidden, num_blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.01)
    
    batch_size = min(32, n)
    model.train()
    
    for step in range(num_steps):
        idx = torch.randint(0, n, (batch_size,))
        x = inputs[idx]
        y = targets[idx]
        
        logits = model(x)  # (B, 10, H, W)
        loss = F.cross_entropy(logits, y, ignore_index=-1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        if verbose and step % 200 == 0:
            with torch.inference_mode():
                pred = logits[0].argmax(dim=0).cpu().numpy()
                tgt = y[0].cpu().numpy()
                mask = tgt >= 0
                acc = (pred[mask] == tgt[mask]).mean()
            print(f"    Step {step}: loss={loss.item():.4f} acc={acc:.3f}")
    
    return model, max_h, max_w


def predict_same_size(
    model: nn.Module,
    task: dict,
    device: torch.device,
    max_h: int,
    max_w: int,
    num_vote: int = 32,
) -> List[np.ndarray]:
    """Predict test outputs using voting across augmentations."""
    model.eval()
    predictions = []
    
    for test_pair in task["test"]:
        test_inp = np.array(test_pair["input"], dtype=np.uint8)
        th, tw = test_inp.shape
        
        # Accumulate per-cell votes using logits
        vote_logits = np.zeros((NUM_COLORS, th, tw), dtype=np.float64)
        vote_count = 0
        
        for _ in range(num_vote):
            tid = np.random.randint(0, 8)
            perm = np.arange(10, dtype=np.uint8)
            perm[1:] = np.random.permutation(perm[1:])
            inv_perm = np.argsort(perm).astype(np.uint8)
            
            aug_inp = dihedral_transform(perm[test_inp], tid)
            ah, aw = aug_inp.shape
            
            if ah > MAX_GRID or aw > MAX_GRID:
                continue
            
            # Pad to model size
            pad_h = max(max_h, ah)
            pad_w = max(max_w, aw)
            
            x = grid_to_tensor(aug_inp, pad_h, pad_w).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                logits = model(x)  # (1, 10, pad_h, pad_w)
            
            # Extract and inverse augment
            pred_logits = logits[0, :, :ah, :aw].cpu().numpy()  # (10, ah, aw)
            
            # Inverse color permutation on logit channels
            inv_logits = np.zeros_like(pred_logits)
            for c in range(10):
                inv_logits[inv_perm[c]] += pred_logits[c]
            
            # Inverse dihedral
            inv_pred_logits = np.stack([inverse_dihedral(inv_logits[c], tid) for c in range(10)])
            
            if inv_pred_logits.shape[1:] == (th, tw):
                vote_logits += inv_pred_logits
                vote_count += 1
        
        if vote_count > 0:
            pred = vote_logits.argmax(axis=0).astype(np.uint8)
        else:
            pred = test_inp.copy()
            vote_logits = np.zeros((NUM_COLORS, th, tw), dtype=np.float64)
        
        predictions.append((pred, vote_logits))
    
    return predictions


# ─── Full Solver ─────────────────────────────────────────────────────────────

def solve_task(
    task: dict,
    device: torch.device,
    num_aug: int = 100,
    num_steps: int = 1000,
    num_vote: int = 32,
    verbose: bool = False,
) -> List[np.ndarray]:
    """Solve a single ARC task using per-task ConvNet training."""
    info = analyze_task(task)
    
    # Train model
    model, max_h, max_w = train_same_size(
        task, device, num_aug=num_aug, num_steps=num_steps, verbose=verbose
    )
    
    # Self-validate on training examples
    model.eval()
    train_correct = 0
    for pair in task["train"]:
        inp = np.array(pair["input"], dtype=np.uint8)
        exp = np.array(pair["output"], dtype=np.uint8)
        ih, iw = inp.shape if info["same_size"] else exp.shape
        
        x = grid_to_tensor(inp, max(max_h, inp.shape[0]), max(max_w, inp.shape[1])).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(x)
        pred = logits[0, :, :ih, :iw].argmax(dim=0).cpu().numpy().astype(np.uint8)
        
        if pred.shape == exp.shape and np.array_equal(pred, exp):
            train_correct += 1
    
    if verbose:
        print(f"    Train self-validation: {train_correct}/{len(task['train'])}")
    
    # If model doesn't get training examples right, retrain with more steps
    if train_correct < len(task["train"]) and num_steps < 3000:
        if verbose:
            print(f"    Retraining with 2x steps...")
        model, max_h, max_w = train_same_size(
            task, device, num_aug=num_aug * 2, num_steps=num_steps * 2, verbose=verbose
        )
    
    results = predict_same_size(model, task, device, max_h, max_w, num_vote=num_vote)
    predictions = [r[0] for r in results]
    confidences = [r[1] for r in results]
    
    # For extraction tasks (output smaller than input), try to auto-crop
    if not info["same_size"] and info["const_out_size"]:
        out_h, out_w = info["out_sizes"][0]
        cropped_preds = []
        cropped_confs = []
        for pred, conf in zip(predictions, confidences):
            if pred.shape[0] >= out_h and pred.shape[1] >= out_w:
                cropped_preds.append(pred[:out_h, :out_w])
                cropped_confs.append(conf[:, :out_h, :out_w])
            else:
                cropped_preds.append(pred)
                cropped_confs.append(conf)
        predictions = cropped_preds
        confidences = cropped_confs
    
    return predictions, confidences


def evaluate_arc(
    data_dir: str,
    device: torch.device,
    num_aug: int = 100,
    num_steps: int = 1000,
    num_vote: int = 32,
    max_tasks: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Evaluate on ARC dataset."""
    tasks = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            task_id = fname[:-5]
            with open(os.path.join(data_dir, fname)) as f:
                tasks[task_id] = json.load(f)
    
    task_items = list(tasks.items())[:max_tasks] if max_tasks else list(tasks.items())
    
    print(f"Evaluating {len(task_items)} tasks with ConvNet TTT (aug={num_aug}, steps={num_steps})")
    
    results = {}
    correct = 0
    total = 0
    
    for i, (task_id, task) in enumerate(task_items):
        t0 = time.time()
        
        try:
            predictions, confidences = solve_task(
                task, device,
                num_aug=num_aug,
                num_steps=num_steps,
                num_vote=num_vote,
                verbose=verbose,
            )
            
            task_correct = True
            for j, (pred, test_pair) in enumerate(zip(predictions, task["test"])):
                expected = np.array(test_pair["output"], dtype=np.uint8) if "output" in test_pair else None
                total += 1
                
                if expected is not None and pred.shape == expected.shape and np.array_equal(pred, expected):
                    correct += 1
                else:
                    task_correct = False
                    if verbose and expected is not None:
                        if pred.shape == expected.shape:
                            cell_match = (pred == expected).sum()
                            cell_total = expected.size
                            print(f"    Test {j}: {cell_match}/{cell_total} cells ({cell_match/cell_total:.1%})")
                        else:
                            print(f"    Test {j}: shape mismatch {pred.shape} vs {expected.shape}")
            
            elapsed = time.time() - t0
            status = "✓" if task_correct else "✗"
            print(f"[{i+1}/{len(task_items)}] {task_id}: {status} ({elapsed:.1f}s) [{correct}/{total}]")
            
            results[task_id] = {
                "predictions": [p.tolist() for p in predictions],
                "correct": task_correct,
                "time": elapsed,
            }
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(task_items)}] {task_id}: ERROR ({e}) ({elapsed:.1f}s)")
            results[task_id] = {"error": str(e), "correct": False, "time": elapsed}
            total += len(task["test"])
    
    accuracy = correct / max(total, 1)
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} ({accuracy:.1%})")
    print(f"{'='*60}")
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "per_task": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConvNet per-task TTT ARC solver")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_aug", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_vote", type=int, default=32)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--output", type=str, default="conv_ttt_results.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    results = evaluate_arc(
        args.data_dir, device,
        num_aug=args.num_aug,
        num_steps=args.num_steps,
        num_vote=args.num_vote,
        max_tasks=args.max_tasks,
        verbose=True,
    )
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")

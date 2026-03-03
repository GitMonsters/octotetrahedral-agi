#!/usr/bin/env python3
"""
Iterative refinement ConvNet for ARC-AGI-2.

Key idea (inspired by TRM):
1. Train ConvNet to predict output from input → get draft D1
2. Train SECOND ConvNet on (input + D1) → predict output → get D2
3. Repeat until convergence or max cycles

The refinement model sees both the original input AND the previous draft,
so it can learn to correct specific errors in the draft.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
from arc_conv_ttt import (
    NUM_COLORS, MAX_GRID, grid_to_tensor, tensor_to_grid,
    dihedral_transform, inverse_dihedral, augment_pairs, analyze_task,
    ResBlock
)


class RefinementNet(nn.Module):
    """ConvNet that takes (input, draft) and predicts refined output.
    Input: 20 channels (10 for original input + 10 for draft).
    Output: 10 channels (color logits).
    """
    def __init__(self, hidden: int = 128, num_blocks: int = 6):
        super().__init__()
        self.input_proj = nn.Conv2d(NUM_COLORS * 2, hidden, 1)
        
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_blocks)])
        
        self.dilated = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, NUM_COLORS, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 20, H, W) -> (B, 10, H, W) logits."""
        h = self.input_proj(x)
        
        local = h
        for block in self.blocks:
            local = block(local)
        
        dilated = self.dilated(h)
        glob = self.global_pool(h).expand_as(h)
        
        combined = torch.cat([local, dilated, glob], dim=1)
        return self.output_head(combined)


def grid_pair_to_tensor(inp: np.ndarray, draft: np.ndarray, max_h: int, max_w: int) -> torch.Tensor:
    """Encode (input, draft) as 20-channel tensor."""
    t_inp = grid_to_tensor(inp, max_h, max_w)
    t_draft = grid_to_tensor(draft, max_h, max_w)
    return torch.cat([t_inp, t_draft], dim=0)  # (20, max_h, max_w)


def train_refinement(
    task: dict,
    drafts_train: List[np.ndarray],
    device: torch.device,
    num_aug: int = 100,
    num_steps: int = 1000,
    lr: float = 3e-4,
    hidden: int = 128,
    verbose: bool = False,
) -> Tuple[nn.Module, int, int]:
    """Train refinement model on (input, corrupted_draft) → output pairs.
    
    Key insight: if training drafts are perfect, we CORRUPT them to simulate
    the kinds of errors the base model makes on test data.
    """
    pairs = [(np.array(ex['input'], dtype=np.uint8), np.array(ex['output'], dtype=np.uint8))
             for ex in task['train']]
    
    # Create (input, draft, output) triples
    triples = []
    for (inp, out), draft in zip(pairs, drafts_train):
        if draft.shape == out.shape:
            triples.append((inp, draft, out))
    
    if not triples:
        return None, 0, 0
    
    # Check if drafts are (near-)perfect — if so, corrupt them
    draft_errors = sum(
        (draft != out).sum()
        for (inp, draft, out) in triples
    )
    
    use_corruption = (draft_errors < 5 * len(triples))  # If avg < 5 errors per draft
    
    # Get the color palette from outputs
    all_colors = set()
    for _, _, out in triples:
        all_colors.update(out.flatten().tolist())
    color_list = sorted(all_colors)
    
    # Augment with dihedral transforms + color permutations + corruption
    aug_triples = []
    for _ in range(num_aug // 8 + 1):
        for inp, draft, out in triples:
            tid = np.random.randint(0, 8)
            perm = np.arange(10, dtype=np.uint8)
            perm[1:] = np.random.permutation(perm[1:])
            
            a_inp = dihedral_transform(perm[inp], tid)
            a_out = dihedral_transform(perm[out], tid)
            
            if use_corruption:
                # Create corrupted draft from output (not from model's draft)
                corrupted = a_out.copy()
                h, w = corrupted.shape
                n_cells = h * w
                # Corrupt 1-10% of cells randomly
                n_corrupt = max(1, np.random.randint(1, max(2, n_cells // 10)))
                for _ in range(n_corrupt):
                    ri = np.random.randint(0, h)
                    ci = np.random.randint(0, w)
                    # Replace with random color from palette or background
                    corrupted[ri, ci] = perm[np.random.choice(color_list)]
                a_draft = corrupted
            else:
                a_draft = dihedral_transform(perm[draft], tid)
            
            aug_triples.append((a_inp, a_draft, a_out))
    
    aug_triples = aug_triples[:num_aug]
    
    max_h = max(a[0].shape[0] for a in aug_triples)
    max_w = max(a[0].shape[1] for a in aug_triples)
    
    # Build tensors
    inputs = torch.stack([
        grid_pair_to_tensor(a[0], a[1], max_h, max_w) for a in aug_triples
    ]).to(device)
    
    targets = torch.stack([
        torch.tensor(
            np.pad(a[2], ((0, max_h - a[2].shape[0]), (0, max_w - a[2].shape[1])),
                   constant_values=-1).astype(np.int64)
        ) for a in aug_triples
    ]).to(device)
    
    n = len(aug_triples)
    if verbose:
        print(f"    Refinement: {n} augmented triples, grid={max_h}x{max_w}")
    
    model = RefinementNet(hidden=hidden, num_blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.01)
    
    batch_size = min(32, n)
    model.train()
    
    for step in range(num_steps):
        idx = torch.randint(0, n, (batch_size,))
        x = inputs[idx]
        y = targets[idx]
        
        logits = model(x)
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
            print(f"      Step {step}: loss={loss.item():.4f} acc={acc:.3f}")
    
    return model, max_h, max_w


def predict_refinement(
    model: nn.Module,
    task: dict,
    drafts_test: List[np.ndarray],
    device: torch.device,
    max_h: int,
    max_w: int,
    num_vote: int = 32,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Predict refined outputs using voting."""
    model.eval()
    results = []
    
    for test_pair, draft in zip(task['test'], drafts_test):
        test_inp = np.array(test_pair['input'], dtype=np.uint8)
        th, tw = test_inp.shape
        
        vote_logits = np.zeros((NUM_COLORS, th, tw), dtype=np.float64)
        vote_count = 0
        
        for _ in range(num_vote):
            tid = np.random.randint(0, 8)
            perm = np.arange(10, dtype=np.uint8)
            perm[1:] = np.random.permutation(perm[1:])
            inv_perm = np.argsort(perm).astype(np.uint8)
            
            aug_inp = dihedral_transform(perm[test_inp], tid)
            aug_draft = dihedral_transform(perm[draft], tid)
            ah, aw = aug_inp.shape
            
            if ah > MAX_GRID or aw > MAX_GRID:
                continue
            
            pad_h = max(max_h, ah)
            pad_w = max(max_w, aw)
            
            x = grid_pair_to_tensor(aug_inp, aug_draft, pad_h, pad_w).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                logits = model(x)
            
            pred_logits = logits[0, :, :ah, :aw].cpu().numpy()
            
            inv_logits = np.zeros_like(pred_logits)
            for c in range(10):
                inv_logits[inv_perm[c]] += pred_logits[c]
            
            inv_pred_logits = np.stack([inverse_dihedral(inv_logits[c], tid) for c in range(10)])
            
            if inv_pred_logits.shape[1:] == (th, tw):
                vote_logits += inv_pred_logits
                vote_count += 1
        
        if vote_count > 0:
            pred = vote_logits.argmax(axis=0).astype(np.uint8)
        else:
            pred = draft.copy()
        
        results.append((pred, vote_logits))
    
    return results


def solve_task_iterative(
    task: dict,
    device: torch.device,
    num_cycles: int = 3,
    num_aug: int = 150,
    num_steps: int = 1500,
    num_vote: int = 48,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Solve task with iterative refinement.
    
    Cycle 0: Standard ConvNet (input → output)
    Cycle 1+: Refinement ConvNet (input + draft → output)
    """
    from arc_conv_ttt import solve_task as base_solve_task
    
    # Cycle 0: Base prediction
    if verbose:
        print(f"  Cycle 0: Base ConvNet")
    predictions, confidences = base_solve_task(
        task, device,
        num_aug=num_aug, num_steps=num_steps, num_vote=num_vote,
        verbose=verbose,
    )
    
    # Get drafts on training examples for refinement training
    from arc_conv_ttt import train_same_size, predict_same_size
    
    # For training self-prediction, we need the model from cycle 0
    # Retrain quickly to get training predictions
    model0, mh0, mw0 = train_same_size(
        task, device, num_aug=num_aug, num_steps=num_steps, verbose=False
    )
    
    # Get training drafts (predict training inputs through the model)
    model0.eval()
    train_drafts = []
    for ex in task['train']:
        inp = np.array(ex['input'], dtype=np.uint8)
        ih, iw = inp.shape
        x = grid_to_tensor(inp, max(mh0, ih), max(mw0, iw)).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model0(x)
        draft = logits[0, :, :ih, :iw].argmax(dim=0).cpu().numpy().astype(np.uint8)
        train_drafts.append(draft)
    
    # Check if training drafts are perfect — if so, no point refining
    train_correct = sum(
        1 for draft, ex in zip(train_drafts, task['train'])
        if np.array_equal(draft, np.array(ex['output']))
    )
    
    if verbose:
        print(f"  Training draft accuracy: {train_correct}/{len(task['train'])}")
    
    # Refinement cycles
    current_preds = predictions
    best_preds = list(predictions)
    
    for cycle in range(1, num_cycles + 1):
        if verbose:
            print(f"  Cycle {cycle}: Refinement")
        
        # Train refinement model on (input, train_draft) → output
        ref_model, ref_mh, ref_mw = train_refinement(
            task, train_drafts, device,
            num_aug=num_aug, num_steps=num_steps,
            verbose=verbose,
        )
        
        if ref_model is None:
            if verbose:
                print(f"    Skipping refinement (no valid triples)")
            break
        
        # Predict refined outputs
        ref_results = predict_refinement(
            ref_model, task, current_preds, device,
            ref_mh, ref_mw, num_vote=num_vote,
        )
        
        refined_preds = [r[0] for r in ref_results]
        refined_confs = [r[1] for r in ref_results]
        
        # Check if refinement changed anything
        any_change = False
        for old, new in zip(current_preds, refined_preds):
            if not np.array_equal(old, new):
                any_change = True
                break
        
        if not any_change:
            if verbose:
                print(f"    No changes in cycle {cycle}, stopping")
            break
        
        if verbose:
            for i, (old, new) in enumerate(zip(current_preds, refined_preds)):
                diff = (old != new).sum()
                print(f"    Test {i}: {diff} cells changed")
        
        # Update for next cycle
        current_preds = refined_preds
        
        # Update training drafts for next cycle
        ref_model.eval()
        new_train_drafts = []
        for ex, old_draft in zip(task['train'], train_drafts):
            inp = np.array(ex['input'], dtype=np.uint8)
            ih, iw = inp.shape
            x = grid_pair_to_tensor(
                inp, old_draft, max(ref_mh, ih), max(ref_mw, iw)
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = ref_model(x)
            new_draft = logits[0, :, :ih, :iw].argmax(dim=0).cpu().numpy().astype(np.uint8)
            new_train_drafts.append(new_draft)
        train_drafts = new_train_drafts
    
    # Return best predictions with confidence
    return current_preds, confidences


def test_iterative():
    """Test iterative refinement on best near-miss eval tasks."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation")
    
    targets = [
        "e376de54",  # 3-4 wrong — best candidate
        "88e364bc",  # 6 wrong
        "dd6b8c4b",  # 7 wrong
        "409aa875",  # 9 wrong
    ]
    
    total_solved = 0
    total_tests = 0
    
    for task_id in targets:
        fpath = os.path.join(data_dir, f"{task_id}.json")
        if not os.path.exists(fpath):
            continue
        
        with open(fpath) as f:
            task = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        
        t0 = time.time()
        predictions, confidences = solve_task_iterative(
            task, device,
            num_cycles=2,
            num_aug=150,
            num_steps=1500,
            num_vote=48,
            verbose=True,
        )
        elapsed = time.time() - t0
        
        for ti, pred in enumerate(predictions):
            test_pair = task['test'][ti]
            total_tests += 1
            
            if 'output' in test_pair:
                expected = np.array(test_pair['output'], dtype=np.uint8)
                if pred.shape == expected.shape:
                    wrong = (pred != expected).sum()
                    total = expected.size
                    if wrong == 0:
                        print(f"  ✅ test{ti}: SOLVED!")
                        total_solved += 1
                    else:
                        print(f"  test{ti}: {wrong} wrong ({(total-wrong)/total:.1%})")
                        if wrong <= 10:
                            for r, c in zip(*np.where(pred != expected)):
                                print(f"    ({r},{c}): pred={pred[r,c]} exp={expected[r,c]}")
        
        print(f"  Time: {elapsed:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_solved}/{total_tests} exact matches")


if __name__ == "__main__":
    test_iterative()

"""
ARC Grid-Level Dataset for OctoTetrahedral AGI

Instead of tokenizing grids into flat text sequences for next-token prediction,
this dataset produces:
    - context_ids: tokenized training examples (input→output text)
    - target_grid: 2D tensor [max_H, max_W] with output cell colors (0-9)
    - grid_mask: bool tensor marking valid cells
    - target_h, target_w: output dimensions

The model sees the context as tokens (leveraging in-context learning),
then predicts the output grid via a dedicated grid head.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from data.arc_dataset import grid_to_tokens

MAX_GRID = 30
NUM_COLORS = 10


def _rotate_grid_90(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    h, w = len(grid), len(grid[0]) if grid else 0
    return [[grid[h - 1 - j][i] for j in range(h)] for i in range(w)]


def _flip_grid_h(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid horizontally (left-right)."""
    return [row[::-1] for row in grid]


def _flip_grid_v(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid vertically (top-bottom)."""
    return grid[::-1]


def _permute_colors(grid: List[List[int]], perm: List[int]) -> List[List[int]]:
    """Apply color permutation to grid."""
    return [[perm[c] for c in row] for row in grid]


def _apply_transform(grid: List[List[int]], rotation: int, flip_h: bool,
                     flip_v: bool, color_perm: Optional[List[int]] = None) -> List[List[int]]:
    """Apply geometric + color transform to a grid."""
    g = grid
    for _ in range(rotation):
        g = _rotate_grid_90(g)
    if flip_h:
        g = _flip_grid_h(g)
    if flip_v:
        g = _flip_grid_v(g)
    if color_perm is not None:
        g = _permute_colors(g, color_perm)
    return g


class ARCGridDataset(Dataset):
    """ARC dataset that produces grid-level targets for direct prediction."""

    def __init__(self, data_dir: str, split: str = "training",
                 tokenizer=None, max_seq_len: int = 2048,
                 max_grid: int = MAX_GRID, curriculum: bool = False,
                 augment: bool = False):
        self.max_seq_len = max_seq_len
        self.max_grid = max_grid
        self.tokenizer = tokenizer
        self.augment = augment
        self.samples = []

        data_path = Path(data_dir) / split
        if not data_path.exists():
            raise FileNotFoundError(f"ARC data not found: {data_path}")

        task_files = sorted(data_path.glob("*.json"))

        for tf in task_files:
            with open(tf) as f:
                task = json.load(f)

            train_examples = task.get("train", [])
            test_examples = task.get("test", [])

            for test_ex in test_examples:
                test_output = test_ex.get("output")
                if not test_output:
                    continue

                h = len(test_output)
                w = len(test_output[0]) if h > 0 else 0
                if h > max_grid or w > max_grid:
                    continue

                self.samples.append({
                    'task_id': tf.stem,
                    'train': train_examples,
                    'test_input': test_ex["input"],
                    'test_output': test_output,
                    'output_h': h,
                    'output_w': w,
                })

        if curriculum:
            self.samples.sort(key=lambda s: self._difficulty(s))

    @staticmethod
    def _difficulty(sample: Dict) -> float:
        """Difficulty score for curriculum learning."""
        total_cells = 0
        colors = set()
        for ex in sample['train']:
            for grid in [ex['input'], ex['output']]:
                for row in grid:
                    total_cells += len(row)
                    colors.update(row)
        return total_cells + len(colors) * 10

    def _format_context(self, train_examples: List[Dict],
                        test_input: List[List[int]]) -> str:
        """Format training examples + test input as text context."""
        parts = []
        for i, ex in enumerate(train_examples):
            inp = grid_to_tokens(ex['input'])
            out = grid_to_tokens(ex['output'])
            parts.append(f"[{inp}] -> [{out}]")

        test_inp = grid_to_tokens(test_input)
        parts.append(f"[{test_inp}] ->")
        return " ".join(parts)

    def _grid_to_tensor(self, grid: List[List[int]]) -> tuple:
        """Convert grid to padded tensor + mask.

        Returns:
            grid_tensor: [max_grid, max_grid] padded with 0
            mask: [max_grid, max_grid] bool, True for valid cells
        """
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        grid_tensor = torch.zeros(self.max_grid, self.max_grid, dtype=torch.long)
        mask = torch.zeros(self.max_grid, self.max_grid, dtype=torch.bool)

        for i in range(h):
            for j in range(w):
                grid_tensor[i, j] = grid[i][j]
                mask[i, j] = True

        return grid_tensor, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Apply augmentation: transform ALL grids consistently
        train_examples = sample['train']
        test_input = sample['test_input']
        test_output = sample['test_output']

        if self.augment:
            rotation = random.randint(0, 3)  # 0, 90, 180, 270
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            # Color permutation: keep 0 (background) fixed, permute 1-9
            color_perm = list(range(10))
            if random.random() < 0.5:
                non_bg = list(range(1, 10))
                random.shuffle(non_bg)
                color_perm[1:] = non_bg

            def xform(g):
                return _apply_transform(g, rotation, flip_h, flip_v, color_perm)

            train_examples = [
                {'input': xform(ex['input']), 'output': xform(ex['output'])}
                for ex in train_examples
            ]
            test_input = xform(test_input)
            test_output = xform(test_output)

        # Format context as text and tokenize
        context_text = self._format_context(train_examples, test_input)

        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(context_text)
        else:
            # Fallback: character-level
            tokens = [ord(c) % 100277 for c in context_text]

        # Truncate/pad to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Pad to max_seq_len
        if len(input_ids) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            attention_mask = torch.cat([
                torch.ones(len(input_ids), dtype=torch.long),
                torch.zeros(self.max_seq_len - len(input_ids), dtype=torch.long)
            ])
            input_ids = torch.cat([input_ids, padding])
        else:
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

        # Target grid (use potentially augmented output)
        target_grid, grid_mask = self._grid_to_tensor(test_output)
        out_h = len(test_output)
        out_w = len(test_output[0]) if out_h > 0 else 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_grid': target_grid,
            'grid_mask': grid_mask,
            'target_h': torch.tensor(out_h, dtype=torch.long),
            'target_w': torch.tensor(out_w, dtype=torch.long),
        }

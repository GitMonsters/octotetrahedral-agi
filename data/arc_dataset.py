"""
ARC-AGI Dataset and Training for OctoTetrahedral AGI

ARC (Abstraction and Reasoning Corpus) format:
- Each task has train examples (input/output pairs) and test examples
- Grids are 2D arrays with values 0-9 representing colors
- Model must learn the transformation rule from examples

Our approach:
1. Serialize grids to text format
2. Present as few-shot learning: "Example 1: [input] -> [output], Example 2: ... Test: [input] -> "
3. Train model to predict the output grid
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np

# Grid serialization formats
def grid_to_text(grid: List[List[int]], compact: bool = True) -> str:
    """Convert 2D grid to text representation"""
    if compact:
        # Compact: each row on one line, space-separated
        return '\n'.join(' '.join(str(c) for c in row) for row in grid)
    else:
        # Verbose: with brackets
        return str(grid)

def text_to_grid(text: str) -> List[List[int]]:
    """Convert text back to 2D grid"""
    lines = text.strip().split('\n')
    return [[int(c) for c in line.split()] for line in lines]

def grid_to_tokens(grid: List[List[int]]) -> str:
    """Convert grid to a flat token string with row separators"""
    rows = []
    for row in grid:
        rows.append(' '.join(str(c) for c in row))
    return ' | '.join(rows)

def tokens_to_grid(tokens: str) -> List[List[int]]:
    """Convert token string back to grid"""
    s = tokens.strip()
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]

    # Be permissive: models sometimes omit spaces or the exact ' | ' delimiter.
    rows = [r.strip() for r in s.split('|')]
    grid: List[List[int]] = []
    for row in rows:
        if not row:
            continue
        if ' ' in row:
            parts = [p for p in row.split() if p]
            grid.append([int(p) for p in parts])
        else:
            digits = [ch for ch in row if ch.isdigit()]
            grid.append([int(ch) for ch in digits])
    return grid


class ARCTask:
    """Single ARC task with train and test examples"""
    
    def __init__(self, task_id: str, data: Dict[str, Any]):
        self.task_id = task_id
        self.train_examples = data['train']
        self.test_examples = data['test']
    
    @property
    def num_train(self) -> int:
        return len(self.train_examples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_examples)
    
    def get_train_pair(self, idx: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Get (input, output) pair from training examples"""
        ex = self.train_examples[idx]
        return ex['input'], ex['output']
    
    def get_test_input(self, idx: int = 0) -> List[List[int]]:
        """Get test input grid"""
        return self.test_examples[idx]['input']
    
    def get_test_output(self, idx: int = 0) -> Optional[List[List[int]]]:
        """Get test output grid (if available)"""
        return self.test_examples[idx].get('output')
    
    def get_grid_size(self, grid: List[List[int]]) -> Tuple[int, int]:
        """Get (height, width) of grid"""
        return len(grid), len(grid[0]) if grid else 0
    
    def format_as_prompt(
        self,
        test_idx: int = 0,
        include_answer: bool = False,
        max_examples: int = 4
    ) -> str:
        """
        Format task as text prompt for language model.
        
        Format:
        Task: Learn the pattern from examples and apply to test input.
        
        Example 1:
        Input:
        0 0 5
        0 5 0
        Output:
        3 3 3
        4 4 4
        
        Example 2:
        ...
        
        Test Input:
        0 0 5
        5 0 0
        
        Test Output:
        """
        lines = ["Task: Learn the pattern from examples and apply to test input.\n"]
        
        # Add training examples
        num_ex = min(len(self.train_examples), max_examples)
        for i in range(num_ex):
            inp, out = self.get_train_pair(i)
            lines.append(f"Example {i+1}:")
            lines.append("Input:")
            lines.append(grid_to_text(inp))
            lines.append("Output:")
            lines.append(grid_to_text(out))
            lines.append("")
        
        # Add test input
        test_inp = self.get_test_input(test_idx)
        lines.append("Test Input:")
        lines.append(grid_to_text(test_inp))
        lines.append("")
        lines.append("Test Output:")
        
        prompt = '\n'.join(lines)
        
        if include_answer:
            test_out = self.get_test_output(test_idx)
            if test_out:
                prompt += '\n' + grid_to_text(test_out)
        
        return prompt
    
    def format_compact(
        self,
        test_idx: int = 0,
        include_answer: bool = False
    ) -> Tuple[str, str]:
        """
        Compact format for training.
        Returns (input_text, target_text)
        """
        parts = []
        
        # Training examples
        for i, ex in enumerate(self.train_examples):
            inp_str = grid_to_tokens(ex['input'])
            out_str = grid_to_tokens(ex['output'])
            parts.append(f"[{inp_str}]->[{out_str}]")
        
        # Test input
        test_inp = self.get_test_input(test_idx)
        inp_str = grid_to_tokens(test_inp)
        input_text = ' '.join(parts) + f" [{inp_str}]->["
        
        # Target
        test_out = self.get_test_output(test_idx)
        if test_out:
            target_text = grid_to_tokens(test_out) + "]"
        else:
            target_text = ""
        
        return input_text, target_text


class ARCDataset(Dataset):
    """PyTorch Dataset for ARC tasks"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'training',
        tokenizer = None,
        max_seq_len: int = 512,
        format_type: str = 'compact',  # 'compact' or 'verbose'
        augment: bool = True,
        seed: int = 42,
        curriculum: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.format_type = format_type
        self.augment = augment
        self.curriculum = curriculum
        self.rng = random.Random(seed)
        
        # Load all tasks
        self.tasks = self._load_tasks()

        # Curriculum: sort tasks easy→hard (by total grid cells across all examples)
        if curriculum:
            self.tasks.sort(key=lambda t: self._task_difficulty(t))
        
        # Create flat list of (task, test_idx) pairs
        self.samples = []
        for task in self.tasks:
            for test_idx in range(task.num_test):
                self.samples.append((task, test_idx))
    
    def _load_tasks(self) -> List[ARCTask]:
        """Load all tasks from directory"""
        task_dir = self.data_dir / self.split
        tasks = []
        
        for json_file in sorted(task_dir.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)
            task = ARCTask(json_file.stem, data)
            tasks.append(task)
        
        return tasks

    @staticmethod
    def _task_difficulty(task: ARCTask) -> int:
        """Score task difficulty by total grid cells + number of unique colors.
        Smaller grids with fewer colors = easier."""
        total_cells = 0
        colors = set()
        for ex in task.train_examples:
            for grid in [ex['input'], ex['output']]:
                h, w = len(grid), len(grid[0]) if grid else 0
                total_cells += h * w
                for row in grid:
                    colors.update(row)
        return total_cells + len(colors) * 10
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task, test_idx = self.samples[idx]
        
        # Apply augmentation (rotation, flipping)
        if self.augment:
            task = self._augment_task(task)
        
        # Format as text
        if self.format_type == 'compact':
            input_text, target_text = task.format_compact(test_idx, include_answer=True)
        else:
            input_text = task.format_as_prompt(test_idx, include_answer=False)
            test_out = task.get_test_output(test_idx)
            target_text = grid_to_text(test_out) if test_out else ""
        
        item = {
            'task_id': task.task_id,
            'test_idx': test_idx,
            'input_text': input_text,
            'target_text': target_text,
            'full_text': input_text + target_text
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            input_tokens_all = self.tokenizer.encode(input_text)
            target_tokens_all = self.tokenizer.encode(target_text)

            # Truncate prompt (from the left) while keeping the full target when possible.
            if len(input_tokens_all) + len(target_tokens_all) > self.max_seq_len:
                max_input = self.max_seq_len - len(target_tokens_all)
                if max_input <= 0:
                    input_tokens = []
                    full_tokens = target_tokens_all[-self.max_seq_len:]
                    input_length = 0
                else:
                    input_tokens = input_tokens_all[-max_input:]
                    full_tokens = input_tokens + target_tokens_all
                    input_length = len(input_tokens)
            else:
                full_tokens = input_tokens_all + target_tokens_all
                input_length = len(input_tokens_all)

            # Next-token prediction setup.
            # We only want to apply loss on the target portion (answer), not the prompt.
            # labels[i] corresponds to predicting full_tokens[i+1] from full_tokens[i].
            prompt_label_len = max(0, min(input_length - 1, len(full_tokens) - 1))

            input_ids = torch.tensor(full_tokens[:-1])
            labels = torch.tensor(full_tokens[1:])
            if prompt_label_len > 0:
                labels[:prompt_label_len] = -100

            # Avoid NaN loss if everything is ignored (can still happen in edge cases).
            if (labels != -100).sum().item() == 0:
                labels = torch.tensor(full_tokens[1:])
                input_length = 0

            item['input_ids'] = input_ids
            item['labels'] = labels
            item['input_length'] = min(input_length, len(full_tokens) - 1)
         
        return item
    
    def _augment_task(self, task: ARCTask) -> ARCTask:
        """Apply random augmentation to task"""
        # For now, just return as-is
        # TODO: implement rotation, flip, color permutation
        return task


def arc_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, Any]:
    """Collate function for ARC DataLoader"""
    if 'input_ids' not in batch[0]:
        return {
            'task_ids': [item['task_id'] for item in batch],
            'input_texts': [item['input_text'] for item in batch],
            'target_texts': [item['target_text'] for item in batch]
        }
    
    # Pad sequences
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        length = len(item['input_ids'])
        padding = max_len - length
        
        input_ids.append(
            torch.cat([item['input_ids'], torch.full((padding,), pad_token_id)])
        )
        labels.append(
            torch.cat([item['labels'], torch.full((padding,), -100)])
        )
        attention_mask.append(
            torch.cat([torch.ones(length), torch.zeros(padding)])
        )
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask).long(),
        'task_ids': [item['task_id'] for item in batch],
        'input_lengths': torch.tensor([item['input_length'] for item in batch])
    }


def create_arc_dataloader(
    data_dir: str,
    split: str = 'training',
    batch_size: int = 4,
    tokenizer = None,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for ARC dataset"""
    dataset = ARCDataset(
        data_dir=data_dir,
        split=split,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: arc_collate_fn(b, pad_token_id=0)
    )


def evaluate_arc_prediction(
    predicted: str,
    target: str
) -> Dict[str, float]:
    """
    Evaluate ARC prediction accuracy.
    
    Returns:
        Dict with 'exact_match', 'grid_accuracy', 'cell_accuracy'
    """
    try:
        pred_grid = tokens_to_grid(predicted.strip().rstrip(']'))
        target_grid = tokens_to_grid(target.strip().rstrip(']'))
    except:
        return {'exact_match': 0.0, 'grid_accuracy': 0.0, 'cell_accuracy': 0.0}
    
    # Exact match
    exact = 1.0 if pred_grid == target_grid else 0.0

    pred_h = len(pred_grid)
    target_h = len(target_grid)
    pred_w = max((len(r) for r in pred_grid), default=0)
    target_w = max((len(r) for r in target_grid), default=0)

    # Cell-level accuracy over the union bounding box; missing cells count as wrong.
    max_h = max(pred_h, target_h)
    max_w = max(pred_w, target_w)
    total_cells = max_h * max_w
    correct_cells = 0
    for i in range(max_h):
        for j in range(max_w):
            pred_cell = pred_grid[i][j] if (i < pred_h and j < len(pred_grid[i])) else None
            target_cell = target_grid[i][j] if (i < target_h and j < len(target_grid[i])) else None
            if pred_cell is not None and target_cell is not None and pred_cell == target_cell:
                correct_cells += 1

    cell_acc = correct_cells / total_cells if total_cells > 0 else 0.0
    same_shape = (pred_h, pred_w) == (target_h, target_w)
    grid_acc = cell_acc if same_shape else 0.0

    return {
        'exact_match': exact,
        'grid_accuracy': grid_acc,
        'cell_accuracy': cell_acc
    }


if __name__ == "__main__":
    print("Testing ARC Dataset...")
    
    # Test with actual data
    data_dir = Path.home() / "ARC_AMD_TRANSFER/data/ARC-AGI/data"
    
    if data_dir.exists():
        print(f"Loading from {data_dir}")
        
        # Load a task
        task_file = list((data_dir / "training").glob("*.json"))[0]
        with open(task_file) as f:
            task_data = json.load(f)
        
        task = ARCTask(task_file.stem, task_data)
        print(f"\nTask: {task.task_id}")
        print(f"Train examples: {task.num_train}")
        print(f"Test examples: {task.num_test}")
        
        # Show formatted prompt
        print("\n" + "="*60)
        print("VERBOSE FORMAT:")
        print("="*60)
        print(task.format_as_prompt(include_answer=True)[:1000])
        
        print("\n" + "="*60)
        print("COMPACT FORMAT:")
        print("="*60)
        inp, tgt = task.format_compact(include_answer=True)
        print(f"Input: {inp[:200]}...")
        print(f"Target: {tgt}")
        
        # Test dataset
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            
            dataset = ARCDataset(
                data_dir=str(data_dir),
                split='training',
                tokenizer=enc,
                max_seq_len=512
            )
            
            print(f"\nDataset size: {len(dataset)}")
            
            item = dataset[0]
            print(f"Sample task: {item['task_id']}")
            print(f"Input IDs shape: {item['input_ids'].shape}")
            print(f"Labels shape: {item['labels'].shape}")
            
            # Test dataloader
            loader = create_arc_dataloader(
                data_dir=str(data_dir),
                split='training',
                batch_size=4,
                tokenizer=enc
            )
            
            batch = next(iter(loader))
            print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
            print(f"Batch task_ids: {batch['task_ids']}")
            
        except ImportError:
            print("tiktoken not available, skipping tokenization test")
        
        print("\nARC Dataset tests passed!")
    else:
        print(f"Data directory not found: {data_dir}")

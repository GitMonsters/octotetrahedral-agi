"""
Synthetic Task Generator for OctoTetrahedral AGI Training
Controllable complexity for debugging and initial training

Task types:
1. Arithmetic - addition, subtraction, multiplication
2. Pattern - sequence completion, transformations
3. Logic - boolean operations, simple reasoning

Key design:
- All tasks are tokenized for language model training
- Difficulty can be controlled via parameters
- Ground truth is always available for supervised learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Generator, Any
import random
import re
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    ARITHMETIC = "arithmetic"
    PATTERN = "pattern"
    LOGIC = "logic"
    COPY = "copy"
    REVERSE = "reverse"


@dataclass
class Task:
    """Single training task"""
    input_text: str
    target_text: str
    task_type: TaskType
    difficulty: int  # 1-5


class ArithmeticGenerator:
    """Generate arithmetic tasks"""
    
    OPERATIONS = {
        'add': ('+', lambda a, b: a + b),
        'sub': ('-', lambda a, b: a - b),
        'mul': ('*', lambda a, b: a * b),
    }
    
    def __init__(self, max_value: int = 100, seed: Optional[int] = None):
        self.max_value = max_value
        self.rng = random.Random(seed)
    
    def generate(self, difficulty: int = 1) -> Task:
        """
        Generate arithmetic task.
        
        Difficulty:
        1: single digit addition
        2: two digit addition/subtraction
        3: two digit with multiplication
        4: three numbers
        5: nested operations
        """
        if difficulty == 1:
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            result = a + b
            input_text = f"Calculate: {a} + {b} = "
            target_text = str(result)
            
        elif difficulty == 2:
            a = self.rng.randint(10, 99)
            b = self.rng.randint(10, 99)
            op_name = self.rng.choice(['add', 'sub'])
            symbol, func = self.OPERATIONS[op_name]
            result = func(a, b)
            input_text = f"Calculate: {a} {symbol} {b} = "
            target_text = str(result)
            
        elif difficulty == 3:
            a = self.rng.randint(10, 99)
            b = self.rng.randint(2, 12)
            op_name = self.rng.choice(['add', 'sub', 'mul'])
            symbol, func = self.OPERATIONS[op_name]
            result = func(a, b)
            input_text = f"Calculate: {a} {symbol} {b} = "
            target_text = str(result)
            
        elif difficulty == 4:
            a = self.rng.randint(10, 50)
            b = self.rng.randint(10, 50)
            c = self.rng.randint(1, 20)
            result = a + b + c
            input_text = f"Calculate: {a} + {b} + {c} = "
            target_text = str(result)
            
        else:  # difficulty 5
            a = self.rng.randint(10, 30)
            b = self.rng.randint(2, 5)
            c = self.rng.randint(10, 30)
            # (a * b) + c
            result = (a * b) + c
            input_text = f"Calculate: ({a} * {b}) + {c} = "
            target_text = str(result)
        
        return Task(
            input_text=input_text,
            target_text=target_text,
            task_type=TaskType.ARITHMETIC,
            difficulty=difficulty
        )


class PatternGenerator:
    """Generate pattern completion tasks"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate(self, difficulty: int = 1) -> Task:
        """
        Generate pattern task.
        
        Difficulty:
        1: counting (+1)
        2: skip counting (+2, +3)
        3: arithmetic sequences
        4: geometric patterns
        5: fibonacci-like
        """
        if difficulty == 1:
            start = self.rng.randint(1, 10)
            seq = [start + i for i in range(5)]
            input_text = f"Continue the pattern: {' '.join(map(str, seq[:-1]))} "
            target_text = str(seq[-1])
            
        elif difficulty == 2:
            start = self.rng.randint(1, 10)
            step = self.rng.choice([2, 3, 5])
            seq = [start + i * step for i in range(5)]
            input_text = f"Continue the pattern: {' '.join(map(str, seq[:-1]))} "
            target_text = str(seq[-1])
            
        elif difficulty == 3:
            start = self.rng.randint(1, 10)
            step = self.rng.randint(2, 5)
            seq = [start + i * step for i in range(6)]
            input_text = f"Continue the pattern: {' '.join(map(str, seq[:-1]))} "
            target_text = str(seq[-1])
            
        elif difficulty == 4:
            start = self.rng.choice([2, 3])
            ratio = 2
            seq = [start * (ratio ** i) for i in range(5)]
            input_text = f"Continue the pattern: {' '.join(map(str, seq[:-1]))} "
            target_text = str(seq[-1])
            
        else:  # difficulty 5
            # Fibonacci-like
            a, b = self.rng.randint(1, 5), self.rng.randint(1, 5)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            input_text = f"Continue the pattern: {' '.join(map(str, seq[:-1]))} "
            target_text = str(seq[-1])
        
        return Task(
            input_text=input_text,
            target_text=target_text,
            task_type=TaskType.PATTERN,
            difficulty=difficulty
        )


class LogicGenerator:
    """Generate logic tasks"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate(self, difficulty: int = 1) -> Task:
        """
        Generate logic task.
        
        Difficulty:
        1: simple true/false
        2: AND/OR
        3: NOT with AND/OR
        4: comparison chains
        5: multi-step reasoning
        """
        if difficulty == 1:
            a = self.rng.randint(1, 10)
            b = self.rng.randint(1, 10)
            op = self.rng.choice(['>', '<', '='])
            if op == '>':
                result = 'true' if a > b else 'false'
            elif op == '<':
                result = 'true' if a < b else 'false'
            else:
                result = 'true' if a == b else 'false'
            input_text = f"Is {a} {op} {b}? Answer true or false: "
            target_text = result
            
        elif difficulty == 2:
            a = self.rng.choice([True, False])
            b = self.rng.choice([True, False])
            op = self.rng.choice(['AND', 'OR'])
            if op == 'AND':
                result = 'true' if (a and b) else 'false'
            else:
                result = 'true' if (a or b) else 'false'
            a_str = 'true' if a else 'false'
            b_str = 'true' if b else 'false'
            input_text = f"What is {a_str} {op} {b_str}? "
            target_text = result
            
        elif difficulty == 3:
            a = self.rng.choice([True, False])
            b = self.rng.choice([True, False])
            result = not (a and b)
            a_str = 'true' if a else 'false'
            b_str = 'true' if b else 'false'
            result_str = 'true' if result else 'false'
            input_text = f"What is NOT ({a_str} AND {b_str})? "
            target_text = result_str
            
        elif difficulty == 4:
            a = self.rng.randint(1, 10)
            b = self.rng.randint(1, 10)
            c = self.rng.randint(1, 10)
            # Is a < b < c?
            result = 'true' if (a < b < c) else 'false'
            input_text = f"Is {a} < {b} < {c}? Answer true or false: "
            target_text = result
            
        else:  # difficulty 5
            # If-then reasoning
            x = self.rng.randint(1, 20)
            threshold = 10
            if x > threshold:
                condition = "greater"
                result = "big"
            else:
                condition = "less"
                result = "small"
            input_text = f"If a number is greater than {threshold}, it is 'big'. If less or equal, it is 'small'. What is {x}? "
            target_text = result
        
        return Task(
            input_text=input_text,
            target_text=target_text,
            task_type=TaskType.LOGIC,
            difficulty=difficulty
        )


class CopyGenerator:
    """Generate copy tasks (for testing basic memorization)"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    
    def generate(self, difficulty: int = 1) -> Task:
        """Generate copy task with varying length"""
        length = difficulty * 3 + 2  # 5, 8, 11, 14, 17
        text = ''.join(self.rng.choices(self.chars, k=length))
        input_text = f"Copy this text: {text} -> "
        target_text = text
        
        return Task(
            input_text=input_text,
            target_text=target_text,
            task_type=TaskType.COPY,
            difficulty=difficulty
        )


class ReverseGenerator:
    """Generate reverse tasks"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    
    def generate(self, difficulty: int = 1) -> Task:
        """Generate reverse task with varying length"""
        length = difficulty * 2 + 3  # 5, 7, 9, 11, 13
        text = ''.join(self.rng.choices(self.chars, k=length))
        input_text = f"Reverse this: {text} -> "
        target_text = text[::-1]
        
        return Task(
            input_text=input_text,
            target_text=target_text,
            task_type=TaskType.REVERSE,
            difficulty=difficulty
        )


class SyntheticTaskDataset(Dataset):
    """PyTorch Dataset for synthetic tasks"""
    
    def __init__(
        self,
        num_samples: int = 10000,
        task_types: Optional[List[TaskType]] = None,
        difficulty_range: Tuple[int, int] = (1, 3),
        seed: int = 42,
        tokenizer=None
    ):
        self.num_samples = num_samples
        self.task_types = task_types or [
            TaskType.ARITHMETIC,
            TaskType.PATTERN,
            TaskType.LOGIC,
            TaskType.COPY,
            TaskType.REVERSE
        ]
        self.difficulty_range = difficulty_range
        self.seed = seed
        self.tokenizer = tokenizer
        
        # Initialize generators
        self.generators = {
            TaskType.ARITHMETIC: ArithmeticGenerator(seed=seed),
            TaskType.PATTERN: PatternGenerator(seed=seed),
            TaskType.LOGIC: LogicGenerator(seed=seed),
            TaskType.COPY: CopyGenerator(seed=seed),
            TaskType.REVERSE: ReverseGenerator(seed=seed)
        }
        
        self.rng = random.Random(seed)
        
        # Pre-generate tasks
        self.tasks = self._generate_tasks()
    
    def _generate_tasks(self) -> List[Task]:
        """Generate all tasks"""
        tasks = []
        for _ in range(self.num_samples):
            task_type = self.rng.choice(self.task_types)
            difficulty = self.rng.randint(*self.difficulty_range)
            
            generator = self.generators[task_type]
            task = generator.generate(difficulty=difficulty)
            tasks.append(task)
        
        return tasks
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        
        item = {
            'input_text': task.input_text,
            'target_text': task.target_text,
            'task_type': task.task_type.value,
            'difficulty': task.difficulty
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            full_text = task.input_text + task.target_text
            tokens = self.tokenizer.encode(full_text)
            input_tokens = self.tokenizer.encode(task.input_text)
            
            item['input_ids'] = torch.tensor(tokens[:-1])
            item['labels'] = torch.tensor(tokens[1:])
            item['input_length'] = len(input_tokens)
        
        return item


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    # Find max length
    if 'input_ids' in batch[0]:
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
                torch.cat([item['labels'], torch.full((padding,), -100)])  # -100 = ignore
            )
            attention_mask.append(
                torch.cat([torch.ones(length), torch.zeros(padding)])
            )
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_mask).long(),
            'task_types': [item['task_type'] for item in batch],
            'difficulties': torch.tensor([item['difficulty'] for item in batch])
        }
    else:
        return {
            'input_texts': [item['input_text'] for item in batch],
            'target_texts': [item['target_text'] for item in batch],
            'task_types': [item['task_type'] for item in batch],
            'difficulties': torch.tensor([item['difficulty'] for item in batch])
        }


def create_dataloader(
    num_samples: int = 10000,
    batch_size: int = 8,
    task_types: Optional[List[TaskType]] = None,
    difficulty_range: Tuple[int, int] = (1, 3),
    seed: int = 42,
    tokenizer=None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for synthetic tasks"""
    dataset = SyntheticTaskDataset(
        num_samples=num_samples,
        task_types=task_types,
        difficulty_range=difficulty_range,
        seed=seed,
        tokenizer=tokenizer
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=0)
    )


if __name__ == "__main__":
    print("Testing Synthetic Task Generators...")
    
    # Test each generator
    generators = {
        'Arithmetic': ArithmeticGenerator(seed=42),
        'Pattern': PatternGenerator(seed=42),
        'Logic': LogicGenerator(seed=42),
        'Copy': CopyGenerator(seed=42),
        'Reverse': ReverseGenerator(seed=42)
    }
    
    for name, gen in generators.items():
        print(f"\n{name} tasks:")
        for diff in range(1, 4):
            task = gen.generate(difficulty=diff)
            print(f"  Difficulty {diff}: {task.input_text}{task.target_text}")
    
    # Test dataset
    print("\nTesting SyntheticTaskDataset...")
    dataset = SyntheticTaskDataset(num_samples=100, seed=42)
    print(f"Dataset size: {len(dataset)}")
    
    # Sample items
    for i in [0, 50, 99]:
        item = dataset[i]
        print(f"  Item {i}: [{item['task_type']}] {item['input_text']}{item['target_text']}")
    
    # Test dataloader (without tokenizer)
    print("\nTesting DataLoader...")
    loader = create_dataloader(num_samples=100, batch_size=4, seed=42)
    batch = next(iter(loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Difficulties: {batch['difficulties']}")
    print(f"Task types: {batch['task_types'][:4]}")
    
    # Test with tiktoken if available
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        print("\nTesting with tiktoken tokenizer...")
        dataset_tok = SyntheticTaskDataset(
            num_samples=100,
            seed=42,
            tokenizer=enc
        )
        
        item = dataset_tok[0]
        print(f"Input IDs shape: {item['input_ids'].shape}")
        print(f"Labels shape: {item['labels'].shape}")
        print(f"Input length: {item['input_length']}")
        
        # Decode to verify
        decoded = enc.decode(item['input_ids'].tolist())
        print(f"Decoded: {decoded}")
        
    except ImportError:
        print("\nSkipping tiktoken test (not installed)")
    
    print("\nAll synthetic task tests passed!")

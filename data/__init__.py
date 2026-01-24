"""
OctoTetrahedral AGI - Data Module
Synthetic task generation for training
"""

from .synthetic_tasks import (
    TaskType,
    Task,
    ArithmeticGenerator,
    PatternGenerator,
    LogicGenerator,
    CopyGenerator,
    ReverseGenerator,
    SyntheticTaskDataset,
    create_dataloader,
    collate_fn
)

__all__ = [
    'TaskType', 'Task',
    'ArithmeticGenerator', 'PatternGenerator', 'LogicGenerator',
    'CopyGenerator', 'ReverseGenerator',
    'SyntheticTaskDataset', 'create_dataloader', 'collate_fn'
]

"""
SIMULA Integration for ARC Training
Augments ARCTrainer with synthetic data generation and injection
"""

import logging
from copy import copy
from typing import Dict, List, Optional, Any, Tuple
import torch
from pathlib import Path
import random

from core.simula_compound_bridge import SimulaCompoundBridge
from data.arc_dataset import ARCTask, grid_to_text, ARCDataset

logger = logging.getLogger(__name__)


class SyntheticDataAugmentation:
    """
    Manages SIMULA synthetic data generation and injection into training.
    
    Features:
    - Generate synthetic ARC examples using SIMULA taxonomy
    - Track generation statistics (coverage, quality)
    - Inject into training loop at configurable ratio
    - Record learning experiences to compound engine
    """
    
    def __init__(
        self,
        enabled: bool = True,
        simula_ratio: float = 0.2,
        simula_complexity: int = 3,
        simula_examples_per_epoch: int = 50,
        use_compound_learning: bool = False
    ):
        """
        Initialize synthetic data augmentation.
        
        Args:
            enabled: Whether to generate synthetic data
            simula_ratio: Fraction of training data to be synthetic (0.0-1.0)
            simula_complexity: Complexity level for generation (1-5)
            simula_examples_per_epoch: Number of synthetic examples to generate per epoch
            use_compound_learning: Whether to record to compound learning engine
        """
        self.enabled = enabled
        self.simula_ratio = max(0.0, min(1.0, simula_ratio))
        self.simula_complexity = max(1, min(5, simula_complexity))
        self.simula_examples_per_epoch = simula_examples_per_epoch
        self.use_compound_learning = use_compound_learning
        
        # Initialize SIMULA bridge
        self.simula = SimulaCompoundBridge() if enabled else None
        
        # Statistics
        self.stats = {
            'synthetic_examples_generated': 0,
            'synthetic_examples_used': 0,
            'total_real_examples': 0,
            'average_coverage': 0.0,
            'average_quality': 0.0,
        }
        
        # Synthetic example cache
        self.synthetic_cache: List[ARCTask] = []
        
        if self.enabled:
            logger.info(f"SIMULA augmentation enabled: ratio={self.simula_ratio}, "
                       f"complexity={self.simula_complexity}")
    
    def initialize_taxonomy(self):
        """Build domain taxonomy for synthetic generation"""
        if not self.enabled or self.simula is None:
            return
        
        logger.info("Initializing ARC domain taxonomy...")
        self.simula.create_arc_taxonomy()
        
        # Log taxonomy info
        coverage_report = self.simula.get_taxonomy_coverage_report("arc")
        logger.info(f"Taxonomy: {coverage_report.get('total_nodes', 0)} nodes, "
                   f"coverage: {coverage_report.get('coverage_percentage', 0) / 100:.1%}")
    
    def generate_synthetic_batch(
        self,
        num_examples: int = 10,
        domain: str = "arc"
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of synthetic ARC examples using SIMULA.
        
        Returns:
            List of generated examples with metadata
        """
        if not self.enabled or self.simula is None:
            return []
        
        logger.info(f"Generating {num_examples} synthetic examples "
                   f"(complexity={self.simula_complexity})...")
        
        examples = []
        for i in range(num_examples):
            try:
                # Generate with learning recording
                result = self.simula.generate_with_learning(
                    domain=domain,
                    num_examples=1,
                    complexity_range=(self.simula_complexity, self.simula_complexity),
                )
                
                if result and len(result) > 0:
                    example = result[0]
                    examples.append({
                        'task_id': f"synthetic_{self.stats['synthetic_examples_generated'] + i}",
                        'train_data': example.input_data,
                        'test_data': example.output_data,
                        'complexity': self.simula_complexity,
                        'quality_score': example.quality_score,
                        'source': 'simula'
                    })
            except Exception as e:
                logger.warning(f"Failed to generate synthetic example {i}: {e}")
                continue
        
        self.stats['synthetic_examples_generated'] += len(examples)
        
        logger.info(f"Generated {len(examples)} synthetic examples")
        return examples
    
    def inject_into_dataloader(
        self,
        base_dataset: ARCDataset,
        synthetic_examples: List[Dict[str, Any]]
    ) -> ARCDataset:
        """
        Create an augmented dataset with synthetic examples injected.
        
        Args:
            base_dataset: Original ARCDataset
            synthetic_examples: List of generated synthetic examples
        
        Returns:
            New ARCDataset with synthetic examples added
        """
        if not synthetic_examples:
            return base_dataset
        
        logger.info(f"Injecting {len(synthetic_examples)} synthetic examples into dataset")
        
        # Create copies of tasks list
        augmented_tasks = list(base_dataset.tasks)
        
        # Convert synthetic examples to ARCTask format
        for syn_ex in synthetic_examples:
            task_data = {
                'train': syn_ex['train_data'],
                'test': syn_ex['test_data']
            }
            
            # Create ARCTask
            task = ARCTask(syn_ex['task_id'], task_data)
            augmented_tasks.append(task)
            
            self.stats['synthetic_examples_used'] += 1
        
        # ARCDataset loads from disk in __init__, so clone the existing dataset
        # and replace its in-memory task/sample lists.
        augmented_dataset = copy(base_dataset)
        augmented_dataset.tasks = augmented_tasks
        augmented_dataset.samples = []
        for task in augmented_tasks:
            for test_idx in range(task.num_test):
                augmented_dataset.samples.append((task, test_idx))
        
        logger.info(f"Dataset augmented: {len(base_dataset.tasks)} → "
                    f"{len(augmented_dataset.tasks)} examples")
        
        self.stats['total_real_examples'] = len(base_dataset.tasks)
        return augmented_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            **self.stats,
            'enabled': self.enabled,
            'ratio': self.simula_ratio,
            'complexity': self.simula_complexity,
        }
    
    def log_statistics(self):
        """Log augmentation statistics"""
        if not self.enabled:
            return
        
        stats = self.get_statistics()
        logger.info("SIMULA Augmentation Statistics:")
        logger.info(f"  Synthetic examples generated: {stats['synthetic_examples_generated']}")
        logger.info(f"  Synthetic examples used: {stats['synthetic_examples_used']}")
        logger.info(f"  Real examples: {stats['total_real_examples']}")
        if stats['total_real_examples'] > 0:
            ratio = stats['synthetic_examples_used'] / (
                stats['total_real_examples'] + stats['synthetic_examples_used']
            )
            logger.info(f"  Synthetic ratio: {ratio:.1%}")

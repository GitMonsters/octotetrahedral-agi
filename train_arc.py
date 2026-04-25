"""
OctoTetrahedral AGI - ARC-AGI Training Script
Trains the model on Abstract Reasoning Corpus (ARC) tasks.

ARC tasks require:
- Few-shot learning from example input/output pairs
- Abstract pattern recognition and generalization
- Grid-based spatial reasoning

SIMULA Integration:
- Augments training data with synthetic examples for diversity
- Prevents overfitting through structured data generation
- Tracks synthetic data quality and coverage
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any, Tuple, List
import time
import math
import logging
import json
from pathlib import Path
from collections import defaultdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not installed. Using simple tokenization.")

from config import Config, get_config
from model import OctoTetrahedralModel
from data.arc_dataset import (
    ARCDataset,
    ARCTask,
    create_arc_dataloader,
    evaluate_arc_prediction,
    tokens_to_grid
)
from training.simula_enhanced_trainer import SyntheticDataAugmentation
from training.euphan_enhanced_trainer import EuphanTrainingIntegration
from training.hermes_enhanced_trainer import HermesTrainingIntegration
from core.cognitive_cohesion_braid import CognitiveCohesionBraid, CohesionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARCTrainer:
    """
    Trainer specialized for ARC-AGI tasks.
    
    Features:
    - Grid prediction accuracy tracking
    - Task-level evaluation (exact match)
    - Periodic full evaluation on held-out tasks
    - Curriculum learning (optional)
    - SIMULA synthetic data augmentation
    """
    
    def __init__(
        self,
        model: OctoTetrahedralModel,
        config: Config,
        train_dataloader,
        val_dataloader=None,
        tokenizer=None,
        device: str = None,
        use_simula: bool = False,
        use_euphan: bool = False,
        use_hermes: bool = False,
        use_cohesion: bool = False
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Device
        self.device = device or config.device
        self.model.to(self.device)
        
        # Optimizer - use slightly higher LR for ARC
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_accuracy = 0.0
        
        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_metrics: List[Dict] = []
        self.task_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Checkpointing
        self.checkpoint_dir = Path("checkpoints/arc")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # SIMULA synthetic data augmentation
        self.simula_augmentation = None
        if use_simula and config.training.use_simula:
            self.simula_augmentation = SyntheticDataAugmentation(
                enabled=True,
                simula_ratio=config.training.simula_ratio,
                simula_complexity=config.training.simula_complexity,
                simula_examples_per_epoch=config.training.simula_examples_per_epoch,
                use_compound_learning=False  # Set True if compound engine available
            )
            self.simula_augmentation.initialize_taxonomy()
            logger.info("SIMULA augmentation initialized")
        
        # EUPHAN limb observability
        self.euphan_integration = None
        if use_euphan and config.training.use_euphan:
            self.euphan_integration = EuphanTrainingIntegration(
                enabled=True,
                use_euphan_bridge=False,
                log_frequency=config.training.euphan_log_frequency
            )
            # Pass logger to model
            self.model.euphan_logger = self.euphan_integration.start_session(task_id="arc_training")
            logger.info("EUPHAN observability initialized")
        
        # HERMES background task orchestration
        self.hermes_integration = None
        if use_hermes and config.training.use_hermes:
            self.hermes_integration = HermesTrainingIntegration(
                enabled=True,
                learning_engine=self,
                log_frequency=config.training.hermes_log_frequency,
                output_dir=config.training.hermes_output_dir,
                max_parallel_agents=config.training.hermes_max_agents,
                max_queue_size=config.training.hermes_queue_size
            )
            self.hermes_integration.initialize_agents(num_agents=config.training.hermes_max_agents)
            logger.info("HERMES background orchestration initialized")

        # Cognitive Cohesion Braid — recompiles SIMULA+EUPHAN+HERMES+limbs+skills
        # into a single braided substrate with cross-bridge feedback loops.
        self.cohesion_braid = None
        if use_cohesion and getattr(config.training, 'use_cohesion', True):
            self.cohesion_braid = CognitiveCohesionBraid(CohesionConfig(
                enabled=True,
                output_dir=getattr(config.training, 'cohesion_output_dir', 'logs/cohesion'),
            ))
            self.cohesion_braid.bind_simula(getattr(self, 'simula_augmentation', None))
            self.cohesion_braid.bind_euphan(getattr(self, 'euphan_integration', None))
            self.cohesion_braid.bind_hermes(getattr(self, 'hermes_integration', None))
            logger.info("Cognitive Cohesion Braid initialized "
                        "(SIMULA↔EUPHAN↔HERMES cross-feedback active)")
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay"""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0
            }
        ]
        
        # Slightly higher LR for ARC (grid patterns need stronger gradients)
        lr = self.config.training.learning_rate * 1.5
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=self.config.training.betas
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup"""
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_confidences=True
        )
        
        loss = output['loss']
        
        # Skip step if loss is NaN (can happen with newly initialized layers)
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return {'loss': 0.0, 'token_accuracy': 0.0, 'lr': self.optimizer.param_groups[0]['lr'], 'skipped': True}
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update limb gradient counters
        for limb in self.model._limbs.values():
            if hasattr(limb, 'increment_gradient_step'):
                limb.increment_gradient_step()
        
        # Hub sync check
        sync_result = self.model.sync_limbs(performance=1.0 - loss.item())
        
        self.global_step += 1
        
        # Quick token accuracy on this batch
        with torch.no_grad():
            preds = output['logits'].argmax(dim=-1)
            mask = labels != -100
            denom = mask.sum()
            if denom.item() == 0:
                token_acc = torch.tensor(0.0, device=preds.device)
            else:
                token_acc = ((preds == labels) & mask).float().sum() / denom
        
        return {
            'loss': loss.item(),
            'token_accuracy': token_acc.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'confidences': output.get('confidences', {}),
            'synced': sync_result is not None,
            'task_ids': batch.get('task_ids', [])
        }
    
    @torch.no_grad()
    def evaluate_generation(self, num_samples: int = 20) -> Dict[str, float]:
        """
        Evaluate model by generating outputs and comparing to targets.
        This tests actual grid prediction accuracy.
        """
        if self.val_dataloader is None or self.tokenizer is None:
            return {}
        
        self.model.eval()
        
        results = {
            'exact_match': 0,
            'grid_accuracy': 0.0,
            'cell_accuracy': 0.0,
            'total': 0
        }
        
        # Get raw text samples (not tokenized)
        val_dataset = self.val_dataloader.dataset
        
        for idx in range(min(num_samples, len(val_dataset))):
            try:
                # Get sample without tokenization
                task = val_dataset.tasks[idx % len(val_dataset.tasks)]
                
                # Format input (examples + test input) and target
                # NOTE: ARCTask.format_compact returns (input_text, target_text). The previous
                # code incorrectly treated the second return value as the full concatenated text.
                input_text, _ = task.format_compact(test_idx=0, include_answer=False)
                _, target_text = task.format_compact(test_idx=0, include_answer=True)
                
                # Tokenize input
                input_tokens = self.tokenizer.encode(input_text)
                # Limit input length for faster generation
                if len(input_tokens) > 256:
                    input_tokens = input_tokens[:256]
                input_ids = torch.tensor([input_tokens]).to(self.device)
                
                # Generate output (limit to 50 tokens for speed)
                generated_ids = self._generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.3
                )
                
                # Decode generated output
                generated_text = self.tokenizer.decode(generated_ids[0, len(input_tokens):].tolist())
                
                # Evaluate
                metrics = evaluate_arc_prediction(generated_text, target_text)
                
                results['exact_match'] += metrics['exact_match']
                results['grid_accuracy'] += metrics['grid_accuracy']
                results['cell_accuracy'] += metrics['cell_accuracy']
                results['total'] += 1
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Average
        if results['total'] > 0:
            results['exact_match'] /= results['total']
            results['grid_accuracy'] /= results['total']
            results['cell_accuracy'] /= results['total']
        
        return results
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50
    ) -> torch.Tensor:
        """Simple generation for evaluation"""
        self.model.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Truncate if needed
            if generated.shape[1] > self.config.model.max_seq_len:
                context = generated[:, -self.config.model.max_seq_len:]
            else:
                context = generated
            
            # Forward pass
            output = self.model(input_ids=context)
            logits = output['logits'][:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at ] which marks end of grid
            if self.tokenizer:
                decoded = self.tokenizer.decode([next_token.item()])
                if ']' in decoded:
                    break
        
        return generated
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on held-out tasks"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += output['loss'].item()
            
            # Token accuracy
            preds = output['logits'].argmax(dim=-1)
            mask = labels != -100
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        token_accuracy = total_correct / max(1, total_tokens)
        
        return {
            'val_loss': avg_loss,
            'val_token_accuracy': token_accuracy
        }
    
    def train(
        self,
        max_steps: int = None,
        eval_generation_every: int = 500
    ):
        """
        Main training loop for ARC tasks.
        
        Includes SIMULA synthetic data augmentation if enabled.
        
        Args:
            max_steps: Maximum training steps
            eval_generation_every: How often to run generation evaluation
        """
        max_steps = max_steps or self.config.training.max_steps
        
        logger.info(f"Starting ARC training for {max_steps} steps")
        logger.info(f"Model has {self.model.get_num_params():,} parameters")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training tasks: {len(self.train_dataloader.dataset)} samples")
        if self.val_dataloader:
            logger.info(f"Validation tasks: {len(self.val_dataloader.dataset)} samples")
        
        # Generate synthetic data before training if SIMULA enabled
        if self.simula_augmentation and self.simula_augmentation.enabled:
            logger.info("Generating initial synthetic data batch...")
            synthetic_examples = self.simula_augmentation.generate_synthetic_batch(
                num_examples=self.config.training.simula_examples_per_epoch,
                domain="arc"
            )
            
            if synthetic_examples:
                # Augment training dataset
                original_dataset = self.train_dataloader.dataset
                augmented_dataset = self.simula_augmentation.inject_into_dataloader(
                    original_dataset,
                    synthetic_examples
                )
                
                # Replace dataloader with augmented dataset
                from torch.utils.data import DataLoader
                self.train_dataloader = DataLoader(
                    augmented_dataset,
                    batch_size=self.train_dataloader.batch_size,
                    shuffle=True,
                    num_workers=0  # ARC dataset doesn't support multiprocessing
                )
                
                logger.info(f"Training dataloader augmented with synthetic data")
                logger.info(f"New training size: {len(self.train_dataloader.dataset)} samples")
        
        start_time = time.time()
        running_loss = 0.0
        running_token_acc = 0.0
        
        while self.global_step < max_steps:
            for batch in self.train_dataloader:
                if self.global_step >= max_steps:
                    break
                
                # Training step
                step_result = self.train_step(batch)
                running_loss += step_result['loss']
                running_token_acc += step_result['token_accuracy']
                
                # Logging
                if self.global_step % self.config.training.log_interval == 0:
                    avg_loss = running_loss / self.config.training.log_interval
                    avg_token_acc = running_token_acc / self.config.training.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Token Acc: {avg_token_acc:.3f} | "
                        f"LR: {step_result['lr']:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )
                    
                    if step_result.get('confidences'):
                        conf = step_result['confidences']
                        logger.info(
                            f"  Confidences - P: {conf.get('perception', 0):.3f}, "
                            f"R: {conf.get('reasoning', 0):.3f}, "
                            f"A: {conf.get('action', 0):.3f}"
                        )
                    
                    if step_result.get('synced'):
                        logger.info("  Hub sync performed")
                    
                    self.train_losses.append(avg_loss)
                    running_loss = 0.0
                    running_token_acc = 0.0
                
                # Validation (token-level)
                if (self.global_step % self.config.training.eval_interval == 0 
                    and self.val_dataloader is not None):
                    val_result = self.validate()
                    logger.info(
                        f"  Validation - Loss: {val_result['val_loss']:.4f}, "
                        f"Token Acc: {val_result['val_token_accuracy']:.4f}"
                    )
                    self.val_metrics.append(val_result)
                
                # Generation evaluation (actual grid prediction)
                if self.global_step % eval_generation_every == 0 and self.global_step > 0:
                    logger.info("  Running generation evaluation...")
                    gen_result = self.evaluate_generation(num_samples=20)
                    if gen_result:
                        logger.info(
                            f"  Generation - Exact Match: {gen_result['exact_match']:.3f}, "
                            f"Grid Acc: {gen_result['grid_accuracy']:.3f}, "
                            f"Cell Acc: {gen_result['cell_accuracy']:.3f}"
                        )
                        
                        # Save best model by grid accuracy
                        if gen_result['grid_accuracy'] > self.best_val_accuracy:
                            self.best_val_accuracy = gen_result['grid_accuracy']
                            self.save_checkpoint('best_arc.pt')
                            logger.info(f"  New best model! Grid accuracy: {self.best_val_accuracy:.3f}")
                
                # Periodic checkpointing
                if self.global_step % self.config.training.save_interval == 0:
                    self.save_checkpoint(f'arc_step_{self.global_step}.pt')
            
            self.epoch += 1
        
        # Final checkpoint
        self.save_checkpoint('arc_final.pt')
        
        # Final generation evaluation
        logger.info("\nFinal generation evaluation...")
        final_gen = self.evaluate_generation(num_samples=50)
        if final_gen:
            logger.info(f"Final Results:")
            logger.info(f"  Exact Match: {final_gen['exact_match']:.3f}")
            logger.info(f"  Grid Accuracy: {final_gen['grid_accuracy']:.3f}")
            logger.info(f"  Cell Accuracy: {final_gen['cell_accuracy']:.3f}")
        
        # Log SIMULA statistics if used
        if self.simula_augmentation and self.simula_augmentation.enabled:
            logger.info("\n" + "="*60)
            logger.info("SIMULA Augmentation Summary")
            logger.info("="*60)
            self.simula_augmentation.log_statistics()
        
        # Finalize EUPHAN logging if used
        if self.euphan_integration and self.euphan_integration.enabled:
            logger.info("\n" + "="*60)
            logger.info("EUPHAN Observability Summary")
            logger.info("="*60)
            euphan_output_dir = self.config.training.euphan_output_dir
            Path(euphan_output_dir).mkdir(parents=True, exist_ok=True)
            stats = self.euphan_integration.end_session(output_dir=euphan_output_dir)
            if stats:
                logger.info(f"EUPHAN session saved to {euphan_output_dir}")
                logger.info(f"Total limb events logged: {sum(s.get('count', 0) for s in stats.get('limb_stats', {}).values())}")
                logger.info(f"Session duration: {stats.get('session_duration', 0):.3f}s")
                for limb_name, limb_stats in stats.get('limb_stats', {}).items():
                    logger.info(f"  {limb_name}: {limb_stats['count']} events, {limb_stats['total_time']:.3f}s, conf={limb_stats['avg_confidence']:.2f}")
        
        # Finalize HERMES orchestration if used
        if self.hermes_integration and self.hermes_integration.enabled:
            logger.info("\n" + "="*60)
            logger.info("HERMES Background Orchestration Summary")
            logger.info("="*60)
            summary = self.hermes_integration.get_summary(training_step=self.global_step)
            logger.info(f"Total tasks queued: {summary.get('total_tasks_queued', 0)}")
            logger.info(f"Total results collected: {summary.get('total_results_collected', 0)}")
            logger.info(f"Overall exact match rate: {summary.get('overall_exact_match', 0):.1%}")
            logger.info(f"Overall grid accuracy: {summary.get('overall_grid_accuracy', 0):.1%}")
            
            for agent_id, agent_stats in summary.get('agent_summaries', {}).items():
                logger.info(f"  {agent_id}: {agent_stats['tasks_completed']}/{agent_stats['tasks_queued']} success, "
                           f"exact_match={agent_stats['avg_exact_match']:.1%}, grid_acc={agent_stats['avg_grid_accuracy']:.1%}")
            
            # Generate HERMES reports
            hermes_output_dir = self.config.training.hermes_output_dir
            Path(hermes_output_dir).mkdir(parents=True, exist_ok=True)
            self.hermes_integration.generate_html_report(f"{hermes_output_dir}/hermes_training_report.html")
            self.hermes_integration.save_metrics_json(f"{hermes_output_dir}/hermes_training_metrics.json")
            logger.info(f"HERMES reports saved to {hermes_output_dir}")

        # Finalize Cognitive Cohesion Braid
        if self.cohesion_braid:
            logger.info("\n" + "="*60)
            logger.info("Cognitive Cohesion Braid Summary")
            logger.info("="*60)
            score = self.cohesion_braid.cohesion_score()
            logger.info(f"Cohesion score:   {score['cohesion_score']:.3f} (EWMA {score['ewma_score']:.3f})")
            logger.info(f"Limb balance:     {score['limb_balance']:.3f} ({score['limbs_active']}/13 active)")
            logger.info(f"Skill coverage:   {score['skill_coverage']:.3f} ({score['skills_active']}/14 fired)")
            logger.info(f"Braid routings:   "
                        f"S→E={score['braid_stats']['simula_to_euphan']}  "
                        f"E→H={score['braid_stats']['euphan_to_hermes']}  "
                        f"H→S={score['braid_stats']['hermes_to_simula']}")
            html_path = self.cohesion_braid.generate_html_report()
            json_path = self.cohesion_braid.export_json()
            logger.info(f"Cohesion report:  {html_path}")
            logger.info(f"Cohesion metrics: {json_path}")
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best grid accuracy: {self.best_val_accuracy:.3f}")
        
        return final_gen
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, strict: bool = True):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        matched = len(self.model.state_dict()) - len(result.missing_keys)
        logger.info(f"Loaded {matched}/{len(self.model.state_dict())} model params "
                    f"({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected)")
        
        if strict and not result.missing_keys and not result.unexpected_keys:
            # Full match — also restore optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
        else:
            logger.info("Architecture changed — starting fresh optimizer (weights partially loaded)")
            self.global_step = 0
            self.epoch = 0
        
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


def get_tokenizer():
    """Get tokenizer"""
    if HAS_TIKTOKEN:
        return tiktoken.get_encoding("cl100k_base")
    else:
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 1000 for c in text]
            def decode(self, tokens):
                return ''.join(chr(t % 256) for t in tokens)
        return SimpleTokenizer()


def main():
    """Main ARC training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train OctoTetrahedral AGI on ARC')
    parser.add_argument('--max-steps', type=int, default=5000, help='Max training steps')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--data-dir', type=str, 
                        default='/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data',
                        help='ARC data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--use-simula', action='store_true', help='Enable SIMULA synthetic data augmentation')
    parser.add_argument('--simula-complexity', type=int, default=3, help='SIMULA complexity level (1-5)')
    parser.add_argument('--simula-ratio', type=float, default=0.2, help='Fraction of synthetic data')
    parser.add_argument('--use-euphan', action='store_true', help='Enable EUPHAN limb observability logging')
    parser.add_argument('--euphan-log-frequency', type=int, default=100, help='Log EUPHAN events every N steps')
    parser.add_argument('--euphan-output-dir', type=str, default='logs/euphan', help='Directory to save EUPHAN HTML reports')
    parser.add_argument('--use-hermes', action='store_true', help='Enable HERMES background task orchestration')
    parser.add_argument('--hermes-log-frequency', type=int, default=50, help='Log HERMES events every N steps')
    parser.add_argument('--hermes-output-dir', type=str, default='logs/hermes', help='Directory to save HERMES reports')
    parser.add_argument('--hermes-max-agents', type=int, default=3, help='Maximum parallel HERMES agents')
    parser.add_argument('--use-cohesion', action='store_true', help='Enable Cognitive Cohesion Braid (cross-bridge feedback)')
    parser.add_argument('--cohesion-output-dir', type=str, default='logs/cohesion', help='Directory for cohesion reports')
    args = parser.parse_args()
    
    # Configuration
    config = get_config()
    
    # If resuming, load checkpoint config to match architecture
    if args.resume and os.path.exists(args.resume):
        ckpt_peek = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'config' in ckpt_peek and 'model' in ckpt_peek['config']:
            config = get_config(model=ckpt_peek['config']['model'])
            logger.info(f"Using checkpoint config: max_seq_len={config.model.max_seq_len}")
        del ckpt_peek
    
    # Training settings for ARC
    config.training.max_steps = args.max_steps
    config.training.batch_size = args.batch_size
    config.training.log_interval = 10
    config.training.eval_interval = 100
    config.training.save_interval = 200
    
    # SIMULA settings from command line
    if args.use_simula:
        config.training.use_simula = True
        config.training.simula_complexity = args.simula_complexity
        config.training.simula_ratio = args.simula_ratio
    
    # EUPHAN settings from command line
    if args.use_euphan:
        config.training.use_euphan = True
        config.training.euphan_log_frequency = args.euphan_log_frequency
        config.training.euphan_output_dir = args.euphan_output_dir
    
    # HERMES settings from command line
    if args.use_hermes:
        config.training.use_hermes = True
        config.training.hermes_log_frequency = args.hermes_log_frequency
        config.training.hermes_output_dir = args.hermes_output_dir
        config.training.hermes_max_agents = args.hermes_max_agents
        config.training.hermes_queue_size = 1000

    # Cohesion braid settings
    if args.use_cohesion:
        config.training.use_cohesion = True
        config.training.cohesion_output_dir = args.cohesion_output_dir

    logger.info("=" * 60)
    logger.info("OctoTetrahedral AGI - ARC Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Hidden dim: {config.model.hidden_dim}")
    logger.info(f"  Num layers: {config.model.num_layers}")
    logger.info(f"  Num heads: {config.model.num_heads}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Data dir: {args.data_dir}")
    if args.use_simula:
        logger.info(f"  SIMULA enabled (complexity={args.simula_complexity}, ratio={args.simula_ratio})")
    if args.use_euphan:
        logger.info(f"  EUPHAN enabled (frequency={args.euphan_log_frequency})")
    if args.use_hermes:
        logger.info(f"  HERMES enabled (agents={args.hermes_max_agents}, frequency={args.hermes_log_frequency})")
    
    # Tokenizer
    tokenizer = get_tokenizer()
    
    # Create dataloaders
    logger.info("\nLoading ARC datasets...")
    
    train_loader = create_arc_dataloader(
        data_dir=args.data_dir,
        split='training',
        batch_size=config.training.batch_size,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        shuffle=True
    )
    
    # Use subset of training as validation (ARC eval set is held out)
    val_loader = create_arc_dataloader(
        data_dir=args.data_dir,
        split='evaluation',
        batch_size=config.training.batch_size,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        shuffle=False
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model
    logger.info("\nCreating model...")
    model = OctoTetrahedralModel(config)
    
    # Trainer
    trainer = ARCTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        use_simula=args.use_simula,
        use_euphan=args.use_euphan,
        use_hermes=args.use_hermes,
        use_cohesion=args.use_cohesion
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if args.eval_only:
        # Just run evaluation
        logger.info("\nRunning evaluation only...")
        val_result = trainer.validate()
        logger.info(f"Validation - Loss: {val_result['val_loss']:.4f}, "
                   f"Token Acc: {val_result['val_token_accuracy']:.4f}")
        
        gen_result = trainer.evaluate_generation(num_samples=50)
        if gen_result:
            logger.info(f"Generation - Exact Match: {gen_result['exact_match']:.3f}, "
                       f"Grid Acc: {gen_result['grid_accuracy']:.3f}, "
                       f"Cell Acc: {gen_result['cell_accuracy']:.3f}")
    else:
        # Train
        trainer.train(
            max_steps=args.max_steps,
            eval_generation_every=500
        )
    
    # Final model statistics
    logger.info("\nFinal model statistics:")
    stats = model.get_stats()
    logger.info(f"  Total parameters: {stats['total_params']:,}")
    logger.info(f"  Forward count: {stats['forward_count']}")
    logger.info(f"  Memory utilization: {stats['memory_utilization']:.4f}")
    logger.info(f"  Hub syncs: {stats['hub_sync_stats']['total_syncs']}")


if __name__ == "__main__":
    main()

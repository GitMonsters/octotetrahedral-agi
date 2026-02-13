"""
OctoTetrahedral AGI - ARC-AGI Training Script
Trains the model on Abstract Reasoning Corpus (ARC) tasks.

ARC tasks require:
- Few-shot learning from example input/output pairs
- Abstract pattern recognition and generalization
- Grid-based spatial reasoning
"""

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
    """
    
    def __init__(
        self,
        model: OctoTetrahedralModel,
        config: Config,
        train_dataloader,
        val_dataloader=None,
        tokenizer=None,
        device: str = None
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
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
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
    args = parser.parse_args()
    
    # Configuration
    config = get_config()
    
    # Training settings for ARC
    config.training.max_steps = args.max_steps
    config.training.batch_size = args.batch_size
    config.training.log_interval = 10
    config.training.eval_interval = 100
    config.training.save_interval = 500
    
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
        tokenizer=tokenizer
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

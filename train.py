"""
OctoTetrahedral AGI - Training Script
Implements curiosity-driven learning with limb synchronization

Training features:
- Prediction loss + information gain bonus
- Periodic limb synchronization (FedAvg)
- Gradient clipping for stability
- Learning rate warmup and decay
- Validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from typing import Optional, Dict, Any, Tuple
import time
import math
import logging
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not installed. Using simple tokenization.")

from config import Config, get_config
from model import OctoTetrahedralModel
from data.synthetic_tasks import (
    create_dataloader, 
    SyntheticTaskDataset,
    TaskType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for OctoTetrahedral AGI.
    
    Features:
    - Curiosity-driven learning (prediction loss + info gain)
    - Periodic limb synchronization
    - Gradient clipping
    - Learning rate scheduling
    - Validation and checkpointing
    """
    
    def __init__(
        self,
        model: OctoTetrahedralModel,
        config: Config,
        train_dataloader,
        val_dataloader=None,
        device: str = None,
        gradient_checkpointing: bool = False,
        mixed_precision: bool = False,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Device
        self.device = device or config.device
        self.model.to(self.device)
        
        # Mixed precision (AMP)
        self.mixed_precision = mixed_precision
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        if mixed_precision:
            logger.info("Mixed precision (bfloat16) training enabled")
        
        # Enable gradient checkpointing for memory efficiency with large models
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.train_losses: list = []
        self.val_losses: list = []
        
        # Checkpointing
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on transformer layers for memory savings.
        
        Note: With MoE, gradient checkpointing wraps only the attention sub-block
        since MoE routing is non-deterministic and incompatible with recomputation.
        For MoE models, consider using FSDP activation offloading instead.
        """
        if self.config.moe.enabled:
            logger.warning(
                "Gradient checkpointing with MoE may cause issues due to "
                "non-deterministic routing. Using standard checkpointing on "
                "attention blocks only."
            )
        from torch.utils.checkpoint import checkpoint
        for layer in self.model.core.layers:
            layer._original_forward = layer.forward
            def make_ckpt_forward(mod):
                def ckpt_forward(*args, **kwargs):
                    return checkpoint(mod._original_forward, *args, use_reentrant=False, **kwargs)
                return ckpt_forward
            layer.forward = make_ckpt_forward(layer)
        logger.info(f"Gradient checkpointing enabled on {len(self.model.core.layers)} layers")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
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
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=self.config.training.betas
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup"""
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with optional mixed precision"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass (with optional AMP)
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.mixed_precision else torch.amp.autocast('cuda', enabled=False)
        with amp_ctx:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_confidences=True
            )
            loss = output['loss']
        
        # Collect MoE metrics for logging
        moe_aux_loss = output.get('moe_aux_loss')
        
        # Backward pass (with optional scaler)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.config.training.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
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
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'confidences': output.get('confidences', {}),
            'synced': sync_result is not None,
            'moe_aux_loss': moe_aux_loss.item() if moe_aux_loss is not None else None,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
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
            
            # Compute accuracy (ignoring padding)
            preds = output['logits'].argmax(dim=-1)
            mask = labels != -100
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        accuracy = total_correct / max(1, total_tokens)
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(
        self,
        num_epochs: int = None,
        max_steps: int = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs (overrides config)
            max_steps: Max steps (overrides config)
        """
        max_steps = max_steps or self.config.training.max_steps
        
        logger.info(f"Starting training for {max_steps} steps")
        logger.info(f"Model has {self.model.get_num_params():,} total parameters")
        if self.config.moe.enabled:
            logger.info(
                f"MoE: {self.config.moe.num_experts} experts, "
                f"top-{self.config.moe.top_k} routing, "
                f"~{self.model.get_active_params():,} active params/token"
            )
        logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        running_loss = 0.0
        
        while self.global_step < max_steps:
            for batch in self.train_dataloader:
                if self.global_step >= max_steps:
                    break
                
                # Training step
                step_result = self.train_step(batch)
                running_loss += step_result['loss']
                
                # Logging
                if self.global_step % self.config.training.log_interval == 0:
                    avg_loss = running_loss / self.config.training.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {step_result['lr']:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )
                    
                    if step_result.get('moe_aux_loss') is not None:
                        logger.info(
                            f"  MoE aux loss: {step_result['moe_aux_loss']:.4f}"
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
                
                # Validation
                if (self.global_step % self.config.training.eval_interval == 0 
                    and self.val_dataloader is not None):
                    val_result = self.validate()
                    logger.info(
                        f"  Validation - Loss: {val_result['val_loss']:.4f}, "
                        f"Accuracy: {val_result['val_accuracy']:.4f}"
                    )
                    self.val_losses.append(val_result['val_loss'])
                    
                    # Save best
                    if val_result['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_result['val_loss']
                        self.save_checkpoint('best.pt')
                
                # Periodic checkpointing
                if self.global_step % self.config.training.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')
            
            self.epoch += 1
        
        # Final checkpoint
        self.save_checkpoint('final.pt')
        
        total_time = time.time() - start_time
        logger.info(f"Training complete! Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
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
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


def get_tokenizer():
    """Get tokenizer (tiktoken if available, else simple)"""
    if HAS_TIKTOKEN:
        return tiktoken.get_encoding("cl100k_base")
    else:
        # Simple character-level tokenizer fallback
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 1000 for c in text]
            def decode(self, tokens):
                return ''.join(chr(t % 256) for t in tokens)
        return SimpleTokenizer()


def main():
    """Main training entry point"""
    # Configuration
    config = get_config()
    
    # Extended training run
    config.training.max_steps = 3000
    config.training.log_interval = 50
    config.training.eval_interval = 200
    config.training.save_interval = 500
    
    logger.info("Configuration:")
    logger.info(f"  Hidden dim: {config.model.hidden_dim}")
    logger.info(f"  Num layers: {config.model.num_layers}")
    logger.info(f"  Num heads: {config.model.num_heads}")
    logger.info(f"  Sync frequency: {config.sync.sync_frequency}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    
    # Tokenizer
    tokenizer = get_tokenizer()
    
    # Data
    logger.info("Creating datasets...")
    train_loader = create_dataloader(
        num_samples=5000,
        batch_size=config.training.batch_size,
        task_types=[
            TaskType.ARITHMETIC,
            TaskType.PATTERN,
            TaskType.COPY
        ],
        difficulty_range=(1, 3),
        seed=config.seed,
        tokenizer=tokenizer,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        num_samples=500,
        batch_size=config.training.batch_size,
        task_types=[
            TaskType.ARITHMETIC,
            TaskType.PATTERN,
            TaskType.COPY
        ],
        difficulty_range=(1, 3),
        seed=config.seed + 1,
        tokenizer=tokenizer,
        shuffle=False
    )
    
    # Model
    logger.info("Creating model...")
    model = OctoTetrahedralModel(config)
    
    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Resume from checkpoint if available
    checkpoint_path = Path("checkpoints/best.pt")
    if checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
    
    # Train
    trainer.train()
    
    # Final evaluation
    logger.info("\nFinal model statistics:")
    stats = model.get_stats()
    logger.info(f"  Total parameters: {stats['total_params']:,}")
    logger.info(f"  Forward count: {stats['forward_count']}")
    logger.info(f"  Memory utilization: {stats['memory_utilization']:.4f}")
    logger.info(f"  Hub syncs: {stats['hub_sync_stats']['total_syncs']}")


if __name__ == "__main__":
    main()

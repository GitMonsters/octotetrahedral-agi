"""
OctoTetrahedral AGI - Small Model Training Script
Faster iteration with ~15M parameter model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any
import time
import math
import logging
from pathlib import Path

import tiktoken

from config import Config, get_config, ModelConfig, GeometricPhysicsConfig
from model import OctoTetrahedralModel
from data.synthetic_tasks import create_dataloader, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_small_config() -> Config:
    """Create a small model configuration for fast iteration"""
    config = get_config()
    
    # Smaller model dimensions (~15M params instead of 108M)
    config.model.hidden_dim = 128  # Was 256
    config.model.ffn_dim = 512     # Was 1024
    config.model.num_layers = 2    # Was 3
    config.model.num_heads = 4     # Was 8
    config.model.head_dim = 32     # 128 // 4
    config.model.memory_slots = 2  # Was 4
    
    # Smaller geometry
    config.rna_editing.num_gated_heads = 4  # Match num_heads
    
    # Simpler physics (disable heavy modules for speed)
    config.geometric_physics.enable_fuller = True
    config.geometric_physics.enable_lloyd = False  # Expensive
    config.geometric_physics.enable_morphogenesis = False  # Expensive
    config.geometric_physics.enable_tpms = True
    config.geometric_physics.enable_qbit_nexus = False
    config.geometric_physics.enable_parallel_universe = False
    config.geometric_physics.combination_mode = 'learnable'  # Simpler than compound
    
    # Smaller physics modules
    config.geometric_physics.fuller_ve_vertices = 6  # Was 12
    config.geometric_physics.tpms_num_heads = 4  # Was 8
    
    # Faster training
    config.training.learning_rate = 3e-4  # Higher LR for small model
    config.training.batch_size = 16  # Larger batches
    config.training.warmup_steps = 50
    config.training.max_steps = 5000
    
    # More frequent logging
    config.training.log_interval = 25
    config.training.eval_interval = 100
    config.training.save_interval = 500
    
    # Less frequent sync
    config.sync.sync_frequency = 20
    
    return config


class SmallTrainer:
    """Simplified trainer for small model"""
    
    def __init__(self, model, config, train_loader, val_loader, device):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        warmup = config.training.warmup_steps
        max_steps = config.training.max_steps
        
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("checkpoints/small")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch):
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output['loss']
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += output['loss'].item()
            
            preds = output['logits'].argmax(dim=-1)
            mask = labels != -100
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / max(1, num_batches),
            'val_acc': total_correct / max(1, total_tokens)
        }
    
    def save_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self):
        max_steps = self.config.training.max_steps
        log_interval = self.config.training.log_interval
        eval_interval = self.config.training.eval_interval
        save_interval = self.config.training.save_interval
        
        logger.info(f"Training for {max_steps} steps")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        running_loss = 0
        
        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                result = self.train_step(batch)
                running_loss += result['loss']
                
                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    speed = self.global_step / elapsed
                    
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {result['lr']:.2e} | "
                        f"Speed: {speed:.1f} steps/s"
                    )
                    running_loss = 0
                
                if self.global_step % eval_interval == 0:
                    val_result = self.validate()
                    logger.info(
                        f"  Val Loss: {val_result['val_loss']:.4f} | "
                        f"Val Acc: {val_result['val_acc']*100:.1f}%"
                    )
                    
                    if val_result['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_result['val_loss']
                        self.save_checkpoint('best.pt')
                
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')
        
        self.save_checkpoint('final.pt')
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")


def test_inference(model, device, enc):
    """Quick inference test"""
    model.eval()
    
    test_cases = [
        ('Calculate: 5 + 3 = ', '8'),
        ('Calculate: 2 + 2 = ', '4'),
        ('Continue the pattern: 1 2 3 4 ', '5'),
        ('Continue the pattern: 2 4 6 8 ', '10'),
    ]
    
    print("\n" + "="*60)
    print("INFERENCE TEST")
    print("="*60)
    
    correct = 0
    for prompt, expected in test_cases:
        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens]).to(device)
        
        with torch.no_grad():
            output = model(input_ids)
            logits = output['logits'][0, -1]
            probs = torch.softmax(logits, dim=-1)
            
            top5_probs, top5_ids = probs.topk(5)
            top5_tokens = [enc.decode([tid.item()]).strip() for tid in top5_ids]
            
            is_correct = expected in top5_tokens
            if is_correct:
                correct += 1
        
        status = '✓' if is_correct else '✗'
        top_pred = top5_tokens[0]
        print(f"{status} {prompt}→ Expected: {expected} | Predicted: {top_pred} | Top5: {top5_tokens}")
    
    print(f"\nAccuracy (top-5): {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
    return correct / len(test_cases)


def main():
    # Get small config
    config = get_small_config()
    
    logger.info("="*60)
    logger.info("SMALL MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Hidden dim: {config.model.hidden_dim}")
    logger.info(f"Num layers: {config.model.num_layers}")
    logger.info(f"Num heads: {config.model.num_heads}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Max steps: {config.training.max_steps}")
    
    # Tokenizer
    enc = tiktoken.get_encoding('cl100k_base')
    
    # Data - focus on simpler tasks
    logger.info("Creating datasets...")
    train_loader = create_dataloader(
        num_samples=10000,
        batch_size=config.training.batch_size,
        task_types=[TaskType.ARITHMETIC, TaskType.PATTERN, TaskType.COPY],
        difficulty_range=(1, 2),  # Easier tasks
        seed=config.seed,
        tokenizer=enc,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        num_samples=500,
        batch_size=config.training.batch_size,
        task_types=[TaskType.ARITHMETIC, TaskType.PATTERN, TaskType.COPY],
        difficulty_range=(1, 2),
        seed=config.seed + 1,
        tokenizer=enc,
        shuffle=False
    )
    
    # Model
    logger.info("Creating model...")
    model = OctoTetrahedralModel(config)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Trainer
    trainer = SmallTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device
    )
    
    # Train
    trainer.train()
    
    # Test inference
    test_inference(model, config.device, enc)


if __name__ == "__main__":
    main()

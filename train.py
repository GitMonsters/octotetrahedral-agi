"""
Training Script for Unified Cognitive Stack
============================================

Complete training pipeline with metrics, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from datetime import datetime
import argparse

from unified import UnifiedForwardModel


class ARCDataset(Dataset):
    """Dummy ARC dataset for demonstration."""
    
    def __init__(self, num_examples: int = 1000, seq_len: int = 32, vocab_size: int = 1000):
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


class Trainer:
    """Training orchestrator for unified model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_interval = log_interval
        
        self.metrics_history = {
            'train_loss': [],
            'train_rna_loss': [],
            'train_quantum_loss': [],
            'val_loss': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_rna_loss = 0.0
        total_quantum_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            output = self.model(input_ids, labels=labels)
            loss = output['loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_rna_loss += output['metrics'].get('rna_loss', 0.0)
            total_quantum_loss += output['metrics'].get('quantum_loss', 0.0)
            num_batches += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_rna_loss = total_rna_loss / num_batches
        avg_quantum_loss = total_quantum_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_rna_loss': avg_rna_loss,
            'train_quantum_loss': avg_quantum_loss,
        }
    
    def eval(
        self,
        val_loader: DataLoader,
    ) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(input_ids, labels=labels)
                loss = output['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': self.metrics_history,
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to file and history."""
        for key, value in train_metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        for key, value in val_metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Save to JSON
        log_path = self.checkpoint_dir / "metrics.json"
        with open(log_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


def main(args):
    """Main training script."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = UnifiedForwardModel(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_limbs=8,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        enable_quantum=args.enable_quantum,
        enable_rna_editing=True,
    ).to(device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    train_dataset = ARCDataset(num_examples=args.train_size, seq_len=args.seq_len)
    val_dataset = ARCDataset(num_examples=args.val_size, seq_len=args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch + 1)
        
        # Eval
        val_metrics = trainer.eval(val_loader)
        
        # Log
        trainer.log_metrics(epoch + 1, train_metrics, val_metrics)
        
        # Print summary
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
        print(f"Train RNA Loss: {train_metrics['train_rna_loss']:.4f}")
        print(f"Train Quantum Loss: {train_metrics['train_quantum_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            trainer.save_checkpoint(epoch + 1, optimizer)
        
        # LR scheduling
        scheduler.step()
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    
    # Final evaluation
    print(f"\nFinal model stats:")
    stats = model.get_stats()
    for key, value in stats.items():
        if key != 'last_metrics':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unified cognitive model")
    
    # Model
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--enable-quantum', action='store_true', default=True)
    
    # Data
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-interval', type=int, default=10)
    
    args = parser.parse_args()
    main(args)

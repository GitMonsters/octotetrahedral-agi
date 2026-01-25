"""
Quick Training Script with Quantum Coupling

Trains the OctoTetrahedral model with quantum coupling regularization.
This script is designed for quick iterations to test the quantum dynamics hypothesis.

Usage:
    python train_quantum.py --steps 500 --device cpu
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import get_config
from model import OctoTetrahedralModel
from sync.quantum_coupling import QuantumCouplingLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumTrainer:
    """
    Trainer with quantum coupling regularization.
    """
    
    def __init__(
        self,
        model: OctoTetrahedralModel,
        device: str = 'cpu',
        lr: float = 1e-4,
        coherent_weight: float = 0.01,
        quantization_weight: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        
        # Quantum coupling layer
        self.quantum_coupling = QuantumCouplingLayer(
            hidden_dim=model.hidden_dim,
            num_limbs=8,
            zero_point_energy=7.0,
            coupling_strength=0.1
        ).to(device)
        
        # Optimizer for both model and coupling
        self.optimizer = AdamW([
            {'params': model.parameters(), 'lr': lr},
            {'params': self.quantum_coupling.parameters(), 'lr': lr * 10}  # Faster coupling learning
        ])
        
        # Loss weights
        self.coherent_weight = coherent_weight
        self.quantization_weight = quantization_weight
        
        # Tracking
        self.step = 0
        self.losses = []
    
    def train_step(self, batch_size: int = 4, seq_len: int = 32) -> dict:
        """Single training step with synthetic data."""
        self.model.train()
        self.quantum_coupling.train()
        
        # Generate synthetic data
        input_ids = torch.randint(
            0, self.model.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        labels = input_ids.clone()
        
        # Forward pass
        output = self.model(
            input_ids=input_ids,
            labels=labels,
            return_confidences=True
        )
        
        main_loss = output['loss']
        
        # Collect limb hidden states for quantum coupling
        # We'll use a proxy: the hidden_states output
        hidden_states = output.get('hidden_states', None)
        
        if hidden_states is not None:
            # Create mock limb states from hidden states
            # Split hidden state into 8 parts for 8 limbs
            limb_dim = hidden_states.shape[-1] // 8
            limb_states = {}
            for i, name in enumerate(QuantumCouplingLayer.LIMB_NAMES):
                start = i * limb_dim
                end = start + limb_dim
                if end <= hidden_states.shape[-1]:
                    # Pad to full hidden_dim
                    partial = hidden_states[:, :, start:end]
                    padded = F.pad(partial, (0, self.model.hidden_dim - limb_dim))
                    limb_states[name] = padded
            
            # Apply quantum coupling
            _, quantum_losses = self.quantum_coupling(limb_states, time_step=self.step)
            
            coherent_loss = quantum_losses.get('coherent_loss', torch.tensor(0.0))
            quant_loss = quantum_losses.get('quantization_loss', torch.tensor(0.0))
        else:
            coherent_loss = torch.tensor(0.0)
            quant_loss = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            main_loss + 
            self.coherent_weight * coherent_loss +
            self.quantization_weight * quant_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'coherent_loss': coherent_loss.item() if isinstance(coherent_loss, torch.Tensor) else 0,
            'quant_loss': quant_loss.item() if isinstance(quant_loss, torch.Tensor) else 0
        }
    
    def train(self, num_steps: int, log_interval: int = 10, save_interval: int = 100):
        """Main training loop."""
        logger.info(f"Starting training for {num_steps} steps")
        logger.info(f"Model params: {self.model.get_num_params():,}")
        logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        running_loss = 0.0
        
        for step in range(num_steps):
            result = self.train_step()
            running_loss += result['total_loss']
            self.losses.append(result['total_loss'])
            
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                
                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Main: {result['main_loss']:.4f} | "
                    f"Coherent: {result['coherent_loss']:.4f} | "
                    f"Quant: {result['quant_loss']:.4f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )
                
                # Log quantum coupling stats
                stats = self.quantum_coupling.get_stats()
                logger.info(
                    f"  Quantum - ZPE: {stats['zero_point_energy']:.2f} | "
                    f"Coupling: {stats['mean_coupling']:.4f} | "
                    f"Energy: {stats['total_energy']:.2f}"
                )
                
                running_loss = 0.0
            
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(f'quantum_step_{step + 1}.pt')
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f}s")
        
        # Final save
        self.save_checkpoint('quantum_final.pt')
        
        return self.losses
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = Path('checkpoints') / filename
        path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'quantum_coupling_state_dict': self.quantum_coupling.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'losses': self.losses
        }, path)
        
        logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--coherent_weight', type=float, default=0.01, help='Coherent loss weight')
    parser.add_argument('--quant_weight', type=float, default=0.001, help='Quantization loss weight')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create model
    config = get_config()
    model = OctoTetrahedralModel(config)
    
    # Create trainer
    trainer = QuantumTrainer(
        model=model,
        device=device,
        lr=args.lr,
        coherent_weight=args.coherent_weight,
        quantization_weight=args.quant_weight
    )
    
    # Train
    trainer.train(num_steps=args.steps)
    
    print("\nTraining complete!")
    print(f"Final quantum coupling stats:")
    stats = trainer.quantum_coupling.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ARC Neural Solver - Specialized Grid-to-Grid Transformer

Instead of treating ARC as language modeling, this treats it as:
1. Encoding input grids as 2D tensors
2. Processing examples with cross-attention
3. Directly predicting output grid cells

Key innovations:
- Small vocabulary: just 0-9 colors + padding
- Grid-aware positional encoding (2D)
- Example-based conditioning (few-shot attention)
- Direct cell prediction (not token-by-token generation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import time


# ==============================================================================
# Grid Encoding/Decoding
# ==============================================================================

class GridEncoder:
    """Encode ARC grids for neural processing"""
    
    # Vocabulary: 0-9 colors + PAD (10) + SEP (11)
    VOCAB_SIZE = 12
    PAD_TOKEN = 10
    SEP_TOKEN = 11
    
    @staticmethod
    def grid_to_tensor(grid: List[List[int]], max_h: int = 30, max_w: int = 30) -> torch.Tensor:
        """Convert grid to padded tensor [H, W]"""
        h, w = len(grid), len(grid[0]) if grid else 0
        tensor = torch.full((max_h, max_w), GridEncoder.PAD_TOKEN, dtype=torch.long)
        
        for i in range(min(h, max_h)):
            for j in range(min(w, max_w)):
                tensor[i, j] = grid[i][j]
        
        return tensor
    
    @staticmethod
    def tensor_to_grid(tensor: torch.Tensor, original_h: int, original_w: int) -> List[List[int]]:
        """Convert tensor back to grid"""
        grid = []
        for i in range(original_h):
            row = []
            for j in range(original_w):
                val = tensor[i, j].item()
                if val == GridEncoder.PAD_TOKEN:
                    val = 0
                row.append(val)
            grid.append(row)
        return grid
    
    @staticmethod
    def get_grid_size(grid: List[List[int]]) -> Tuple[int, int]:
        """Get (h, w) of grid"""
        return len(grid), len(grid[0]) if grid else 0


# ==============================================================================
# 2D Positional Encoding
# ==============================================================================

class GridPositionalEncoding(nn.Module):
    """2D positional encoding for grid structure"""
    
    def __init__(self, hidden_dim: int, max_h: int = 30, max_w: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable position embeddings
        self.row_embed = nn.Embedding(max_h, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_w, hidden_dim // 2)
    
    def forward(self, h: int, w: int) -> torch.Tensor:
        """Return positional encoding [H, W, D]"""
        device = self.row_embed.weight.device
        
        row_pos = torch.arange(h, device=device)
        col_pos = torch.arange(w, device=device)
        
        row_enc = self.row_embed(row_pos)  # [H, D/2]
        col_enc = self.col_embed(col_pos)  # [W, D/2]
        
        # Combine: [H, 1, D/2] + [1, W, D/2] -> [H, W, D]
        pos_enc = torch.cat([
            row_enc.unsqueeze(1).expand(-1, w, -1),
            col_enc.unsqueeze(0).expand(h, -1, -1)
        ], dim=-1)
        
        return pos_enc


# ==============================================================================
# Example Encoder (processes input-output pairs)
# ==============================================================================

class ExampleEncoder(nn.Module):
    """Encode a single input-output example pair"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embed colors (0-9 + PAD + SEP)
        self.color_embed = nn.Embedding(GridEncoder.VOCAB_SIZE, hidden_dim)
        
        # 2D position encoding
        self.pos_enc = GridPositionalEncoding(hidden_dim, max_size, max_size)
        
        # Self-attention for grid
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention between input and output
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def encode_grid(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a single grid [H, W] -> [H*W, D]"""
        h, w = grid_tensor.shape
        device = grid_tensor.device
        
        # Color embedding
        x = self.color_embed(grid_tensor)  # [H, W, D]
        
        # Add 2D positional encoding
        pos = self.pos_enc(h, w).to(device)  # [H, W, D]
        x = x + pos
        
        # Flatten to sequence
        x = x.view(h * w, -1)  # [H*W, D]
        
        return x
    
    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input-output pair.
        
        Args:
            input_grid: [H_in, W_in]
            output_grid: [H_out, W_out]
            
        Returns:
            Example representation [N, D] where N = H_in*W_in + H_out*W_out
        """
        # Encode both grids
        in_enc = self.encode_grid(input_grid).unsqueeze(0)   # [1, H*W_in, D]
        out_enc = self.encode_grid(output_grid).unsqueeze(0)  # [1, H*W_out, D]
        
        # Self-attention on input
        attn_out, _ = self.self_attn(in_enc, in_enc, in_enc)
        in_enc = self.norm1(in_enc + attn_out)
        in_enc = self.norm2(in_enc + self.ffn(in_enc))
        
        # Cross-attention: output attends to input
        cross_out, _ = self.cross_attn(out_enc, in_enc, in_enc)
        out_enc = self.norm3(out_enc + cross_out)
        
        # Concatenate for full example representation
        example_enc = torch.cat([in_enc, out_enc], dim=1)  # [1, N_in + N_out, D]
        
        return example_enc.squeeze(0)  # [N, D]


# ==============================================================================
# ARC Neural Model
# ==============================================================================

class ARCNeuralModel(nn.Module):
    """
    Neural model for ARC tasks.
    
    Architecture:
    1. Encode each training example (input->output pair)
    2. Aggregate example encodings
    3. Condition on test input
    4. Predict output grid directly
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_grid_size: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Example encoder
        self.example_encoder = ExampleEncoder(hidden_dim, num_heads, max_grid_size)
        
        # Color embedding for test input
        self.color_embed = nn.Embedding(GridEncoder.VOCAB_SIZE, hidden_dim)
        self.pos_enc = GridPositionalEncoding(hidden_dim, max_grid_size, max_grid_size)
        
        # Transformer layers for test input processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention to examples
        self.example_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.example_norm = nn.LayerNorm(hidden_dim)
        
        # Size predictor (predict output size)
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [height, width]
        )
        
        # Output cell predictor
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 10 colors (0-9)
        )
    
    def encode_examples(
        self,
        examples: List[Dict]
    ) -> torch.Tensor:
        """Encode all training examples into a single representation"""
        device = self.color_embed.weight.device
        
        example_encodings = []
        for ex in examples:
            in_tensor = GridEncoder.grid_to_tensor(ex['input'], self.max_grid_size, self.max_grid_size).to(device)
            out_tensor = GridEncoder.grid_to_tensor(ex['output'], self.max_grid_size, self.max_grid_size).to(device)
            
            # Get actual sizes
            in_h, in_w = GridEncoder.get_grid_size(ex['input'])
            out_h, out_w = GridEncoder.get_grid_size(ex['output'])
            
            # Encode only the actual grid (not padding)
            enc = self.example_encoder(
                in_tensor[:in_h, :in_w],
                out_tensor[:out_h, :out_w]
            )
            example_encodings.append(enc)
        
        # Pad and stack examples
        max_len = max(enc.shape[0] for enc in example_encodings)
        padded = []
        for enc in example_encodings:
            if enc.shape[0] < max_len:
                pad = torch.zeros(max_len - enc.shape[0], self.hidden_dim, device=device)
                enc = torch.cat([enc, pad], dim=0)
            padded.append(enc)
        
        # Stack: [num_examples, max_len, hidden_dim]
        stacked = torch.stack(padded, dim=0)
        
        # Mean pool across examples and positions -> [1, D]
        pooled = stacked.mean(dim=(0, 1))  # [D]
        return pooled.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
    
    def forward(
        self,
        test_input: torch.Tensor,
        examples_enc: torch.Tensor,
        target_output: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            test_input: [B, H, W] test input grid
            examples_enc: [B, N, D] encoded examples
            target_output: [B, H_out, W_out] target grid (for training)
            target_size: (H_out, W_out) for inference
        """
        batch_size = test_input.shape[0]
        h, w = test_input.shape[1], test_input.shape[2]
        device = test_input.device
        
        # Encode test input
        x = self.color_embed(test_input)  # [B, H, W, D]
        pos = self.pos_enc(h, w).to(device)  # [H, W, D]
        x = x + pos.unsqueeze(0)  # [B, H, W, D]
        
        # Flatten to sequence
        x = x.view(batch_size, h * w, self.hidden_dim)  # [B, H*W, D]
        
        # Transformer processing
        x = self.transformer(x)  # [B, H*W, D]
        
        # Cross-attention to examples
        examples_expanded = examples_enc.expand(batch_size, -1, -1)  # [B, N, D]
        attn_out, _ = self.example_attn(x, examples_expanded, examples_expanded)
        x = self.example_norm(x + attn_out)  # [B, H*W, D]
        
        # Global representation for size prediction
        global_repr = x.mean(dim=1)  # [B, D]
        
        # Predict output size
        size_pred = self.size_predictor(global_repr)  # [B, 2]
        pred_h = torch.clamp(size_pred[:, 0], 1, self.max_grid_size).round().int()
        pred_w = torch.clamp(size_pred[:, 1], 1, self.max_grid_size).round().int()
        
        # For output prediction, we need output positions
        if target_output is not None:
            out_h, out_w = target_output.shape[1], target_output.shape[2]
        elif target_size is not None:
            out_h, out_w = target_size
        else:
            out_h, out_w = pred_h[0].item(), pred_w[0].item()
        
        # Create output position queries
        out_pos = self.pos_enc(out_h, out_w).to(device)  # [H_out, W_out, D]
        out_queries = out_pos.view(1, out_h * out_w, self.hidden_dim)  # [1, N_out, D]
        out_queries = out_queries.expand(batch_size, -1, -1)  # [B, N_out, D]
        
        # Cross-attention: output queries attend to encoded input
        out_repr, _ = self.example_attn(out_queries, x, x)  # [B, N_out, D]
        
        # Predict colors
        logits = self.output_head(out_repr)  # [B, N_out, 10]
        logits = logits.view(batch_size, out_h, out_w, 10)  # [B, H_out, W_out, 10]
        
        result = {
            'logits': logits,
            'pred_size': (pred_h, pred_w)
        }
        
        # Compute loss if target provided
        if target_output is not None:
            # Cross-entropy loss over valid positions
            loss = F.cross_entropy(
                logits.view(-1, 10),
                target_output.view(-1),
                ignore_index=GridEncoder.PAD_TOKEN
            )
            
            # Size loss
            target_h = torch.tensor([target_output.shape[1]], device=device, dtype=torch.float)
            target_w = torch.tensor([target_output.shape[2]], device=device, dtype=torch.float)
            size_loss = F.mse_loss(size_pred[:, 0], target_h) + F.mse_loss(size_pred[:, 1], target_w)
            
            result['loss'] = loss + 0.1 * size_loss
            result['cell_loss'] = loss
            result['size_loss'] = size_loss
        
        return result
    
    def predict(
        self,
        task: Dict,
        test_idx: int = 0
    ) -> List[List[int]]:
        """Predict output for a task"""
        self.eval()
        device = self.color_embed.weight.device
        
        with torch.no_grad():
            # Encode examples
            examples_enc = self.encode_examples(task['train'])
            
            # Get test input
            test_input_grid = task['test'][test_idx]['input']
            test_h, test_w = GridEncoder.get_grid_size(test_input_grid)
            test_tensor = GridEncoder.grid_to_tensor(
                test_input_grid, test_h, test_w
            ).unsqueeze(0).to(device)
            
            # Infer output size from examples
            out_sizes = [(len(ex['output']), len(ex['output'][0])) for ex in task['train']]
            avg_h = int(np.mean([s[0] for s in out_sizes]))
            avg_w = int(np.mean([s[1] for s in out_sizes]))
            
            # Forward pass
            result = self.forward(
                test_tensor,
                examples_enc,  # Already [1, 1, D]
                target_size=(avg_h, avg_w)
            )
            
            # Get predictions
            pred = result['logits'].argmax(dim=-1)  # [1, H, W]
            pred_grid = GridEncoder.tensor_to_grid(pred[0], avg_h, avg_w)
        
        return pred_grid


# ==============================================================================
# Training
# ==============================================================================

class ARCNeuralTrainer:
    """Trainer for ARC Neural Model"""
    
    def __init__(
        self,
        model: ARCNeuralModel,
        data_dir: str,
        device: str = 'cpu',
        lr: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.data_dir = Path(data_dir)
        
        # Load tasks
        self.train_tasks = self._load_tasks('training')
        self.val_tasks = self._load_tasks('evaluation')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-5
        )
        
        # Metrics
        self.train_losses = []
        self.val_accuracies = []
    
    def _load_tasks(self, split: str) -> List[Dict]:
        """Load all tasks from a split"""
        task_dir = self.data_dir / split
        tasks = []
        for json_file in sorted(task_dir.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)
            data['task_id'] = json_file.stem
            tasks.append(data)
        return tasks
    
    def train_step(self, task: Dict) -> float:
        """Single training step on one task"""
        self.model.train()
        
        # Encode examples
        examples_enc = self.model.encode_examples(task['train'])
        
        # For each training example, predict output from input
        total_loss = 0.0
        for ex in task['train']:
            in_h, in_w = GridEncoder.get_grid_size(ex['input'])
            out_h, out_w = GridEncoder.get_grid_size(ex['output'])
            
            input_tensor = GridEncoder.grid_to_tensor(
                ex['input'], in_h, in_w
            ).unsqueeze(0).to(self.device)
            
            target_tensor = GridEncoder.grid_to_tensor(
                ex['output'], out_h, out_w
            ).unsqueeze(0).to(self.device)
            
            # Forward
            result = self.model(
                input_tensor,
                examples_enc,  # Already [1, 1, D]
                target_output=target_tensor
            )
            
            total_loss += result['loss']
        
        # Average loss
        loss = total_loss / len(task['train'])
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, num_tasks: int = 50) -> Dict[str, float]:
        """Evaluate on validation tasks"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        for task in self.val_tasks[:num_tasks]:
            if 'output' not in task['test'][0]:
                continue
            
            pred = self.model.predict(task, test_idx=0)
            target = task['test'][0]['output']
            
            if pred == target:
                correct += 1
            total += 1
        
        return {
            'accuracy': correct / max(total, 1),
            'correct': correct,
            'total': total
        }
    
    def train(self, num_epochs: int = 10, eval_every: int = 100):
        """Main training loop"""
        print(f"Training on {len(self.train_tasks)} tasks")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        step = 0
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Shuffle tasks each epoch
            import random
            random.shuffle(self.train_tasks)
            
            epoch_loss = 0.0
            for task in self.train_tasks:
                loss = self.train_step(task)
                epoch_loss += loss
                step += 1
                
                if step % 10 == 0:
                    print(f"Step {step} | Loss: {loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
                
                if step % eval_every == 0:
                    metrics = self.evaluate(num_tasks=50)
                    print(f"  Eval: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.1%}")
                    
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        torch.save(self.model.state_dict(), 'checkpoints/arc/arc_neural_best.pt')
                        print(f"  New best model saved!")
                    
                    self.val_accuracies.append(metrics['accuracy'])
                
                self.scheduler.step()
            
            avg_loss = epoch_loss / len(self.train_tasks)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        
        # Final evaluation
        print("\nFinal evaluation on all validation tasks...")
        final_metrics = self.evaluate(num_tasks=len(self.val_tasks))
        print(f"Final accuracy: {final_metrics['correct']}/{final_metrics['total']} = {final_metrics['accuracy']:.1%}")
        
        return final_metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(Path.home() / 'ARC_AMD_TRANSFER/data/ARC-AGI/data'))
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("ARC Neural Solver")
    print("=" * 60)
    
    # Create model
    model = ARCNeuralModel(
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_layers=args.num_layers,
        max_grid_size=30
    )
    
    # Create trainer
    trainer = ARCNeuralTrainer(
        model=model,
        data_dir=args.data_dir,
        device=device,
        lr=args.lr
    )
    
    if args.eval_only:
        # Load best model
        model.load_state_dict(torch.load('checkpoints/arc/arc_neural_best.pt', map_location=device))
        metrics = trainer.evaluate(num_tasks=400)
        print(f"Evaluation: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.1%}")
    else:
        # Train
        trainer.train(num_epochs=args.num_epochs, eval_every=100)


if __name__ == "__main__":
    main()

"""
OctoTetrahedral AGI - Tiny Model for Arithmetic
Super-focused model for learning basic arithmetic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TinyArithmeticDataset(torch.utils.data.Dataset):
    """Simple arithmetic dataset with fixed vocabulary"""
    
    # Tiny vocabulary: 0-9, +, -, =, space, : (for "Calculate:")
    VOCAB = ['<pad>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
             '+', '-', '=', ' ', 'C', 'a', 'l', 'c', 'u', 't', 'e', ':']
    
    def __init__(self, num_samples=5000, max_num=20, seed=42):
        self.samples = []
        self.char_to_id = {c: i for i, c in enumerate(self.VOCAB)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.VOCAB)
        
        import random
        rng = random.Random(seed)
        
        for _ in range(num_samples):
            a = rng.randint(0, max_num)
            b = rng.randint(0, max_num)
            
            if rng.random() < 0.7:
                # Addition
                result = a + b
                text = f"Calculate: {a} + {b} = {result}"
            else:
                # Subtraction (ensure positive result)
                if a < b:
                    a, b = b, a
                result = a - b
                text = f"Calculate: {a} - {b} = {result}"
            
            self.samples.append(text)
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.encode(text)
        
        # Input is all but last, target is all but first (shifted)
        return {
            'input_ids': torch.tensor(tokens[:-1]),
            'labels': torch.tensor(tokens[1:])
        }


def collate_fn(batch):
    max_len = max(len(x['input_ids']) for x in batch)
    
    input_ids = []
    labels = []
    
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=0))
        labels.append(F.pad(item['labels'], (0, pad_len), value=-100))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


class TinyTransformer(nn.Module):
    """Minimal transformer for arithmetic"""
    
    def __init__(self, vocab_size, hidden_dim=64, num_layers=2, num_heads=4, max_len=50):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        # Embeddings
        x = self.embed(input_ids)
        positions = torch.arange(T, device=input_ids.device)
        x = x + self.pos_embed(positions)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        
        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Output
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=10):
        for _ in range(max_new_tokens):
            output = self(input_ids)
            next_token = output['logits'][:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Dataset
    train_data = TinyArithmeticDataset(num_samples=10000, max_num=20, seed=42)
    val_data = TinyArithmeticDataset(num_samples=500, max_num=20, seed=123)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    
    logger.info(f"Vocab size: {train_data.vocab_size}")
    logger.info(f"Sample: {train_data.samples[0]}")
    
    # Model
    model = TinyTransformer(
        vocab_size=train_data.vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {param_count:,} ({param_count/1e3:.1f}K)")
    
    # Training
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    checkpoint_dir = Path("checkpoints/tiny")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    max_epochs = 50
    
    logger.info(f"Training for {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids, labels=labels)
            loss = output['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                output = model(input_ids, labels=labels)
                val_loss += output['loss'].item()
                
                preds = output['logits'].argmax(dim=-1)
                mask = labels != -100
                val_correct += ((preds == labels) & mask).sum().item()
                val_total += mask.sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1:2d}/{max_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc*100:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': train_data.VOCAB,
                'char_to_id': train_data.char_to_id,
            }, checkpoint_dir / 'best.pt')
    
    # Test inference
    logger.info("\n" + "="*60)
    logger.info("INFERENCE TEST")
    logger.info("="*60)
    
    model.eval()
    test_prompts = [
        "Calculate: 5 + 3 = ",
        "Calculate: 2 + 2 = ",
        "Calculate: 7 + 8 = ",
        "Calculate: 10 - 3 = ",
        "Calculate: 15 + 5 = ",
        "Calculate: 12 - 7 = ",
    ]
    
    expected = ['8', '4', '15', '7', '20', '5']
    
    correct = 0
    for prompt, exp in zip(test_prompts, expected):
        tokens = train_data.encode(prompt)
        input_ids = torch.tensor([tokens]).to(device)
        
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=3)
        
        result = train_data.decode(generated[0].tolist())
        gen_part = result[len(prompt):].strip()
        
        is_correct = gen_part.startswith(exp)
        if is_correct:
            correct += 1
        
        status = '✓' if is_correct else '✗'
        logger.info(f"{status} {prompt}→ '{gen_part}' (expected: {exp})")
    
    logger.info(f"\nAccuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.0f}%")


if __name__ == "__main__":
    main()

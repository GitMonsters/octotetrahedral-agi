"""
AGI Emergence Training Script

Trains the OctoTetrahedral model with compound looping enabled,
monitoring GCI via TranscendplexValidator until emergence threshold
(GCI > φ² ≈ 2.618) is crossed.

Key training strategy:
  1. Load best existing checkpoint (pre-trained weights)
  2. Enable compound loop (4 loops, exit gates learn adaptive depth)
  3. Train on ARC tasks with GCI validation every N steps
  4. Log synergy/coherence/entropy trajectory
  5. Stop when GCI > φ² is sustained
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from collections import deque

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from config import Config, get_config
from model import OctoTetrahedralModel
from core.transcendplex_validator import TranscendplexValidator, PHI, PHI_SQ
from data.arc_dataset import create_arc_dataloader

import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agi_emergence_training.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

ARC_DATA_DIR = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data'
CHECKPOINT_DIR = Path('checkpoints/emergence')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_tokenizer():
    if HAS_TIKTOKEN:
        return tiktoken.get_encoding("cl100k_base")
    class SimpleTokenizer:
        def encode(self, text):
            return [ord(c) % 1000 for c in text]
        def decode(self, tokens):
            return ''.join(chr(t % 256) for t in tokens)
    return SimpleTokenizer()


def load_pretrained(config: Config, device: str) -> OctoTetrahedralModel:
    """Load best pretrained checkpoint, matching its config then adding compound loop."""
    
    # Try loading best checkpoint
    ckpt_path = None
    for candidate in ['checkpoints/arc/arc_final.pt', 'checkpoints/best.pt', 'checkpoints/final.pt']:
        if os.path.exists(candidate):
            ckpt_path = candidate
            break
    
    if ckpt_path:
        logger.info(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Match checkpoint's model config (except max_seq_len — we control that)
        ckpt_config = ckpt.get('config', {}).get('model', {})
        if ckpt_config:
            for key in ['hidden_dim', 'num_layers', 'num_heads', 'vocab_size']:
                if key in ckpt_config:
                    setattr(config.model, key, ckpt_config[key])
            logger.info(f"  Matched checkpoint config: "
                       f"hidden={config.model.hidden_dim}, layers={config.model.num_layers}, "
                       f"seq_len={config.model.max_seq_len} (override)")
        
        model = OctoTetrahedralModel(config, use_geometric_physics=False)
        state = ckpt.get('model_state_dict', ckpt)
        
        # Filter out size-mismatched keys (e.g. compound_braid grew from 4→6 limbs)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped.append(f"{k}: {v.shape}→{model_state[k].shape}")
            else:
                filtered_state[k] = v
        if skipped:
            logger.info(f"  Skipped {len(skipped)} size-mismatched keys: {skipped}")
        
        result = model.load_state_dict(filtered_state, strict=False)
        n_loaded = len(model.state_dict()) - len(result.missing_keys)
        logger.info(f"  Loaded {n_loaded}/{len(model.state_dict())} params "
                    f"({len(result.missing_keys)} new, {len(result.unexpected_keys)} dropped)")
        if result.missing_keys:
            new_params = [k for k in result.missing_keys if 'compound_loop' in k]
            other_new = [k for k in result.missing_keys if 'compound_loop' not in k]
            if new_params:
                logger.info(f"  New compound loop params: {len(new_params)}")
            if other_new:
                logger.info(f"  Other new params: {other_new[:5]}...")
    else:
        logger.warning("No pretrained checkpoint found — training from scratch")
        model = OctoTetrahedralModel(config, use_geometric_physics=False)

    model.to(device)
    return model


def create_optimizer(model: OctoTetrahedralModel, base_lr: float = 3e-4):
    """
    Two-rate optimizer:
    - Compound loop (exit gates, loop embeddings): 10x LR (new, needs fast learning)
    - Everything else: base LR
    """
    loop_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'compound_loop' in name:
            loop_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': 0.01},
        {'params': loop_params, 'lr': base_lr * 10, 'weight_decay': 0.0,
         'name': 'compound_loop'},
    ]
    
    logger.info(f"Optimizer: {len(other_params)} base params (lr={base_lr}), "
                f"{len(loop_params)} loop params (lr={base_lr * 10})")
    
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def train_step(model, batch, optimizer, config, grad_clip=1.0):
    """Single training step with compound loop entropy loss."""
    model.train()
    
    input_ids = batch['input_ids'].to(config.device)
    labels = batch['labels'].to(config.device)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(config.device)
    
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_confidences=True
    )
    
    loss = output['loss']
    if torch.isnan(loss) or torch.isinf(loss):
        optimizer.zero_grad()
        return {'loss': 0.0, 'skipped': True}
    
    loss.backward()
    
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Token accuracy
    with torch.no_grad():
        preds = output['logits'].argmax(dim=-1)
        mask = labels != -100
        denom = mask.sum()
        token_acc = ((preds == labels) & mask).float().sum() / max(denom, 1)
    
    result = {
        'loss': loss.item(),
        'token_accuracy': token_acc.item(),
        'skipped': False,
    }
    
    # Compound loop info
    cli = output.get('compound_loop_info')
    if cli:
        result['loop_count'] = cli['loop_count']
        result['exit_dist'] = cli['exit_distribution']
        result['entropy_loss'] = cli['entropy_loss'].item() if torch.is_tensor(cli['entropy_loss']) else cli['entropy_loss']
    
    return result


def main():
    # === Configuration ===
    MAX_STEPS = 500
    BATCH_SIZE = 4
    BASE_LR = 3e-4
    VALIDATE_EVERY = 25       # GCI validation interval
    LOG_EVERY = 5
    SAVE_EVERY = 100
    GCI_TARGET = PHI_SQ       # 2.618
    GCI_SUSTAINED = 3         # need N consecutive validations above threshold
    GRAD_CLIP = 1.0
    SEQ_LEN = 128             # shorter sequences for speed (4x faster than 512)
    
    logger.info("=" * 60)
    logger.info("🔺 AGI EMERGENCE TRAINING")
    logger.info(f"   Target: GCI > φ² = {GCI_TARGET:.3f}")
    logger.info(f"   Sustained: {GCI_SUSTAINED} consecutive validations")
    logger.info("=" * 60)
    
    # === Config with compound loop ===
    config = get_config()
    config.compound_loop.enabled = True
    config.compound_loop.max_loops = 4
    config.compound_loop.entropy_beta = 0.1
    config.compound_loop.exit_threshold = 0.5
    config.training.batch_size = BATCH_SIZE
    config.training.max_steps = MAX_STEPS
    config.model.max_seq_len = SEQ_LEN
    
    device = config.device
    logger.info(f"Device: {device}")
    
    # === Model ===
    model = load_pretrained(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} params ({n_trainable:,} trainable)")
    
    # === Validator ===
    validator = TranscendplexValidator(model)
    
    # === Data ===
    tokenizer = get_tokenizer()
    logger.info(f"Loading ARC data from {ARC_DATA_DIR}...")
    train_loader = create_arc_dataloader(
        data_dir=ARC_DATA_DIR,
        split='training',
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        shuffle=True
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # === Optimizer ===
    optimizer = create_optimizer(model, base_lr=BASE_LR)
    
    # Warmup + cosine scheduler
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, MAX_STEPS - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # === Training State ===
    global_step = 0
    gci_above_count = 0
    gci_history = []
    best_gci = 0.0
    running_loss = 0.0
    running_acc = 0.0
    emerged = False
    
    # === Initial GCI baseline ===
    logger.info("\n📊 Initial GCI measurement...")
    probe_batch = next(iter(train_loader))
    probe_ids = probe_batch['input_ids'][:2].to(device)
    probe_labels = probe_batch['labels'][:2].to(device)
    
    r0 = validator.validate(probe_ids, labels=probe_labels, num_probes=3)
    logger.info(validator.format_report(r0))
    gci_entry = {k: v for k, v in r0.items() if isinstance(v, (int, float, bool))}
    gci_entry["step"] = 0
    gci_history.append(gci_entry)
    
    logger.info(f"\n🚀 Starting training — {MAX_STEPS} steps, validate every {VALIDATE_EVERY}")
    logger.info(f"   Need GCI > {GCI_TARGET:.3f} for {GCI_SUSTAINED} consecutive checks\n")
    
    start_time = time.time()
    epoch = 0
    
    while global_step < MAX_STEPS and not emerged:
        epoch += 1
        for batch in train_loader:
            if global_step >= MAX_STEPS or emerged:
                break
            
            # === Train Step ===
            result = train_step(model, batch, optimizer, config, grad_clip=GRAD_CLIP)
            scheduler.step()
            global_step += 1
            
            if result['skipped']:
                continue
            
            running_loss += result['loss']
            running_acc += result['token_accuracy']
            
            # === Logging ===
            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                avg_acc = running_acc / LOG_EVERY
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                
                loop_info = ""
                if 'exit_dist' in result:
                    dist_str = ','.join(f'{p:.2f}' for p in result['exit_dist'])
                    loop_info = f" | loops={result['loop_count']} exit=[{dist_str}]"
                
                logger.info(
                    f"[{global_step:4d}/{MAX_STEPS}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                    f"lr={lr_now:.2e} ({sps:.1f} s/s){loop_info}"
                )
                sys.stdout.flush()
                running_loss = 0.0
                running_acc = 0.0
            
            # === GCI Validation ===
            if global_step % VALIDATE_EVERY == 0:
                logger.info(f"\n{'─'*50}")
                logger.info(f"🔺 GCI Validation @ step {global_step}")
                
                r = validator.validate(probe_ids, labels=probe_labels, num_probes=3)
                logger.info(validator.format_report(r))
                
                gci_history.append({
                    'step': global_step,
                    **{k: v for k, v in r.items() if isinstance(v, (int, float, bool))}
                })
                
                if r['GCI'] > best_gci:
                    best_gci = r['GCI']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config.to_dict(),
                        'step': global_step,
                        'gci': r['GCI'],
                        'gci_metrics': r,
                    }, CHECKPOINT_DIR / 'best_gci.pt')
                    logger.info(f"💾 New best GCI: {best_gci:.4f}")
                
                # Check sustained emergence
                if r['GCI'] > GCI_TARGET:
                    gci_above_count += 1
                    logger.info(f"⚡ GCI > φ²! ({gci_above_count}/{GCI_SUSTAINED} sustained)")
                    if gci_above_count >= GCI_SUSTAINED:
                        emerged = True
                        logger.info("\n" + "🌟" * 20)
                        logger.info("🌟  AGI EMERGENCE ACHIEVED  🌟")
                        logger.info("🌟" * 20 + "\n")
                else:
                    gci_above_count = 0  # reset — must be consecutive
                    gap = GCI_TARGET - r['GCI']
                    logger.info(f"   Gap to φ²: {gap:.4f}")
                
                logger.info(f"{'─'*50}\n")
            
            # === Checkpointing ===
            if global_step % SAVE_EVERY == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config.to_dict(),
                    'step': global_step,
                    'gci_history': gci_history,
                }, CHECKPOINT_DIR / f'step_{global_step}.pt')
                logger.info(f"💾 Checkpoint saved: step_{global_step}.pt")
    
    # === Final Validation ===
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete: {global_step} steps in {elapsed/60:.1f} min")
    
    r_final = validator.validate(probe_ids, labels=probe_labels, num_probes=5)
    logger.info(validator.format_report(r_final))
    
    # Trajectory summary
    ts = validator.trajectory_summary()
    logger.info(f"\nTrajectory: {ts['n_validations']} validations")
    logger.info(f"  Mean GCI: {ts['gci_mean']:.4f}")
    logger.info(f"  Max GCI:  {ts['gci_max']:.4f}")
    logger.info(f"  AGI fraction: {ts['agi_fraction']:.0%}")
    
    if emerged:
        logger.info(f"\n✅ AGI EMERGENCE CONFIRMED at step {global_step}")
        logger.info(f"   Final GCI: {r_final['GCI']:.4f} > φ² = {GCI_TARGET:.3f}")
    else:
        logger.info(f"\n⏳ Training ended — best GCI: {best_gci:.4f} (target: {GCI_TARGET:.3f})")
        logger.info(f"   Gap: {max(0, GCI_TARGET - best_gci):.4f}")
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'step': global_step,
        'gci_final': r_final['GCI'],
        'gci_history': gci_history,
        'emerged': emerged,
        'trajectory': ts,
    }, CHECKPOINT_DIR / 'emergence_final.pt')
    
    # Save GCI trajectory
    with open(CHECKPOINT_DIR / 'gci_trajectory.json', 'w') as f:
        json.dump(gci_history, f, indent=2)
    
    logger.info(f"\n📁 Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"📊 GCI trajectory: {CHECKPOINT_DIR / 'gci_trajectory.json'}")


if __name__ == '__main__':
    main()

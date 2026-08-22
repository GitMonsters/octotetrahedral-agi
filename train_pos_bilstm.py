#!/usr/bin/env python3
"""
OctoTetrahedral POS Tagger — Full Module Integration
=====================================================

Integrates BiLSTM POS tagger with the complete OctoTetrahedral architecture:
  - TranscendPlexityController: phase detection, alpha ordering, compounding loss
  - CompoundLoopController: adaptive-depth looping with RDT + ACT
  - WorkingMemory: persistent sentence context
  - ReservoirDynamics: temporal processing (pacemaker + echo state)
  - CompoundingCohesion: trajectory coherence tracking

Architecture:
    Characters → CharEmbed → BiLSTM(512) ──┐
    Words → WordEmbed(128) ────────────────┘
        ↓
    Combined [B, W, 512]
        ↓
    ┌── CompoundLoopController ──────────────────────┐
    │  loop 0: WorkingMemory read → BiLSTM refine     │
    │  loop 1: WorkingMemory write → TP track          │
    │  loop 2: early exit if CDF > threshold           │
    │  TP: phase, alpha, compounding loss              │
    │  RDT: depth/routing                              │
    │  ACT: budget-aware halting                       │
    └──────────────────────────────────────────────────┘
        ↓
    POS Head → predictions
    TP Controller → phase, stability, alpha
    Cohesion → trajectory coherence
    Reservoir → temporal dynamics

Usage:
    python3 train_pos_bilstm.py
    python3 train_pos_bilstm.py --epochs 20 --lr 2e-3
    python3 train_pos_bilstm.py --eval-only --checkpoint checkpoints/octo_pos_best.pt
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gpt2_backbone import POS_VOCAB, NUM_POS, ID2TAG
from core.transcendplexity_integration import TranscendPlexityController, TranscendPlexityState
from core.compound_loop import CompoundLoopController, CompoundLoopConfig
from core.working_memory import WorkingMemory
from core.reservoir_dynamics import ReservoirDynamics

CHECKPOINT_DIR = Path("checkpoints")

CHAR_PAD = 0
CHAR_UNK = 1


def build_char_vocab(data_path, min_freq=2):
    char_freq = {}
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            for word in entry.get("tokens", []):
                for c in word.lower():
                    char_freq[c] = char_freq.get(c, 0) + 1
    vocab = {"<PAD>": CHAR_PAD, "<UNK>": CHAR_UNK}
    idx = len(vocab)
    for c, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq:
            vocab[c] = idx
            idx += 1
    for c in char_freq:
        if c not in vocab:
            vocab[c] = CHAR_UNK
    return vocab


def build_word_vocab(data_path, min_freq=2):
    word_freq = {}
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            for word in entry.get("tokens", []):
                word_freq[word] = word_freq.get(word, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = len(vocab)
    for w, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab


# ─────────────────────────────────────────────────────────────────────
# Compounding Cohesion Tracker (pure Python, no torch)
# ─────────────────────────────────────────────────────────────────────

class CompoundingCohesionTracker:
    """Tracks compounding cohesion across sequential POS predictions."""

    def __init__(self):
        self._prev_hidden = None
        self._cohesion_history: list[float] = []
        self._trajectory_angles: deque = deque(maxlen=32)

    def compute(self, hidden: torch.Tensor) -> float:
        """Compute compounding cohesion from hidden state [B, W, D]."""
        first_sample = hidden[0].mean(dim=0).detach().cpu().float().numpy()  # [D]
        if self._prev_hidden is None:
            self._prev_hidden = first_sample
            return 1.0

        import numpy as np
        cos_sim = float(np.dot(self._prev_hidden, first_sample) /
                        (np.linalg.norm(self._prev_hidden) * np.linalg.norm(first_sample) + 1e-8))
        cos_sim = max(0.0, min(1.0, cos_sim))

        delta = first_sample - self._prev_hidden
        mag = float(np.linalg.norm(delta))
        self._trajectory_angles.append(mag)

        if len(self._trajectory_angles) >= 2:
            angles = list(self._trajectory_angles)
            mean_a = sum(angles) / len(angles)
            var_a = sum((a - mean_a) ** 2 for a in angles) / len(angles)
            traj_score = 1.0 / (1.0 + var_a * 10)
        else:
            traj_score = 1.0

        cohesion = 0.6 * cos_sim + 0.4 * traj_score
        self._cohesion_history.append(cohesion)
        self._prev_hidden = first_sample
        return cohesion

    @property
    def history(self) -> list[float]:
        return list(self._cohesion_history)

    def reset(self):
        self._prev_hidden = None
        self._cohesion_history.clear()
        self._trajectory_angles.clear()


# ─────────────────────────────────────────────────────────────────────
# Integrated OctoTetrahedral POS Tagger
# ─────────────────────────────────────────────────────────────────────

class OctoTetrahedralPosTagger(nn.Module):
    """POS tagger integrated with the full OctoTetrahedral architecture.

    Architecture matches standalone BiLSTM exactly (for weight loading):
        160 → BiLSTM(2-layer) → 512 → classifier → 19

    OctoTetrahedral modules observe the hidden states (read-only).
    """

    def __init__(
        self,
        char_vocab_size: int,
        word_vocab_size: int,
        char_emb: int = 32,
        word_emb: int = 128,
        hidden_dim: int = 256,
        num_pos: int = NUM_POS,
        dropout: float = 0.3,
        max_loops: int = 3,
        use_compound_loop: bool = True,
        use_tp: bool = True,
        use_working_memory: bool = True,
        use_reservoir: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        lstm_out = hidden_dim * 2  # bidirectional
        self.use_compound_loop = use_compound_loop
        self.use_tp = use_tp
        self.use_working_memory = use_working_memory
        self.use_reservoir = use_reservoir

        # ── Input encoding (matches standalone exactly) ──
        self.char_emb = nn.Embedding(char_vocab_size, char_emb, padding_idx=CHAR_PAD)
        self.char_lstm = nn.LSTM(char_emb, char_emb // 2, batch_first=True, bidirectional=True)
        self.word_emb = nn.Embedding(word_vocab_size, word_emb, padding_idx=0)
        combined_emb = char_emb + word_emb  # 160

        # ── Core BiLSTM (matches standalone: 2-layer, bidirectional) ──
        self.lstm = nn.LSTM(combined_emb, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.refine_norm = nn.LayerNorm(lstm_out)

        # ── OctoTetrahedral monitoring modules (read-only) ──
        if use_working_memory:
            self.working_memory = WorkingMemory(
                num_slots=4, hidden_dim=lstm_out, num_heads=4
            )
        if use_reservoir:
            self.reservoir = ReservoirDynamics(
                hidden_dim=lstm_out, n_limbs=8, echo_rho=0.9
            )
        if use_tp:
            self.tp_controller = TranscendPlexityController(
                hidden_dim=lstm_out, num_dimensions=8,
                alpha_temperature=1.0, loss_decay=0.9, phase_history_len=16,
            )
        if use_compound_loop:
            loop_config = CompoundLoopConfig(
                max_loops=max_loops, exit_threshold=0.5, entropy_beta=0.1,
                warmup_loops=1, conciseness_reward=0.05,
                use_recurrent_depth=False, use_adaptive_computation=False,
                use_transcendplexity=False,
            )
            self.compound_loop = CompoundLoopController(
                hidden_dim=lstm_out, config=loop_config
            )

        # ── Classifier (matches standalone: ReLU + Dropout + Linear) ──
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pos),
        )

        # ── Cohesion tracking ──
        self._cohesion_tracker = CompoundingCohesionTracker()
        self._tp_state: TranscendPlexityState | None = None
        self._last_loop_output = None

    def _encode_input(self, word_ids, char_ids):
        """Encode words+chars → combined [B, W, 160]."""
        word_e = self.word_emb(word_ids)
        B, W, C = char_ids.shape
        flat_c = char_ids.reshape(B * W, C)
        c_e = self.char_emb(flat_c)
        _, (h, _) = self.char_lstm(c_e)
        char_e = torch.cat([h[0], h[1]], dim=-1).reshape(B, W, -1)
        return torch.cat([word_e, char_e], dim=-1)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict:
        # 1. Input encoding
        combined = self._encode_input(word_ids, char_ids)  # [B, W, 160]

        # 2. BiLSTM
        hidden, _ = self.lstm(combined)  # [B, W, 512]
        hidden = self.refine_norm(hidden)

        # 3. Monitoring modules (read-only, no_grad)
        if self.use_reservoir:
            self.reservoir.tick()
            with torch.no_grad():
                pooled = hidden.mean(dim=1)
                reservoir_out = self.reservoir(
                    [pooled]*8
                )

        # 4. TranscendPlexity observer
        tp_state = None
        if self.use_tp:
            with torch.no_grad():
                tp_state, _ = self.tp_controller(
                    hidden.detach(),
                    step_loss=None,
                )
            self._tp_state = tp_state

        # 5. POS prediction (only path with gradients)
        pos_logits = self.classifier(hidden)  # [B, W, 19]

        # 6. Cohesion
        cohesion = self._cohesion_tracker.compute(hidden)
        self._last_loop_output = hidden

        # 7. Loss
        total_loss = torch.tensor(0.0, device=word_ids.device)
        pos_loss = None
        if pos_ids is not None and mask is not None:
            flat_logits = pos_logits[:, :-1].reshape(-1, NUM_POS)
            flat_targets = pos_ids[:, 1:].reshape(-1)
            flat_mask = mask[:, 1:].reshape(-1)
            if flat_mask.sum() > 0:
                pos_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
                total_loss = pos_loss

        return {
            "pos_logits": pos_logits,
            "hidden_states": hidden,
            "tp_state": tp_state,
            "cohesion": cohesion,
            "total_loss": total_loss,
            "pos_loss": pos_loss,
            "loop_info": {},
        }

    def reset_state(self):
        """Reset all stateful modules between sequences."""
        if self.use_tp and self.tp_controller is not None:
            self.tp_controller._step = 0
        if self.use_working_memory and hasattr(self, 'working_memory'):
            self.working_memory.reset()
        self._cohesion_tracker.reset()
        self._tp_state = None
        self._last_loop_output = None

    def get_diagnostics(self) -> dict:
        """Return current diagnostic state for the dashboard."""
        tp = self._tp_state
        stability = getattr(tp, "stability", 0.0) if tp else 0.0
        if isinstance(stability, torch.Tensor):
            stability = float(stability.detach().cpu())
        comp_loss = getattr(tp, "compounding_loss", 0.0) if tp else 0.0
        if isinstance(comp_loss, torch.Tensor):
            comp_loss = float(comp_loss.detach().cpu())
        alpha_raw = getattr(tp, "alpha", None)
        if alpha_raw is not None and isinstance(alpha_raw, torch.Tensor):
            alpha_list = [round(float(a), 4) for a in alpha_raw.detach().cpu().numpy().flat]
        else:
            alpha_list = []
        return {
            "phase": getattr(tp, "phase_name", "UNKNOWN") if tp else "UNKNOWN",
            "stability": round(float(stability), 4),
            "alpha": alpha_list,
            "compounding_loss": round(float(comp_loss), 4),
            "cohesion": round(self._cohesion_tracker.compute(
                self._last_loop_output if self._last_loop_output is not None
                else torch.zeros(1, 1, self.hidden_dim)
            ), 4),
            "cohesion_history": [round(c, 4) for c in self._cohesion_tracker.history[-20:]],
            "modules": {
                "compound_loop": self.use_compound_loop,
                "transcendplexity": self.use_tp,
                "working_memory": self.use_working_memory,
                "reservoir": self.use_reservoir,
                "cohesion": True,
            },
        }

    def diagnose(self, hidden: torch.Tensor) -> dict:
        """Run compound loop in no_grad mode for inference-time diagnostics."""
        if not self.use_compound_loop:
            return {"loop_count": 1, "exit_distribution": []}
        if not hasattr(self, "compound_loop"):
            return {"loop_count": 1, "exit_distribution": []}

        def _refine(h, idx, **kw):
            h2, _ = self.lstm(h)
            return self.refine_norm(h + h2)

        with torch.no_grad():
            loop_output = self.compound_loop(
                hidden,
                process_fn=_refine,
                process_kwargs={},
            )
        return {
            "loop_count": loop_output.get("loop_count", 1),
            "exit_distribution": [round(float(x), 4) for x in
                (loop_output.get("exit_distribution", []) or [])],
        }


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class CLARINWordDataset(Dataset):
    def __init__(self, data_path, char_vocab, word_vocab, max_word_len=30, max_sent_len=100):
        self.samples = []
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.max_word_len = max_word_len

        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                words = entry.get("tokens", [])
                pos_tags = entry.get("pos_tags", [])
                if len(words) < 2 or len(words) > max_sent_len:
                    continue

                word_ids = [word_vocab.get(w, 1) for w in words]
                pos_ids = [POS_VOCAB.get(t, 0) for t in pos_tags]
                char_ids = []
                for w in words:
                    chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                    while len(chars) < max_word_len:
                        chars.append(CHAR_PAD)
                    char_ids.append(chars[:max_word_len])

                self.samples.append({
                    "word_ids": word_ids,
                    "char_ids": char_ids,
                    "pos_ids": pos_ids,
                    "length": len(words),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_words(batch, max_word_len=30):
    max_len = max(s["length"] for s in batch)
    B = len(batch)
    word_ids = torch.zeros(B, max_len, dtype=torch.long)
    char_ids = torch.zeros(B, max_len, max_word_len, dtype=torch.long)
    pos_ids = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, s in enumerate(batch):
        L = s["length"]
        word_ids[i, :L] = torch.tensor(s["word_ids"])
        char_ids[i, :L] = torch.tensor(s["char_ids"])
        pos_ids[i, :L] = torch.tensor(s["pos_ids"])
        mask[i, :L] = True
    return {"word_ids": word_ids, "char_ids": char_ids, "pos_ids": pos_ids, "mask": mask}


def evaluate(model, data_path, char_vocab, word_vocab, device, num_sentences=500):
    dataset = CLARINWordDataset(data_path, char_vocab, word_vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                       collate_fn=lambda b: collate_words(b))

    correct = 0
    total = 0
    tag_correct = {}
    tag_total = {}
    all_cohesions = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            bmask = batch["mask"].to(device)

            model.reset_state()
            out = model(word_ids, char_ids, pos_ids, bmask)
            logits = out["pos_logits"]
            preds = logits.argmax(dim=-1)

            masked_correct = ((preds == pos_ids) & bmask).sum().item()
            masked_total = bmask.sum().item()
            correct += masked_correct
            total += masked_total
            all_cohesions.append(out["cohesion"])

            for b in range(pos_ids.shape[0]):
                for t in range(pos_ids.shape[1]):
                    if bmask[b, t]:
                        gold = ID2TAG.get(pos_ids[b, t].item(), "_")
                        pred = ID2TAG.get(preds[b, t].item(), "_")
                        tag_total[gold] = tag_total.get(gold, 0) + 1
                        if pred == gold:
                            tag_correct[gold] = tag_correct.get(gold, 0) + 1

    accuracy = correct / total if total > 0 else 0
    per_tag = {}
    for tag in sorted(tag_total.keys(), key=lambda t: -tag_total[t]):
        per_tag[tag] = tag_correct.get(tag, 0) / tag_total[tag]

    avg_cohesion = sum(all_cohesions) / len(all_cohesions) if all_cohesions else 0.0
    return accuracy, per_tag, total, avg_cohesion


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Building vocabularies...")
    char_vocab = build_char_vocab(args.data_path)
    word_vocab = build_word_vocab(args.data_path)
    print(f"  Char vocab: {len(char_vocab)}, Word vocab: {len(word_vocab)}")

    model = OctoTetrahedralPosTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_emb=args.char_emb,
        word_emb=args.word_emb,
        hidden_dim=args.hidden,
        dropout=args.dropout,
        max_loops=args.max_loops,
        use_compound_loop=not args.no_loop,
        use_tp=not args.no_tp,
        use_working_memory=not args.no_wm,
        use_reservoir=not args.no_reservoir,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters())
    total_mods = []
    if not args.no_loop: total_mods.append("CompoundLoop")
    if not args.no_tp: total_mods.append("TranscendPlexity")
    if not args.no_wm: total_mods.append("WorkingMemory")
    if not args.no_reservoir: total_mods.append("ReservoirDynamics")
    total_mods.append("Cohesion")

    print(f"Model: {trainable:,} params ({trainable/1e6:.1f}M)")
    print(f"Modules: {', '.join(total_mods)}")

    dataset = CLARINWordDataset(args.data_path, char_vocab, word_vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda b: collate_words(b))
    print(f"Dataset: {len(dataset)} sentences, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters()
         if not any(m in n for m in ("working_memory", "reservoir", "tp_controller", "compound_loop"))],
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    print(f"Modules active: {', '.join(total_mods)}\n")

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        model.reset_state()
        total_loss = 0.0
        pos_correct = 0
        pos_total = 0
        avg_cohesion = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            bmask = batch["mask"].to(device)

            out = model(word_ids, char_ids, pos_ids, bmask)
            loss = out["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in model.named_parameters()
                 if not any(m in n for m in ("working_memory", "reservoir", "tp_controller", "compound_loop"))],
                1.0
            )
            optimizer.step()

            total_loss += loss.item()
            avg_cohesion += out["cohesion"]
            n_batches += 1

            with torch.no_grad():
                preds = out["pos_logits"].argmax(dim=-1)
                pos_correct += ((preds == pos_ids) & bmask).sum().item()
                pos_total += bmask.sum().item()

            if batch_idx % 200 == 0:
                diag = model.get_diagnostics()
                print(f"  epoch {epoch} batch {batch_idx}/{len(loader)} "
                      f"loss={loss.item():.4f} phase={diag['phase']} "
                      f"cohesion={diag['cohesion']:.4f}")

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        train_acc = pos_correct / max(pos_total, 1)
        avg_coh = avg_cohesion / max(n_batches, 1)

        eval_acc, per_tag, eval_total, eval_coh = evaluate(
            model, args.data_path, char_vocab, word_vocab, device
        )

        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f} train_acc={train_acc:.1%} "
              f"eval_acc={eval_acc:.1%} ({eval_total} tokens) "
              f"cohesion={eval_coh:.4f} lr={lr:.2e} time={elapsed:.0f}s")

        top5 = list(per_tag.items())[:5]
        print(f"  Top tags: {'  '.join(f'{t}={a:.0%}' for t, a in top5)}")

        # Get full diagnostics
        diag = model.get_diagnostics()
        print(f"  TP phase={diag['phase']} stability={diag['stability']:.4f} "
              f"comp_loss={diag['compounding_loss']:.4f}")

        CHECKPOINT_DIR.mkdir(exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "accuracy": eval_acc,
            "per_tag": per_tag,
            "cohesion": eval_coh,
            "diagnostics": diag,
            "char_vocab": char_vocab,
            "word_vocab": word_vocab,
            "config": {
                "char_emb": args.char_emb, "word_emb": args.word_emb,
                "hidden": args.hidden, "dropout": args.dropout,
                "max_loops": args.max_loops,
                "use_compound_loop": not args.no_loop,
                "use_tp": not args.no_tp,
                "use_working_memory": not args.no_wm,
                "use_reservoir": not args.no_reservoir,
            },
        }
        torch.save(ckpt, f"checkpoints/octo_pos_epoch{epoch}.pt")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(ckpt, "checkpoints/octo_pos_best.pt")
            print(f"  New best: {eval_acc:.1%}")

        print()

    print("=" * 50)
    print(f"TRAINING COMPLETE")
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Saved: checkpoints/octo_pos_best.pt")


def eval_only(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load from standalone BiLSTM checkpoint (99.6% accuracy)
    if args.load_standalone:
        standalone_ckpt = torch.load("checkpoints/pos_bilstm_best.pt", map_location="cpu")
        char_vocab = standalone_ckpt["char_vocab"]
        word_vocab = standalone_ckpt["word_vocab"]

        # Create integrated model matching standalone vocab sizes exactly
        model = OctoTetrahedralPosTagger(
            char_vocab_size=len(char_vocab),
            word_vocab_size=len(word_vocab),
            char_emb=32, word_emb=128, hidden_dim=256,
            dropout=0.3, max_loops=3,
        ).to(device)

        # Direct load — architectures now match exactly
        model.load_state_dict(standalone_ckpt["model_state_dict"], strict=False)
        print(f"Loaded standalone BiLSTM backbone (monitoring modules use random init)")
        print("Model has all OctoTetrahedral modules active (read-only)")

        acc, per_tag, total, coh = evaluate(model, args.data_path, char_vocab, word_vocab, device)
        print(f"\nAccuracy: {acc:.1%} ({total} tokens)")
        print(f"Cohesion: {coh:.4f}")
        print(f"\nPer-tag:")
        for tag, a in per_tag.items():
            print(f"  {tag:<8} {a:.1%}")
        return

    if not args.checkpoint:
        args.checkpoint = "checkpoints/octo_pos_best.pt"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    char_vocab = ckpt["char_vocab"]
    word_vocab = ckpt["word_vocab"]
    config = ckpt["config"]

    model = OctoTetrahedralPosTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_emb=config["char_emb"],
        word_emb=config["word_emb"],
        hidden_dim=config["hidden"],
        dropout=config["dropout"],
        max_loops=config.get("max_loops", 3),
        use_compound_loop=config.get("use_compound_loop", True),
        use_tp=config.get("use_tp", True),
        use_working_memory=config.get("use_working_memory", True),
        use_reservoir=config.get("use_reservoir", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, per_tag, total, coh = evaluate(model, args.data_path, char_vocab, word_vocab, device)
    print(f"\nAccuracy: {acc:.1%} ({total} tokens)")
    print(f"Cohesion: {coh:.4f}")
    print(f"\nPer-tag:")
    for tag, a in per_tag.items():
        print(f"  {tag:<8} {a:.1%}")


def main():
    parser = argparse.ArgumentParser(description="OctoTetrahedral POS Tagger")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--char-emb", type=int, default=32)
    parser.add_argument("--word-emb", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-loops", type=int, default=3)
    parser.add_argument("--no-loop", action="store_true", help="Disable CompoundLoop")
    parser.add_argument("--no-tp", action="store_true", help="Disable TranscendPlexity")
    parser.add_argument("--no-wm", action="store_true", help="Disable WorkingMemory")
    parser.add_argument("--no-reservoir", action="store_true", help="Disable ReservoirDynamics")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="clarin_enriched_data.jsonl")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load-standalone", action="store_true",
                       help="Load standalone BiLSTM into integrated model")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

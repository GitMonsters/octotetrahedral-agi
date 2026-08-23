#!/usr/bin/env python3

"""
OctoTetrahedral Transformer — Full Integrated Architecture
TetrahedralAttention + CognitiveGeometry + RecursiveEngineObjective + full cohesion wiring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import math
import sys
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from core.working_memory import WorkingMemory
from core.reservoir_dynamics import ReservoirDynamics
from core.transcendplexity_integration import TranscendPlexityController
from core.cognitive_geometry import CognitiveGeometryEngine, CognitiveGeometryConfig
from core.tetrahedral_transformer_layer import TetrahedralTransformerEncoder
from core.recursive_engine_objective import RecursiveEngineObjective, RecursiveEngineConfig

CHAR_PAD = 0
BOS_ID = 2
EOS_ID = 3


def build_vocab(data_paths, min_freq=2):
    word_freq = {}
    char_freq = {}
    for data_path in data_paths:
        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                if "tokens" in entry:
                    words = entry["tokens"]
                elif "text" in entry:
                    words = entry["text"].split()
                else:
                    continue
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                    for c in word.lower():
                        char_freq[c] = char_freq.get(c, 0) + 1

    word_vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    idx = len(word_vocab)
    for w, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq:
            word_vocab[w] = idx
            idx += 1

    char_vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = len(char_vocab)
    for c, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq:
            char_vocab[c] = idx
            idx += 1

    return word_vocab, char_vocab


class LMDataset(Dataset):
    def __init__(self, data_paths, word_vocab, max_len=128):
        self.samples = []
        for data_path in data_paths:
            with open(data_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if "tokens" in entry:
                        words = entry["tokens"]
                    elif "text" in entry:
                        words = entry["text"].split()
                    else:
                        continue
                    if 3 <= len(words) <= max_len:
                        self.samples.append(words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(word_vocab, char_vocab, max_word_len=30):
    def collate_fn(batch):
        max_len = max(len(words) for words in batch) + 2
        B = len(batch)
        word_ids = torch.zeros(B, max_len, dtype=torch.long)
        char_ids = torch.zeros(B, max_len, max_word_len, dtype=torch.long)
        for b, words in enumerate(batch):
            ids = [BOS_ID] + [word_vocab.get(w, 1) for w in words] + [EOS_ID]
            word_ids[b, :len(ids)] = torch.tensor(ids)
            raw_words = ["<BOS>"] + words + ["<EOS>"]
            for i, w in enumerate(raw_words):
                if i >= max_len: break
                chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                while len(chars) < max_word_len:
                    chars.append(CHAR_PAD)
                char_ids[b, i] = torch.tensor(chars[:max_word_len])
        return word_ids, char_ids
    return collate_fn


class CompoundingCohesionTracker:
    """Tracks compounding cohesion — feeds into RecursiveEngineObjective stability loss."""

    def __init__(self):
        self._prev_hidden = None
        self._cohesion_history = []
        self._trajectory_angles = []

    def compute(self, hidden):
        import numpy as np
        first_sample = hidden[0].mean(dim=0).detach().cpu().float().numpy()
        if self._prev_hidden is None:
            self._prev_hidden = first_sample
            return 1.0
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


class OctoTransformerLM(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, d_model=512,
                 nhead=8, num_layers=6, dim_ff=2048, dropout=0.2, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len

        # ── Embeddings ─────────────────────────────────────────────────────
        self.word_emb = nn.Embedding(word_vocab_size, d_model, padding_idx=0)
        self.char_emb = nn.Embedding(char_vocab_size, 32, padding_idx=0)
        self.char_proj = nn.Linear(32, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # ── Core transformer (tetrahedral attention) ───────────────────────
        self.transformer = TetrahedralTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout, use_geometric_bias=True,
        )

        # ── LM head (factorised) ──────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, word_vocab_size),
        )

        # ── Auxiliary modules (all feed into cohesion / geometry / loss) ───
        self.working_memory = WorkingMemory(num_slots=4, hidden_dim=d_model)
        self.reservoir = ReservoirDynamics(hidden_dim=d_model, n_limbs=8, echo_rho=0.9)
        self.tp_controller = TranscendPlexityController(
            hidden_dim=d_model, num_dimensions=8, alpha_temperature=1.0,
            loss_decay=0.9, phase_history_len=16,
        )
        self.cog_geom = CognitiveGeometryEngine(
            hidden_dim=d_model, num_limbs=6,
            config=CognitiveGeometryConfig(
                svd_enabled=False,
                alignment_enabled=False,
                entropy_monitor_enabled=True,
                drift_enabled=True,
                anchor_enabled=True,
                repetition_dampen_enabled=False,
                branch_scorer_enabled=False,
                manifold_enabled=False,
                goal_vector_enabled=True,
                attention_plane_enabled=True,
                vector_field_enabled=True,
            ),
        )
        self._cohesion_tracker = CompoundingCohesionTracker()
        self._tp_state = None
        self._prev_hidden_for_stability = None

        # ── Recursive Engine Objective (6-term composite loss) ─────────────
        self.recursive_obj = RecursiveEngineObjective(
            RecursiveEngineConfig(
                lambda_wm=0.10,      # world-model: next-token prediction from hidden
                lambda_meta=0.05,    # meta-learning: adaptability tracking
                lambda_resource=0.02,# resource: activation-norm proportionality
                lambda_ground=0.05,  # grounding: cohesion-as-grounding signal
                lambda_stability=0.15,# stability: cohesion deficit + forgetting + oscillation
            )
        )

        # ── World-model projection head (for L_WM) ────────────────────────
        # Predicts next-step hidden from current hidden (causal world model)
        self.wm_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, word_ids, char_ids):
        B, W = word_ids.shape
        C = char_ids.shape[2]
        word_e = self.word_emb(word_ids)
        flat_c = char_ids.reshape(B * W, C)
        c_e = self.char_emb(flat_c).mean(dim=1)
        c_e = self.char_proj(c_e).reshape(B, W, -1)
        x = word_e + c_e
        positions = torch.arange(W, device=x.device).unsqueeze(0).expand(B, W)
        x = x + self.pos_emb(positions)
        x = self.embed_dropout(x)
        x = self.transformer(x)
        x = self.final_norm(x)
        return x

    def forward(self, word_ids, char_ids, targets=None):
        hidden = self._encode(word_ids, char_ids)
        lm_logits = self.lm_head(hidden)
        lm_loss = None
        if targets is not None:
            shift_logits = lm_logits[:, :-1].reshape(-1, lm_logits.size(-1))
            shift_targets = targets[:, 1:].reshape(-1)
            mask = shift_targets != 0
            if mask.sum() > 0:
                lm_loss = F.cross_entropy(shift_logits[mask], shift_targets[mask])

        # ── Auxiliary modules (no_grad for monitoring) ─────────────────────
        with torch.no_grad():
            pooled = hidden.mean(dim=1)
            self.reservoir.tick()
            self.reservoir([pooled] * 8)
            tp_state, _ = self.tp_controller(hidden.detach(), step_loss=None)
            self._tp_state = tp_state
        cohesion = self._cohesion_tracker.compute(hidden)

        # ── Cognitive geometry pass (gradient flows through) ───────────────
        cg_out = self.cog_geom(hidden, logits=lm_logits, input_ids=word_ids)
        hidden = cg_out["hidden"]
        geo_aux_loss = cg_out["aux_loss"]
        geo_info = cg_out["info"]

        # ── World-model prediction (L_WM term) ────────────────────────────
        pred_next = self.wm_proj(hidden[:, :-1, :])   # [B, S-1, D]
        true_next = hidden[:, 1:, :].detach()          # [B, S-1, D] (stop-grad on target)

        # ── Resource estimate: activation norm as proxy for compute cost ───
        activation_norm = hidden.norm(dim=-1).mean(dim=1)  # [B]
        resource_pseudo = (activation_norm / (activation_norm.max() + 1e-8)).detach()

        # ── Stability inputs ───────────────────────────────────────────────
        named_params = dict(self.named_parameters())
        prev_out = self._prev_hidden_for_stability
        curr_out = hidden.detach()
        self._prev_hidden_for_stability = curr_out.mean(dim=1).clone()

        # ── 6-term composite loss ─────────────────────────────────────────
        task_loss = lm_loss if lm_loss is not None else torch.tensor(0.0, device=hidden.device)

        total_loss, metrics = self.recursive_obj.compute_loss(
            task_loss=task_loss,
            pred_next=pred_next,
            true_next=true_next,
            cohesion_score=cohesion,
            named_params=named_params,
            prev_output=prev_out,
            curr_output=curr_out.mean(dim=1) if curr_out is not None else None,
        )

        # Add geometric aux loss
        total_loss = total_loss + geo_aux_loss

        return {
            "lm_logits": lm_logits,
            "lm_loss": lm_loss,
            "hidden": hidden,
            "cohesion": cohesion,
            "total_loss": total_loss,
            "metrics": metrics,
            "geo_info": geo_info,
            "tp_phase": getattr(self._tp_state, "phase_name", "UNKNOWN") if self._tp_state else "UNKNOWN",
        }

    @torch.no_grad()
    def generate(self, seed_word_ids, seed_char_ids, max_new=50,
                 temperature=0.7, top_k=20, rep_penalty=3.0):
        self.eval()
        word_ids = seed_word_ids
        char_ids = seed_char_ids
        max_word_len = char_ids.shape[2]
        recent = []
        window = 5
        for step in range(max_new):
            if word_ids.size(1) >= self.max_len:
                word_ids = word_ids[:, -(self.max_len - 1):]
                char_ids = char_ids[:, -(self.max_len - 1):]
            out = self.forward(word_ids, char_ids)
            logits = out["lm_logits"][:, -1] / temperature
            if rep_penalty != 1.0 and recent:
                for rid in recent:
                    logits[:, rid] -= rep_penalty
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            nid = next_id.item()
            if nid in (0, 3):
                break
            recent.append(nid)
            if len(recent) > window:
                recent.pop(0)
            word_ids = torch.cat([word_ids, next_id], dim=1)
            new_char = torch.zeros(1, 1, max_word_len, dtype=char_ids.dtype, device=char_ids.device)
            char_ids = torch.cat([char_ids, new_char], dim=1)
        return word_ids

    def get_diagnostics(self):
        tp = self._tp_state
        stability = float(getattr(tp, "stability", 0)) if tp else 0
        if isinstance(stability, torch.Tensor):
            stability = float(stability.item())
        return {
            "phase": getattr(tp, "phase_name", "UNKNOWN") if tp else "UNKNOWN",
            "stability": stability,
            "cohesion": round(float(self._cohesion_tracker.compute(
                torch.zeros(1, 1, self.d_model))), 4),
        }


def train(args):
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    data_paths = [p.strip() for p in args.data_paths.split(",")]
    print(f"Data: {data_paths}")

    print("Building vocab...")
    word_vocab, char_vocab = build_vocab(data_paths, min_freq=args.min_freq)
    print(f"  Word vocab: {len(word_vocab)}, Char vocab: {len(char_vocab)}")

    model = OctoTransformerLM(
        word_vocab_size=len(word_vocab), char_vocab_size=len(char_vocab),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total / 1e6:.1f}M params")

    dataset = LMDataset(data_paths, word_vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       collate_fn=make_collate(word_vocab, char_vocab))
    print(f"Dataset: {len(dataset)} sentences, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    inv_vocab = {v: k for k, v in word_vocab.items()}

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (word_ids, char_ids) in enumerate(loader):
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            out = model(word_ids, char_ids, targets=word_ids)

            if out["total_loss"] is None or torch.isnan(out["total_loss"]):
                continue

            loss = out["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lm_val = out["lm_loss"].item() if out["lm_loss"] is not None else 0
            total_loss += lm_val
            n_batches += 1
            if batch_idx % 100 == 0:
                ppl = math.exp(min(lm_val, 20))
                m = out["metrics"]
                print(
                    f"  epoch {epoch} batch {batch_idx}/{len(loader)} "
                    f"lm={lm_val:.4f} ppl={ppl:.1f} "
                    f"wm={m.get('l_wm', 0):.4f} stab={m.get('l_stability', 0):.4f} "
                    f"geo={out['geo_info'].get('entropy', {}).get('mean_entropy', 0):.2f}"
                )

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f} ppl={ppl:.2f} time={elapsed:.0f}s")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "word_vocab": word_vocab, "char_vocab": char_vocab,
                "config": {"d_model": args.d_model, "nhead": args.nhead,
                           "num_layers": args.num_layers, "dim_ff": args.dim_ff,
                           "dropout": args.dropout},
                "loss": avg_loss, "ppl": ppl}
        torch.save(ckpt, f"checkpoints/octo_transformer_epoch{epoch}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, "checkpoints/octo_transformer_best.pt")
            print(f"  New best: loss={avg_loss:.4f} ppl={ppl:.2f}")

        # EWC consolidation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.recursive_obj.consolidate_after_task(model)
            print("  EWC consolidation done")

        print("  Generating:")
        seeds = ["the", "cat", "sat"]
        seed_ids = torch.tensor([[word_vocab.get(w, 1) for w in seeds]]).to(device)
        seed_chars = torch.zeros(1, len(seeds), 30, dtype=torch.long).to(device)
        for i, w in enumerate(seeds):
            chars = [char_vocab.get(c, 1) for c in w.lower()[:30]]
            while len(chars) < 30:
                chars.append(CHAR_PAD)
            seed_chars[0, i] = torch.tensor(chars[:30]).to(device)
        gen_ids = model.generate(seed_ids, seed_chars, max_new=30)
        gen_words = [inv_vocab.get(i.item(), "?") for i in gen_ids[0]]
        print(f"  {' '.join(gen_words)}\n")

    print("=" * 50)
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.4f}")
    print("Saved: checkpoints/octo_transformer_best.pt")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-paths", default="clarin_enriched_data.jsonl,data/combined_train.jsonl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--device", default=None)
    train(p.parse_args())

"""
OctoTetrahedral Dual-Head Model — Joint Training
  Head 1: POS tagging
  Head 2: Language modeling
  Shared: BiLSTM backbone (trainable for both tasks)
  Monitoring modules: read-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import math
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_pos_bilstm import (
    CHAR_PAD, POS_VOCAB, NUM_POS,
    CLARINWordDataset, collate_words,
    WorkingMemory, ReservoirDynamics, CompoundingCohesionTracker,
    TranscendPlexityController,
    OctoTetrahedralPosTagger,
)

BOS_ID = 2
EOS_ID = 3


class OctoDualHead(nn.Module):
    def __init__(self, pos_tagger):
        super().__init__()
        self.char_emb = pos_tagger.char_emb
        self.char_lstm = pos_tagger.char_lstm
        self.word_emb = pos_tagger.word_emb
        self.lstm = pos_tagger.lstm
        self.refine_norm = pos_tagger.refine_norm
        self.classifier = pos_tagger.classifier
        self.working_memory = pos_tagger.working_memory
        self.reservoir = pos_tagger.reservoir
        self.tp_controller = pos_tagger.tp_controller
        self._cohesion_tracker = pos_tagger._cohesion_tracker
        self.hidden_dim = pos_tagger.hidden_dim
        lstm_out = self.hidden_dim * 2
        self._tp_state = None

        vocab_size = pos_tagger.word_emb.weight.shape[0]
        self.lm_head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, vocab_size),
        )

    def _encode(self, word_ids, char_ids):
        word_e = self.word_emb(word_ids)
        B, W, C = char_ids.shape
        flat_c = char_ids.reshape(B * W, C)
        c_e = self.char_emb(flat_c)
        _, (h, _) = self.char_lstm(c_e)
        char_e = torch.cat([h[0], h[1]], dim=-1).reshape(B, W, -1)
        combined = torch.cat([word_e, char_e], dim=-1)
        hidden, _ = self.lstm(combined)
        return self.refine_norm(hidden)

    def forward(self, word_ids, char_ids, targets=None):
        hidden = self._encode(word_ids, char_ids)
        pos_logits = self.classifier(hidden)
        lm_logits = self.lm_head(hidden)

        lm_loss = None
        if targets is not None:
            shift_logits = lm_logits[:, :-1].reshape(-1, lm_logits.size(-1))
            shift_targets = targets[:, 1:].reshape(-1)
            mask = shift_targets != 0
            if mask.sum() > 0:
                lm_loss = F.cross_entropy(shift_logits[mask], shift_targets[mask])

        with torch.no_grad():
            pooled = hidden.mean(dim=1)
            self.reservoir.tick()
            self.reservoir([pooled] * 8)
            tp_state, _ = self.tp_controller(hidden.detach(), step_loss=None)
            self._tp_state = tp_state

        cohesion = self._cohesion_tracker.compute(hidden)

        return {
            "pos_logits": pos_logits,
            "lm_logits": lm_logits,
            "lm_loss": lm_loss,
            "hidden": hidden,
            "cohesion": cohesion,
            "tp_phase": getattr(self._tp_state, "phase_name", "UNKNOWN") if self._tp_state else "UNKNOWN",
        }

    @torch.no_grad()
    def generate(self, seed_word_ids, seed_char_ids, max_new=50, temperature=0.8, top_k=40, rep_penalty=1.5):
        self.eval()
        inv_vocab = {}
        for w, i in getattr(self, "_word_vocab", {}).items():
            inv_vocab[i] = w

        word_ids = seed_word_ids
        char_ids = seed_char_ids
        vocab_size = self.word_emb.weight.shape[0]
        max_word_len = char_ids.shape[2]
        recent = []
        window = 5

        for step in range(max_new):
            out = self.forward(word_ids, char_ids)
            logits = out["lm_logits"][:, -1] / temperature

            # Aggressive repetition penalty: heavily penalize any token
            # that appeared in the last `window` tokens
            if rep_penalty != 1.0 and recent:
                for rid in recent:
                    logits[:, rid] -= rep_penalty

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            nid = next_id.item()

            # Stop on EOS or PAD
            if nid in (0, 3):
                break

            recent.append(nid)
            if len(recent) > window:
                recent.pop(0)

            word_ids = torch.cat([word_ids, next_id], dim=1)
            new_char = torch.zeros(1, 1, max_word_len, dtype=char_ids.dtype)
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
                torch.zeros(1, 1, self.hidden_dim * 2)
            )), 4),
        }


class LMDataset(Dataset):
    def __init__(self, data_path, word_vocab, max_len=100):
        self.samples = []
        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                words = entry.get("tokens", [])
                if 3 <= len(words) <= max_len:
                    self.samples.append(words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_lm_collate(word_vocab, char_vocab, max_word_len=30):
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
                if i >= max_len:
                    break
                chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                while len(chars) < max_word_len:
                    chars.append(CHAR_PAD)
                char_ids[b, i] = torch.tensor(chars[:max_word_len])

        return word_ids, char_ids
    return collate_fn


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Loading POS backbone...")
    pos_ckpt = torch.load("checkpoints/pos_bilstm_best.pt", map_location="cpu")
    word_vocab = pos_ckpt["word_vocab"]
    char_vocab = pos_ckpt["char_vocab"]

    pos_tagger = OctoTetrahedralPosTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_emb=32, word_emb=128, hidden_dim=256,
        dropout=0.3, max_loops=3,
    ).to(device)
    pos_tagger.load_state_dict(pos_ckpt["model_state_dict"], strict=False)

    model = OctoDualHead(pos_tagger).to(device)
    model._word_vocab = word_vocab
    model._char_vocab = char_vocab

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    pos_dataset = CLARINWordDataset(args.data_path, char_vocab, word_vocab)
    pos_loader = DataLoader(pos_dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda b: collate_words(b))

    lm_dataset = LMDataset(args.data_path, word_vocab, max_len=100)
    lm_collate = make_lm_collate(word_vocab, char_vocab)
    lm_loader = DataLoader(lm_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lm_collate)

    print(f"POS dataset: {len(pos_dataset)} sentences | LM dataset: {len(lm_dataset)} sentences")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_combined = float("inf")
    pos_loss_weight = args.pos_w
    lm_loss_weight = args.lm_w

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, pos_w={pos_loss_weight}, lm_w={lm_loss_weight}\n")

    for epoch in range(args.epochs):
        model.train()
        pos_loss_sum = 0.0
        lm_loss_sum = 0.0
        n_pos = 0
        n_lm = 0
        t0 = time.time()

        pos_iter = iter(pos_loader)
        lm_iter = iter(lm_loader)

        n_batches = max(len(pos_loader), len(lm_loader))

        for batch_idx in range(n_batches):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            try:
                pos_batch = next(pos_iter)
            except StopIteration:
                pos_iter = iter(pos_loader)
                pos_batch = next(pos_iter)

            pos_w = pos_batch["word_ids"].to(device)
            pos_c = pos_batch["char_ids"].to(device)
            pos_y = pos_batch["pos_ids"].to(device)
            pos_out = model(pos_w, pos_c)
            pos_preds = pos_out["pos_logits"].reshape(-1, NUM_POS)
            pos_targets = pos_y.reshape(-1)
            pos_mask = pos_targets != -100
            if pos_mask.sum() > 0:
                pos_loss = F.cross_entropy(pos_preds[pos_mask], pos_targets[pos_mask])
                total_loss = total_loss + pos_loss_weight * pos_loss
                pos_loss_sum += pos_loss.item()
                n_pos += 1

            try:
                lm_batch = next(lm_iter)
            except StopIteration:
                lm_iter = iter(lm_loader)
                lm_batch = next(lm_iter)

            lm_w, lm_c = lm_batch
            lm_out = model(lm_w.to(device), lm_c.to(device), targets=lm_w.to(device))
            if lm_out["lm_loss"] is not None:
                total_loss = total_loss + lm_loss_weight * lm_out["lm_loss"]
                lm_loss_sum += lm_out["lm_loss"].item()
                n_lm += 1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 100 == 0:
                pl = pos_loss_sum / max(n_pos, 1)
                ll = lm_loss_sum / max(n_lm, 1)
                ppl = math.exp(min(ll, 20))
                print(f"  epoch {epoch} batch {batch_idx}/{n_batches} "
                      f"pos_loss={pl:.4f} lm_ppl={ppl:.1f} phase={lm_out['tp_phase']}")

        scheduler.step()
        avg_pos = pos_loss_sum / max(n_pos, 1)
        avg_lm = lm_loss_sum / max(n_lm, 1)
        ppl = math.exp(min(avg_lm, 20))
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}: pos_loss={avg_pos:.4f} lm_ppl={ppl:.2f} time={elapsed:.0f}s")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "word_vocab": word_vocab,
            "char_vocab": char_vocab,
            "config": {"emb_dim": 128, "hidden": 256, "char_emb": 32, "dropout": 0.3},
            "pos_loss": avg_pos,
            "lm_ppl": ppl,
        }, f"checkpoints/octo_dual_epoch{epoch}.pt")

        combined = avg_pos + avg_lm
        if combined < best_combined:
            best_combined = combined
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "word_vocab": word_vocab,
                "char_vocab": char_vocab,
                "config": {"emb_dim": 128, "hidden": 256, "char_emb": 32, "dropout": 0.3},
                "pos_loss": avg_pos,
                "lm_ppl": ppl,
            }, "checkpoints/octo_dual_best.pt")
            print(f"  New best: combined={combined:.4f}")

        print(f"\n  Generating:")
        seeds = ["the", "cat", "sat"]
        seed_ids = torch.tensor([[word_vocab.get(w, 1) for w in seeds]])
        seed_chars = torch.zeros(1, len(seeds), 30, dtype=torch.long)
        for i, w in enumerate(seeds):
            chars = [char_vocab.get(c, 1) for c in w.lower()[:30]]
            while len(chars) < 30:
                chars.append(CHAR_PAD)
            seed_chars[0, i] = torch.tensor(chars[:30])

        gen_ids = model.generate(seed_ids.to(device), seed_chars.to(device), max_new=30)
        inv_vocab = {v: k for k, v in word_vocab.items()}
        gen_words = [inv_vocab.get(i.item(), "?") for i in gen_ids[0]]
        print(f"  {' '.join(gen_words)}\n")

    print("=" * 50)
    print(f"TRAINING COMPLETE — Best combined: {best_combined:.4f}")
    print(f"Saved: checkpoints/octo_dual_best.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="clarin_enriched_data.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--pos-w", type=float, default=1.0)
    parser.add_argument("--lm-w", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(args)

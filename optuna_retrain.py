#!/usr/bin/env python3
"""
Phase 2: Full-scale retrain with top Optuna configs.
Scales up small configs proportionally, trains on MPS for 30 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import time
import sys
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from core.working_memory import WorkingMemory
from core.reservoir_dynamics import ReservoirDynamics
from core.transcendplexity_integration import TranscendPlexityController

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
                chars = [word_vocab.get(c, 1) if False else char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                while len(chars) < max_word_len:
                    chars.append(CHAR_PAD)
                char_ids[b, i] = torch.tensor(chars[:max_word_len])
        return word_ids, char_ids
    return collate_fn


class CompoundingCohesionTracker:
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
        self.max_len = max_len
        self.word_emb = nn.Embedding(word_vocab_size, d_model, padding_idx=0)
        self.char_emb = nn.Embedding(char_vocab_size, 32, padding_idx=0)
        self.char_proj = nn.Linear(32, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, word_vocab_size),
        )
        self.working_memory = WorkingMemory(num_slots=4, hidden_dim=d_model)
        self.reservoir = ReservoirDynamics(hidden_dim=d_model, n_limbs=8, echo_rho=0.9)
        self.tp_controller = TranscendPlexityController(
            hidden_dim=d_model, num_dimensions=8, alpha_temperature=1.0,
            loss_decay=0.9, phase_history_len=16,
        )
        self._cohesion_tracker = CompoundingCohesionTracker()
        self._tp_state = None
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
        mask = nn.Transformer.generate_square_subsequent_mask(W, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
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
        with torch.no_grad():
            pooled = hidden.mean(dim=1)
            self.reservoir.tick()
            self.reservoir([pooled] * 8)
            tp_state, _ = self.tp_controller(hidden.detach(), step_loss=None)
            self._tp_state = tp_state
        cohesion = self._cohesion_tracker.compute(hidden)
        return {
            "lm_logits": lm_logits, "lm_loss": lm_loss,
            "hidden": hidden, "cohesion": cohesion,
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


def load_eval_sentences(path="data/wikitext2_train.jsonl", n=500):
    sentences = []
    try:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if "text" in entry:
                    words = entry["text"].split()
                    if len(words) >= 3:
                        sentences.append(words)
    except FileNotFoundError:
        pass
    if len(sentences) > n:
        random.seed(42)
        sentences = random.sample(sentences, n)
    return sentences


def evaluate_ppl(model, word_vocab, char_vocab, sentences, device, max_word_len=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for words in sentences:
        ids = torch.tensor([[BOS_ID] + [word_vocab.get(w, 1) for w in words] + [EOS_ID]]).to(device)
        char_ids = torch.zeros(1, ids.size(1), max_word_len, dtype=torch.long, device=device)
        raw = ["<BOS>"] + words + ["<EOS>"]
        for j, w in enumerate(raw):
            if j >= ids.size(1): break
            cs = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
            while len(cs) < max_word_len:
                cs.append(CHAR_PAD)
            char_ids[0, j] = torch.tensor(cs[:max_word_len]).to(device)
        with torch.no_grad():
            out = model(ids, char_ids, targets=ids)
        if out["lm_loss"] is not None:
            shift_logits = out["lm_logits"][:, :-1].reshape(-1, out["lm_logits"].size(-1))
            shift_targets = ids[:, 1:].reshape(-1)
            mask = shift_targets != 0
            if mask.sum() > 0:
                total_loss += F.cross_entropy(shift_logits[mask], shift_targets[mask]).item() * mask.sum().item()
                total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 30))
    return avg_loss, ppl


def load_top_configs(path="optuna_results.jsonl", top_n=3):
    trials = []
    with open(path) as f:
        for line in f:
            result = json.loads(line)
            if result.get("value") is not None and result.get("state") == "COMPLETE":
                trials.append(result)
    trials.sort(key=lambda t: t["value"])
    return trials[:top_n]


def scale_config(small_cfg, scale_factor=2):
    scaled = {
        "d_model": small_cfg["d_model"] * scale_factor,
        "nhead": small_cfg["nhead"],
        "num_layers": small_cfg["num_layers"] * scale_factor,
        "dim_ff": small_cfg["dim_ff"] * scale_factor,
        "dropout": small_cfg["dropout"],
        "lr": small_cfg["lr"],
        "min_freq": small_cfg["min_freq"],
    }
    if scaled["d_model"] % scaled["nhead"] != 0:
        for nh in [4, 6, 8, 12]:
            if scaled["d_model"] % nh == 0:
                scaled["nhead"] = nh
                break
    return scaled


def train_full(config, data_paths, eval_sentences, device, epochs=30, batch_size=8, run_name="config"):
    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Config: d={config['d_model']} nh={config['nhead']} layers={config['num_layers']} "
          f"ff={config['dim_ff']} drop={config['dropout']:.3f} lr={config['lr']:.2e} min_freq={config['min_freq']}")
    print(f"{'='*60}")

    word_vocab, char_vocab = build_vocab(data_paths, min_freq=config["min_freq"])
    print(f"Vocab: {len(word_vocab)} words, {len(char_vocab)} chars")

    model = OctoTransformerLM(
        word_vocab_size=len(word_vocab), char_vocab_size=len(char_vocab),
        d_model=config["d_model"], nhead=config["nhead"],
        num_layers=config["num_layers"], dim_ff=config["dim_ff"],
        dropout=config["dropout"],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.1f}M params")

    if total > 200_000_000:
        print(f"  WARNING: {total/1e6:.1f}M params — may OOM on MPS")
        print(f"  Reducing batch_size to 4")
        batch_size = 4

    dataset = LMDataset(data_paths, word_vocab)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=make_collate(word_vocab, char_vocab),
    )
    print(f"Dataset: {len(dataset)} sentences, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    inv_vocab = {v: k for k, v in word_vocab.items()}
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    print(f"\nTraining: {epochs} epochs, lr={config['lr']}\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (word_ids, char_ids) in enumerate(loader):
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            out = model(word_ids, char_ids, targets=word_ids)
            loss = out["lm_loss"]
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if batch_idx % 500 == 0:
                ppl = math.exp(min(loss.item(), 20))
                print(f"  epoch {epoch} batch {batch_idx}/{len(loader)} loss={loss.item():.4f} ppl={ppl:.1f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f} ppl={ppl:.2f} time={elapsed:.0f}s")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "word_vocab": word_vocab, "char_vocab": char_vocab,
            "config": {k: v for k, v in config.items() if k in ("d_model", "nhead", "num_layers", "dim_ff", "dropout")},
            "loss": avg_loss, "ppl": ppl,
        }
        save_path = save_dir / f"optuna_{run_name}_epoch{epoch}.pt"
        torch.save(ckpt, save_path)

        if eval_sentences:
            eval_loss, eval_ppl = evaluate_ppl(model, word_vocab, char_vocab, eval_sentences, device)
            print(f"  Eval ppl: {eval_ppl:.2f} (train ppl: {ppl:.2f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / f"optuna_{run_name}_best.pt"
            torch.save(ckpt, best_path)
            print(f"  New best: loss={avg_loss:.4f} ppl={ppl:.2f}")

        seeds = ["the", "cat", "sat"]
        seed_ids = torch.tensor([[word_vocab.get(w, 1) for w in seeds]]).to(device)
        seed_chars = torch.zeros(1, len(seeds), 30, dtype=torch.long).to(device)
        for i, w in enumerate(seeds):
            cs = [char_vocab.get(c, 1) for c in w.lower()[:30]]
            while len(cs) < 30:
                cs.append(CHAR_PAD)
            seed_chars[0, i] = torch.tensor(cs[:30]).to(device)
        gen_ids = model.generate(seed_ids, seed_chars, max_new=30)
        gen_words = [inv_vocab.get(idx.item(), "?") for idx in gen_ids[0]]
        print(f"  Gen: {' '.join(gen_words)}\n")

    return {
        "run_name": run_name,
        "config": config,
        "best_loss": best_loss,
        "total_params": total,
    }


def main():
    print("Phase 2: Full-Scale Retrain with Top Optuna Configs")
    print("=" * 60)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")

    top_configs = load_top_configs(top_n=3)
    print(f"Loaded {len(top_configs)} top configs from optuna_results.jsonl")

    data_paths = ["data/mega_train_v2.jsonl"]
    eval_sentences = load_eval_sentences(n=500)
    print(f"Eval sentences: {len(eval_sentences)}")

    all_results = []

    for i, trial in enumerate(top_configs):
        small_cfg = trial["params"]
        small_ppl = trial["value"]
        full_cfg = scale_config(small_cfg, scale_factor=2)
        full_cfg["batch_size"] = 8

        result = train_full(
            full_cfg, data_paths, eval_sentences, device,
            epochs=30, batch_size=8, run_name=f"optuna_top{i+1}",
        )
        result["small_eval_ppl"] = small_ppl
        all_results.append(result)

    all_results.sort(key=lambda r: r["best_loss"])

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for i, r in enumerate(all_results):
        cfg = r["config"]
        print(f"\n  #{i+1} {r['run_name']}: loss={r['best_loss']:.4f} params={r['total_params']/1e6:.1f}M")
        print(f"     d={cfg['d_model']} nh={cfg['nhead']} layers={cfg['num_layers']} "
              f"ff={cfg['dim_ff']} drop={cfg['dropout']:.3f} lr={cfg['lr']:.2e}")
        print(f"     Small eval PPL: {r.get('small_eval_ppl', '?')}")

    best = all_results[0]
    best_ckpt = Path(f"checkpoints/optuna_{best['run_name']}_best.pt")
    final_ckpt = Path("checkpoints/octo_transformer_optuna_best.pt")
    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, final_ckpt)
        print(f"\nBest model saved to: {final_ckpt}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()

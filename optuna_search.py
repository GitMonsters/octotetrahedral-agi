#!/usr/bin/env python3
"""
Phase 1: Small-scale Optuna hyperparameter sweep for OctoTetrahedral transformer.
Runs ~50 trials of small models (~3-5M params) on CPU to find optimal configs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import time
import sys
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from core.working_memory import WorkingMemory
from core.reservoir_dynamics import ReservoirDynamics
from core.transcendplexity_integration import TranscendPlexityController

import optuna
from optuna.trial import TrialState

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
    def __init__(self, data_paths, word_vocab, max_len=128, max_samples=0):
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
        if max_samples > 0 and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

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
        if W > self.max_len:
            word_ids = word_ids[:, :self.max_len]
            char_ids = char_ids[:, :self.max_len]
            W = self.max_len
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
        if targets is not None and targets.size(1) > hidden.size(1):
            targets = targets[:, :hidden.size(1)]
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


DATA_PATHS = [
    "data/mega_train_v2.jsonl",
]

MAX_TRAIN_SAMPLES = 5000
EVAL_SENTENCES = []
VOCAB_CACHE = {}


def cached_build_vocab(min_freq):
    if min_freq not in VOCAB_CACHE:
        VOCAB_CACHE[min_freq] = build_vocab(DATA_PATHS, min_freq=min_freq)
    return VOCAB_CACHE[min_freq]


def estimate_params(vocab_size, char_vocab_size, d_model, num_layers, dim_ff):
    word_emb = vocab_size * d_model
    char_emb = char_vocab_size * 32 + 32 * d_model
    pos_emb = 128 * d_model
    attn = num_layers * (4 * d_model * d_model + 4 * d_model)
    ffn = num_layers * (2 * d_model * dim_ff + 3 * d_model)
    head = d_model * 256 + 256 * vocab_size
    norm = num_layers * 4 * d_model + 2 * d_model
    other = d_model * 200
    return word_emb + char_emb + pos_emb + attn + ffn + head + norm + other


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


def evaluate_ppl(model, word_vocab, char_vocab, sentences, max_word_len=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    max_pos = getattr(model, 'max_len', 128)
    for words in sentences:
        words = words[:max_pos - 2]
        ids = torch.tensor([[BOS_ID] + [word_vocab.get(w, 1) for w in words] + [EOS_ID]])
        char_ids = torch.zeros(1, ids.size(1), max_word_len, dtype=torch.long)
        raw = ["<BOS>"] + words + ["<EOS>"]
        for j, w in enumerate(raw):
            if j >= ids.size(1): break
            cs = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
            while len(cs) < max_word_len:
                cs.append(CHAR_PAD)
            char_ids[0, j] = torch.tensor(cs[:max_word_len])
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


def objective(trial):
    global EVAL_SENTENCES

    d_model = trial.suggest_categorical("d_model", [128, 192, 256, 320])
    nhead = trial.suggest_categorical("nhead", [4, 6, 8])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 6])
    dim_ff = trial.suggest_categorical("dim_ff", [256, 512, 768, 1024])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    min_freq = trial.suggest_categorical("min_freq", [10, 20, 50, 100])

    if d_model % nhead != 0:
        raise optuna.TrialPruned(f"d_model={d_model} not divisible by nhead={nhead}")

    if dim_ff < d_model:
        raise optuna.TrialPruned(f"dim_ff={dim_ff} < d_model={d_model}")

    word_vocab, char_vocab = cached_build_vocab(min_freq)
    vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)

    est_params = estimate_params(vocab_size, char_vocab_size, d_model, num_layers, dim_ff)
    trial.set_user_attr("vocab_size", vocab_size)
    trial.set_user_attr("est_params", est_params)

    if est_params > 15_000_000:
        raise optuna.TrialPruned(f"Estimated too large: {est_params/1e6:.1f}M (vocab={vocab_size})")

    device = torch.device("cpu")

    model = OctoTransformerLM(
        word_vocab_size=vocab_size, char_vocab_size=char_vocab_size,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_ff=dim_ff, dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("total_params", total_params)

    if total_params > 15_000_000:
        raise optuna.TrialPruned(f"Model too large: {total_params/1e6:.1f}M params")

    model.to(device)

    dataset = LMDataset(DATA_PATHS, word_vocab, max_samples=MAX_TRAIN_SAMPLES)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=make_collate(word_vocab, char_vocab),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    if not EVAL_SENTENCES:
        EVAL_SENTENCES = load_eval_sentences(n=500)

    eval_evaluated = False
    best_eval_ppl = float("inf")

    for epoch in range(5):
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

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        train_ppl = math.exp(min(avg_loss, 20))

        if EVAL_SENTENCES:
            eval_loss, eval_ppl = evaluate_ppl(model, word_vocab, char_vocab, EVAL_SENTENCES)
            eval_evaluated = True
            if eval_ppl < best_eval_ppl:
                best_eval_ppl = eval_ppl
        else:
            eval_ppl = train_ppl

        trial.set_user_attr(f"train_ppl_epoch{epoch}", train_ppl)
        if eval_evaluated:
            trial.set_user_attr(f"eval_ppl_epoch{epoch}", eval_ppl)

        trial.report(eval_ppl if eval_evaluated else train_ppl, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_eval_ppl if eval_evaluated else train_ppl


def main(n_trials=60):
    print("Phase 1: Small-Scale Optuna Hyperparameter Sweep")
    print("=" * 60)

    study = optuna.create_study(
        study_name="octotetrahedral_small_sweep",
        storage="sqlite:///optuna_study.db",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True,
    )

    results_path = Path("optuna_results.jsonl")

    def callback(study, trial):
        result = {
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "total_params": trial.user_attrs.get("total_params"),
            "vocab_size": trial.user_attrs.get("vocab_size"),
            "duration": trial.duration.total_seconds(),
        }
        for k, v in trial.user_attrs.items():
            if k not in ("total_params", "vocab_size"):
                result[k] = v
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"Running {n_trials} trials...")
    print(f"Search space:")
    print(f"  d_model:    [128, 192, 256, 320]")
    print(f"  nhead:      [4, 6, 8]")
    print(f"  num_layers: [2, 3, 4, 6]")
    print(f"  dim_ff:     [256, 512, 768, 1024]")
    print(f"  dropout:    [0.05, 0.3]")
    print(f"  lr:         [5e-4, 5e-3] (log)")
    print(f"  batch_size: [8, 16, 32]")
    print(f"  min_freq:   [10, 20, 50, 100]")
    print(f"Max params per trial: 15M")
    print(f"Pruner: MedianPruner(startup=5, warmup=3)")
    print()

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)

    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    print(f"  Completed: {len(complete_trials)}")
    print(f"  Pruned:    {len(pruned_trials)}")

    print(f"\nTop 5 configs by eval PPL:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )
    for i, trial in enumerate(sorted_trials[:5]):
        p = trial.params
        tp = trial.user_attrs.get("total_params", 0)
        print(f"\n  #{i+1} eval_ppl={trial.value:.2f} params={tp/1e6:.1f}M duration={trial.duration.total_seconds():.0f}s")
        print(f"     d={p['d_model']} nh={p['nhead']} layers={p['num_layers']} ff={p['dim_ff']} "
              f"drop={p['dropout']:.3f} lr={p['lr']:.2e} bs={p['batch_size']} minfreq={p['min_freq']}")

    print(f"\nResults saved to: optuna_results.jsonl")
    print(f"Study DB: optuna_study.db")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=5000)
    args = p.parse_args()
    MAX_TRAIN_SAMPLES = args.max_samples
    main(n_trials=args.n_trials)

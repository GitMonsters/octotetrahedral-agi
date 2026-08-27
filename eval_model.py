#!/usr/bin/env python3
"""Evaluate transformer LM on WikiText-2 test set."""
import torch
import torch.nn.functional as F
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_transformer import OctoTransformerLM, CHAR_PAD, BOS_ID, EOS_ID


def load_wikitext2_test():
    for p in ("data/wikitext2_test.jsonl", "data/eval_heldout.jsonl", "data/wikitext2_train.jsonl"):
        path = Path(p)
        if path.exists():
            sentences = []
            with open(path) as f:
                for line in f:
                    entry = json.loads(line)
                    if "text" in entry:
                        words = entry["text"].split()
                        if len(words) >= 3:
                            sentences.append(words)
            return sentences
    print("No WikiText-2 data found, using training set as proxy")
    return []


def evaluate_ppl(model, word_vocab, char_vocab, sentences, device, batch_size=16):
    model.eval()
    max_word_len = 30
    max_len = getattr(model, "max_len", 128)
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        max_len_batch = min(max(len(s) for s in batch) + 2, max_len)
        B = len(batch)
        max_len_batch = min(max(len(s) for s in batch) + 2, max_len)

        word_ids = torch.zeros(B, max_len_batch, dtype=torch.long)
        char_ids = torch.zeros(B, max_len_batch, max_word_len, dtype=torch.long)

        for b, words in enumerate(batch):
            ids = [BOS_ID] + [word_vocab.get(w, 1) for w in words] + [EOS_ID]
            ids = ids[:max_len_batch]
            word_ids[b, :len(ids)] = torch.tensor(ids)
            raw = ["<BOS>"] + words + ["<EOS>"]
            for j, w in enumerate(raw):
                if j >= max_len_batch:
                    break
                chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                while len(chars) < max_word_len:
                    chars.append(CHAR_PAD)
                char_ids[b, j] = torch.tensor(chars[:max_word_len])

        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)

        with torch.no_grad():
            out = model(word_ids, char_ids, targets=word_ids)

        if out["lm_loss"] is not None:
            shift_logits = out["lm_logits"][:, :-1].reshape(-1, out["lm_logits"].size(-1))
            shift_targets = word_ids[:, 1:].reshape(-1)
            mask = shift_targets != 0
            if mask.sum() > 0:
                total_loss += F.cross_entropy(shift_logits[mask], shift_targets[mask]).item() * mask.sum().item()
                total_tokens += mask.sum().item()
                total_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 30))
    return avg_loss, ppl, total_tokens


def evaluate_generation(model, word_vocab, char_vocab, device):
    inv = {v: k for k, v in word_vocab.items()}
    prompts = [
        ["the", "cat", "sat"],
        ["in", "the", "year"],
        ["she", "was", "a"],
        ["he", "said", "that"],
        ["the", "university", "of"],
        ["Instruction", ":", "What", "is", "AI", "?"],
        ["Instruction", ":", "Explain", "machine", "learning"],
    ]
    results = []
    for words in prompts:
        ids = torch.tensor([[word_vocab.get(w, 1) for w in words]]).to(device)
        chars = torch.zeros(1, len(words), 30, dtype=torch.long).to(device)
        for i, w in enumerate(words):
            cs = [char_vocab.get(c, 1) for c in w.lower()[:30]]
            while len(cs) < 30:
                cs.append(CHAR_PAD)
            chars[0, i] = torch.tensor(cs[:30]).to(device)
        gen = model.generate(ids, chars, max_new=30, temperature=1.0, top_k=50, rep_penalty=1.5)
        gen_words = [inv.get(j.item(), "?") for j in gen[0]]
        results.append(" ".join(gen_words))
    return results


def evaluate_perturbation(model, word_vocab, char_vocab, sentences, device, n_pairs=200):
    model.eval()
    import random
    random.seed(42)
    max_word_len = 30

    common_words = [w for w, i in word_vocab.items()
                    if i > 3 and w not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")]

    def encode_batch(sents):
        max_len_batch = min(max(len(s) for s in sents) + 2, getattr(model, "max_len", 128))
        B = len(sents)
        wids = torch.zeros(B, max_len_batch, dtype=torch.long)
        cids = torch.zeros(B, max_len_batch, max_word_len, dtype=torch.long)
        for b, words in enumerate(sents):
            ids = [BOS_ID] + [word_vocab.get(w, 1) for w in words] + [EOS_ID]
            ids = ids[:max_len_batch]
            wids[b, :len(ids)] = torch.tensor(ids)
            raw = ["<BOS>"] + words + ["<EOS>"]
            for j, w in enumerate(raw):
                if j >= max_len_batch:
                    break
                chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
                while len(chars) < max_word_len:
                    chars.append(CHAR_PAD)
                cids[b, j] = torch.tensor(chars[:max_word_len])
        return wids.to(device), cids.to(device)

    def batch_ppl(sents):
        if not sents:
            return 0.0
        wids, cids = encode_batch(sents)
        with torch.no_grad():
            out = model(wids, cids, targets=wids)
        if out["lm_loss"] is None:
            return 0.0
        shift_logits = out["lm_logits"][:, :-1].reshape(-1, out["lm_logits"].size(-1))
        shift_targets = wids[:, 1:].reshape(-1)
        mask = shift_targets != 0
        if mask.sum() == 0:
            return 0.0
        loss = F.cross_entropy(shift_logits[mask], shift_targets[mask]).item()
        return math.exp(min(loss, 30))

    pairs_sampled = sentences[:n_pairs] if len(sentences) >= n_pairs else sentences

    base_ppls = []
    perturb_ppls = []
    perturb_positions = []

    for words in pairs_sampled:
        if len(words) < 4:
            continue
        orig_ppl = batch_ppl([words])
        if orig_ppl == 0:
            continue

        pos = random.randint(1, len(words) - 2)
        orig_word = words[pos]
        candidates = [w for w in common_words if w != orig_word]
        if not candidates:
            continue
        new_word = random.choice(candidates)
        perturbed = words[:pos] + [new_word] + words[pos+1:]
        pert_ppl = batch_ppl([perturbed])
        if pert_ppl == 0:
            continue

        base_ppls.append(orig_ppl)
        perturb_ppls.append(pert_ppl)
        perturb_positions.append(pos)

    if not base_ppls:
        return {"error": "no valid pairs"}

    import numpy as np
    base_arr = np.array(base_ppls)
    pert_arr = np.array(perturb_ppls)
    ratio = pert_arr / (base_arr + 1e-8)

    n_increase = int((pert_arr > base_arr).sum())
    n_decrease = int((pert_arr < base_arr).sum())
    n_unchanged = len(base_ppls) - n_increase - n_decrease

    return {
        "n_pairs": len(base_ppls),
        "base_ppl_mean": float(base_arr.mean()),
        "base_ppl_std": float(base_arr.std()),
        "perturb_ppl_mean": float(pert_arr.mean()),
        "perturb_ppl_std": float(pert_arr.std()),
        "ratio_mean": float(ratio.mean()),
        "ratio_median": float(np.median(ratio)),
        "n_increase": n_increase,
        "n_decrease": n_decrease,
        "n_unchanged": n_unchanged,
        "pct_ppl_increased": n_increase / len(base_ppls) * 100,
        "robustness_score": float(1.0 - min(1.0, max(0.0, (ratio.mean() - 1.0) * 2))),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/octo_transformer_best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wc, cc = ckpt["word_vocab"], ckpt["char_vocab"]
    cfg = ckpt["config"]
    model = OctoTransformerLM(len(wc), len(cc), **cfg)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"  Loaded with {len(missing)} missing (monitoring stubs)")

    total = sum(p.numel() for p in model.parameters())
    core = sum(p.numel() for n, p in model.named_parameters()
               if not any(m in n for m in ("working_memory", "reservoir", "tp_controller", "_cohesion", "cog_geom", "recursive_obj", "wm_proj")))
    print(f"  Total: {total/1e6:.1f}M, Core: {core/1e6:.1f}M")
    print(f"  Training loss: {ckpt.get('loss', '?'):.4f}, ppl: {ckpt.get('ppl', '?'):.2f}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print()

    print("=" * 60)
    print("PERPLEXITY EVALUATION")
    print("=" * 60)
    sentences = load_wikitext2_test()
    print(f"Test sentences: {len(sentences)}")
    if sentences:
        t0 = time.time()
        avg_loss, ppl, n_tokens = evaluate_ppl(model, wc, cc, sentences, device, args.batch_size)
        elapsed = time.time() - t0
        print(f"  Loss:      {avg_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Tokens:    {n_tokens:,}")
        print(f"  Time:      {elapsed:.1f}s")
    print()

    print("=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)
    results = evaluate_generation(model, wc, cc, device)
    for r in results:
        print(f"  {r}")
    print()

    print("=" * 60)
    print("PERTURBATION EVALUATION")
    print("=" * 60)
    if sentences:
        t0 = time.time()
        perturb = evaluate_perturbation(model, wc, cc, sentences, device, n_pairs=200)
        elapsed = time.time() - t0
        if "error" in perturb:
            print(f"  Error: {perturb['error']}")
        else:
            print(f"  Pairs tested:       {perturb['n_pairs']}")
            print(f"  Base PPL:           {perturb['base_ppl_mean']:.2f} ± {perturb['base_ppl_std']:.2f}")
            print(f"  Perturbed PPL:      {perturb['perturb_ppl_mean']:.2f} ± {perturb['perturb_ppl_std']:.2f}")
            print(f"  Ratio (pert/base):  {perturb['ratio_mean']:.3f} (median: {perturb['ratio_median']:.3f})")
            print(f"  PPL increased:      {perturb['pct_ppl_increased']:.1f}% ({perturb['n_increase']}/{perturb['n_pairs']})")
            print(f"  PPL decreased:      {perturb['n_decrease']}/{perturb['n_pairs']}")
            print(f"  Robustness score:   {perturb['robustness_score']:.3f}")
            print(f"  Time:               {elapsed:.1f}s")
            print()
            print("  Interpretation:")
            if perturb['ratio_mean'] < 1.05:
                print("    Model is ROBUST — perturbations barely affect predictions")
                print("    (model relies on deep structure, not surface tokens)")
            elif perturb['ratio_mean'] < 1.2:
                print("    Model has MODERATE robustness — some sensitivity to word changes")
                print("    (model uses both surface and structural patterns)")
            else:
                print("    Model is FRAGILE — heavily dependent on specific tokens")
                print("    (model may be memorizing rather than reasoning)")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:         {total/1e6:.1f}M params")
    print(f"  Vocab:         {len(wc)} words, {len(cc)} chars")
    print(f"  Train loss:    {ckpt.get('loss', '?'):.4f}")
    print(f"  Train ppl:     {ckpt.get('ppl', '?'):.2f}")
    if sentences:
        print(f"  Eval ppl:      {ppl:.2f}")
    print(f"  Diagnostics:   {model.get_diagnostics()}")

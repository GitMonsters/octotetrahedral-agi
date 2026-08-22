#!/usr/bin/env python3
"""Fine-tune transformer LM on instruction data for chat."""
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
from train_transformer import (
    OctoTransformerLM, LMDataset, make_collate, build_vocab, CHAR_PAD, BOS_ID, EOS_ID,
)

def fine_tune(args):
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading base model from {args.base_checkpoint}...")
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    wc = ckpt["word_vocab"]
    cc = ckpt["char_vocab"]
    cfg = ckpt["config"]

    model = OctoTransformerLM(len(wc), len(cc), **cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  Loaded with {len(missing)} missing (monitoring stubs)")
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    print(f"Loading instruction data from {args.data_path}...")
    dataset = LMDataset([args.data_path], wc)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(wc, cc),
    )
    print(f"  {len(dataset)} instruction samples, {len(loader)} batches/epoch")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    inv = {v: k for k, v in wc.items()}

    best_loss = float("inf")

    print(f"\nFine-tuning: {args.epochs} epochs, lr={args.lr}\n")

    for epoch in range(args.epochs):
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
            if batch_idx % 50 == 0:
                ppl = math.exp(min(loss.item(), 20))
                print(f"  epoch {epoch} batch {batch_idx}/{len(loader)} loss={loss.item():.4f} ppl={ppl:.1f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f} ppl={ppl:.2f} time={elapsed:.0f}s")

        ckpt_data = {
            "epoch": epoch, "model": model.state_dict(),
            "word_vocab": wc, "char_vocab": cc, "config": cfg,
            "loss": avg_loss, "ppl": ppl,
        }
        torch.save(ckpt_data, f"checkpoints/octo_chat_epoch{epoch}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, "checkpoints/octo_chat_best.pt")
            print(f"  New best: loss={avg_loss:.4f} ppl={ppl:.2f}")

        test_seeds = [
            ["Instruction", ":", "What", "is", "artificial", "intelligence", "?"],
            ["Instruction", ":", "Explain", "machine", "learning", "."],
            ["Instruction", ":", "How", "does", "deep", "learning", "work", "?"],
        ]
        print("  Samples:")
        for seeds in test_seeds:
            ids = torch.tensor([[wc.get(w, 1) for w in seeds]]).to(device)
            chars = torch.zeros(1, len(seeds), 30, dtype=torch.long).to(device)
            for i, w in enumerate(seeds):
                cs = [cc.get(c, 1) for c in w.lower()[:30]]
                while len(cs) < 30:
                    cs.append(CHAR_PAD)
                chars[0, i] = torch.tensor(cs[:30]).to(device)
            gen = model.generate(ids, chars, max_new=30, temperature=0.7, top_k=30, rep_penalty=3.0)
            words = [inv.get(j.item(), "?") for j in gen[0]]
            print(f"    {' '.join(words)}")
        print()

    print("=" * 50)
    print(f"FINE-TUNING COMPLETE - Best loss: {best_loss:.4f}")
    print("Saved: checkpoints/octo_chat_best.pt")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", default="checkpoints/octo_transformer_best.pt")
    p.add_argument("--data-path", default="data/instructions.jsonl")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default=None)
    fine_tune(p.parse_args())

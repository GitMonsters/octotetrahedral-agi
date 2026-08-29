import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_transformer import OctoTransformerLM, CHAR_PAD

QUERIES = [
    "What is physics?",
    "What is string theory?",
    "What is machine learning?",
    "What is quantum computing?",
    "Explain gravity.",
    "Who are you?",
    "Hello!",
    "What is the universe?",
    "What is artificial intelligence?",
]

def run(checkpoint, name, max_new=24):
    print("=" * 70)
    print(name)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    wc, cc, cfg = ckpt["word_vocab"], ckpt["char_vocab"], ckpt["config"]
    model = OctoTransformerLM(len(wc), len(cc), **cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    inv = {v: k for k, v in wc.items()}
    seed_tokens = ["Question", ":", None, "Response", ":"]
    for q in QUERIES:
        seed = ["Question", ":"] + q.split() + ["Response", ":"]
        ids = torch.tensor([[wc.get(w, 1) for w in seed]])
        chars = torch.zeros(1, len(seed), 30, dtype=torch.long)
        for i, w in enumerate(seed):
            cs = [cc.get(c, 1) for c in w.lower()[:30]]
            while len(cs) < 30:
                cs.append(CHAR_PAD)
            chars[0, i] = torch.tensor(cs[:30])
        with torch.no_grad():
            gen = model.generate(ids, chars, max_new=max_new,
                                 temperature=0.7, top_k=30, rep_penalty=3.0)
        words = [inv.get(j.item(), "?") for j in gen[0]]
        print(f"  Q: {q}\n  A: {' '.join(words)}\n")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
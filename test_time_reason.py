"""System-2 test-time reasoning for the OctoTetrahedral transformer LM.

Exploits the model's one measured strength — LM-scoring of candidate text —
to attack the generation wall (token soup). At decode time we:

  1. generate K drafts of the completion with a temperature sweep (explore);
  2. self-score each draft by per-token LM perplexity (typicality prior);
  3. build a position-wise consensus across drafts (self-consistency);

and return the best draft plus a full reasoning trace. No retraining needed;
this is compute spent at inference time, the same trick frontier models use
with chain-of-thought / self-consistency / best-of-N.
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F

from train_transformer import CHAR_PAD, BOS_ID, EOS_ID


def encode_prompt(words, word_vocab, char_vocab, max_word_len=30):
    """Encode a prompt word list into seed word/char tensors ([1, S] / [1, S, L])."""
    seed_ids = torch.tensor([[BOS_ID] + [word_vocab.get(w.lower(), 1) for w in words]], dtype=torch.long)
    bos_chars = [char_vocab.get(c, 1) for c in "<bos>"[:max_word_len]]
    while len(bos_chars) < max_word_len:
        bos_chars.append(CHAR_PAD)
    seed_chars = torch.zeros(1, len(words) + 1, max_word_len, dtype=torch.long)
    seed_chars[0, 0] = torch.tensor(bos_chars[:max_word_len])
    for i, w in enumerate(words):
        cs = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
        while len(cs) < max_word_len:
            cs.append(CHAR_PAD)
        seed_chars[0, i + 1] = torch.tensor(cs[:max_word_len])
    return seed_ids, seed_chars


def decode_ids(ids, inv_vocab):
    return [inv_vocab.get(i, "?") for i in ids]


@torch.no_grad()
def draft(model, seed_ids, seed_chars, max_new, temperature, top_k, rep_penalty):
    """Autoregressively complete the seed; returns the new token id list."""
    S = seed_ids.size(1)
    full = model.generate(
        seed_ids.to(seed_ids.device), seed_chars.to(seed_ids.device),
        max_new=max_new, temperature=temperature, top_k=top_k, rep_penalty=rep_penalty,
    )
    return [int(i) for i in full[0][S:]]


@torch.no_grad()
def score_continuation(model, seed_ids, seed_chars, completion, device=None):
    """Mean per-token NLL / PPL of a completion conditioned on the seed.

    Only the completion positions contribute to the loss, so short drafts are
    not unfairly penalised. PAD tokens inside the completion (none normally)
    are masked out of the reduction.
    """
    if device is not None:
        seed_ids = seed_ids.to(device)
        seed_chars = seed_chars.to(device)
    S = seed_ids.size(1)
    max_len = getattr(model, "max_len", 128)
    L = seed_chars.size(2)

    ids = torch.cat([seed_ids, torch.tensor([completion + [EOS_ID]], dtype=torch.long)], dim=1)
    ids = ids[:, :max_len]
    extra = torch.zeros(1, ids.size(1) - S, L, dtype=torch.long)
    chars = torch.cat([seed_chars, extra], dim=1)[:, :max_len]

    out = model(ids, chars, targets=ids)
    logits = out["lm_logits"]  # [1, T, V]

    # Prediction at column i targets ids[i+1]; completion occupies columns S..S+C-1
    C = len([t for t in completion if t != 0])
    pred_logits = logits[0, S - 1:S - 1 + C, :]
    targets = torch.tensor([t for t in completion[:C] if t != 0], dtype=torch.long, device=pred_logits.device)
    pred_logits = pred_logits[:C]

    nll = F.cross_entropy(pred_logits, targets, reduction="sum")
    if C == 0:
        return None, None
    mean_nll = float(nll) / C
    return mean_nll, math.exp(min(mean_nll, 30.0))


@torch.no_grad()
def next_word_scores(model, seed_words, candidates, word_vocab, char_vocab, device=None):
    """Score each candidate continuation word by conditioned-LM PPL.

    Returns a list of (word, ppl, nll) sorted ascending by PPL (best first).
    """
    seed_ids, seed_chars = encode_prompt(seed_words, word_vocab, char_vocab)
    scored = []
    for w in candidates:
        w_id = word_vocab.get(w.lower(), 1)
        nll, ppl = score_continuation(model, seed_ids, seed_chars, [w_id], device=device)
        if ppl is not None:
            scored.append((w, ppl, nll))
    scored.sort(key=lambda x: x[1])
    return scored


@torch.no_grad()
def reason(model, word_vocab, char_vocab, device, prompt,
           num_drafts=6, max_tokens=24, temperature=0.8, top_k=30,
           rep_penalty=1.3):
    """Multi-draft test-time reasoning over a prompt.

    Returns chosen text (best-PPL draft), the self-consistency trace, and all
    candidate drafts with their LM scores.
    """
    model.eval()
    inv = {v: k for k, v in word_vocab.items()}
    words = prompt.split() or ["the"]
    seed_ids, seed_chars = encode_prompt(words, word_vocab, char_vocab)

    temps = [max(0.2, temperature + d) for d in (-0.3, 0.0, 0.3)]
    candidates = []
    for k in range(num_drafts):
        temp = temps[k % len(temps)]
        comp = draft(model, seed_ids, seed_chars, max_tokens, temp, top_k, rep_penalty)
        nll, ppl = score_continuation(model, seed_ids, seed_chars, comp, device=device)
        text = " ".join([w for w in decode_ids(comp, inv) if w not in ("<PAD>", "<BOS>", "<EOS>", "?")])
        candidates.append({
            "index": k,
            "text": text,
            "tokens": comp,
            "ppl": round(float(ppl), 3) if ppl is not None else None,
            "mean_nll": round(float(nll), 3) if nll is not None else None,
            "temperature": round(temp, 2),
            "empty": len(comp) == 0,
        })

    scored = [c for c in candidates if c["ppl"] is not None and not c["empty"]]
    scored.sort(key=lambda c: c["ppl"])
    chosen = scored[0] if scored else candidates[0]

    consensus_text = ""
    agreement = 0.0
    if scored:
        texts = [c["text"].split() for c in scored]
        maxp = max(len(t) for t in texts)
        votes = []
        for p in range(maxp):
            at_pos = [t[p] for t in texts if len(t) > p]
            if not at_pos:
                break
            top, cnt = Counter(at_pos).most_common(1)[0]
            votes.append(top)
            agreement += cnt / len(at_pos)
        consensus_text = " ".join(votes)
        agreement = agreement / maxp if maxp else 0.0

    dist_first = len({c["text"].split()[0] for c in scored if c["text"].split()}) if scored else 0

    trace = {
        "num_drafts": num_drafts,
        "temps": temps,
        "distinct_first_tokens": dist_first,
        "consensus_agreement": round(agreement, 3),
        "ppl_best": chosen.get("ppl"),
        "ppl_spread": round(max(c["ppl"] or 0 for c in scored) - min(c["ppl"] or 0 for c in scored), 3) if scored else None,
    }
    pull_phase = model.get_diagnostics() if hasattr(model, "get_diagnostics") else {}
    if pull_phase:
        trace["diagnostics"] = pull_phase

    return {
        "prompt": prompt,
        "chosen": chosen,
        "consensus": {"text": consensus_text, "agreement": round(agreement, 3)},
        "candidates": [{k: c[k] for k in ("index", "text", "ppl", "temperature", "empty")} for c in candidates],
        "trace": trace,
    }
#!/usr/bin/env python3
"""Evolve reason() hyperparameters with a small evolutionary search.

Mirrors Weco's AIDE rigor: a VISIBLE practice test (fixed prompts from
eval_heldout) drives the fitness signal; a SEALED hidden final exam
(data/eval_hidden.jsonl, corpus docs the LM never trained on) is touched
exactly ONCE, at the end, to decide whether the evolved config truly beats the
default. Population -> mutate one knob -> tournament selection -> elitism.
A JSON evolution checkpoint is written every generation so a crash resumes
without losing the population.

Fitness (higher is better), evaluated with a deterministic per-config seed:
  0.50 * consensus_agreement + 0.30 * (1 - min(chosen_ppl / 6, 1))
  + 0.20 * min(distinct_first_mean / 3, 1) - 0.5 * fail_fraction
"""
import argparse
import json
import math
import sys
import time
import torch
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_transformer import OctoTransformerLM
from test_time_reason import reason

DEFAULT_CFG = {"num_drafts": 6, "max_tokens": 24, "temperature": 0.8,
               "top_k": 30, "rep_penalty": 1.3, "stall_restart": True}

NUM_DRAFTS_CHOICES = [2, 3, 4, 6, 8]
MAX_TOKENS_CHOICES = [8, 12, 16, 24, 32]


def load_sentences(path, min_len=3):
    out = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            words = (entry.get("text") or " ".join(entry.get("tokens", []))).split()
            if len(words) >= min_len:
                out.append(words)
    return out


def config_seed(cfg):
    return zlib.crc32(json.dumps(cfg, sort_keys=True).encode())


def evaluate_config(model, wc, cc, device, prompts, cfg):
    """Score one config over the visible prompts once, deterministically seeded."""
    torch.manual_seed(config_seed(cfg))
    agrees, ppls, divs = [], [], []
    fails = 0
    for p in prompts:
        try:
            r = reason(model, wc, cc, device, " ".join(p), **cfg)
        except Exception as e:
            fails += 1
            continue
        t = r["trace"]
        if t["consensus_agreement"] is not None:
            agrees.append(t["consensus_agreement"])
        if t["ppl_best"] is not None:
            ppls.append(t["ppl_best"])
        divs.append(t["distinct_first_tokens"])
    ok = len(agrees)
    if ok == 0:
        return None
    agree_m = sum(agrees) / ok
    ppl_m = sum(ppls) / max(len(ppls), 1)
    div_m = sum(divs) / ok
    total = ok + fails
    fitness = (0.50 * agree_m
               + 0.30 * (1.0 - min(ppl_m / 6.0, 1.0))
               + 0.20 * min(div_m / 3.0, 1.0)
               - 0.5 * (fails / total))
    return {"fitness": round(fitness, 4), "agreement": round(agree_m, 3),
            "ppl_best_mean": round(ppl_m, 3), "distinct_first_mean": round(div_m, 2),
            "ok": ok, "fails": fails}


def mutate(cfg, rng):
    c = dict(cfg)
    kw = rng.choice(["num_drafts", "max_tokens", "temperature", "top_k",
                     "rep_penalty", "stall_restart"])
    if kw == "num_drafts":
        c[kw] = int(rng.choice(NUM_DRAFTS_CHOICES))
    elif kw == "max_tokens":
        c[kw] = int(rng.choice(MAX_TOKENS_CHOICES))
    elif kw == "temperature":
        c[kw] = round(min(1.5, max(0.3, c[kw] + rng.gauss(0, 0.15))), 2)
    elif kw == "top_k":
        c[kw] = int(max(3, round(c[kw] + rng.gauss(0, 8))))
    elif kw == "rep_penalty":
        c[kw] = round(min(2.2, max(1.0, c[kw] + rng.gauss(0, 0.2))), 2)
    elif kw == "stall_restart":
        c[kw] = not c[kw]
    return c


def tournament(pop, rng, k=2):
    return max([rng.sample(pop, 1)[0] for _ in range(k)], key=lambda x: x["fitness"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/octo_transformer_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--prompts-visible", type=int, default=12)
    ap.add_argument("--prompts-hidden", type=int, default=15)
    ap.add_argument("--init-seed", type=int, default=1337)
    ap.add_argument("--resume-ckpt", default=None)
    ap.add_argument("--results", default="results/reason_evolution_v1.json")
    ap.add_argument("--warning-only", action="store_true",
                    help="hidden gate logs PASS/FAIL but exits 0 either way")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wc, cc = ck["word_vocab"], ck["char_vocab"]
    model = OctoTransformerLM(len(wc), len(cc), **ck["config"])
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"({Path(args.checkpoint).name}, epoch {ck.get('epoch')})")

    hidden_dir = Path(__file__).resolve().parent.parent / "data"
    visible = load_sentences(hidden_dir / "eval_heldout.jsonl")
    hidden_pool = load_sentences(hidden_dir / "eval_hidden.jsonl")
    assert visible and hidden_pool, "need data/eval_heldout.jsonl + data/eval_hidden.jsonl"
    rng = __import__("random").Random(args.init_seed)
    visible_prompts = rng.sample(visible, min(len(visible), args.prompts_visible))
    hidden_prompts = rng.sample(hidden_pool, min(len(hidden_pool), args.prompts_hidden))
    print(f"Visible prompts: {len(visible_prompts)} | "
          f"Sealed hidden prompts: {len(hidden_prompts)} (touched once, at the gate)")

    pop = []
    gen_start = 0
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ev = json.loads(Path(args.resume_ckpt).read_text())
        pop = ev["population"]
        gen_start = ev["generation"] + 1
        print(f"Resumed evolution at gen {gen_start} with {len(pop)} members")

    if not pop:
        pop.append({"config": dict(DEFAULT_CFG)})
        for _ in range(1, args.population):
            pop.append({"config": mutate(DEFAULT_CFG, rng)})

    for m in pop:
        if "fitness" not in m:
            m.update(evaluate_config(model, wc, cc, device, visible_prompts, m["config"]) or
                     {"fitness": -1.0, "agreement": 0.0, "ppl_best_mean": 999.0,
                      "distinct_first_mean": 0.0, "ok": 0, "fails": len(visible_prompts)})

    best_seen = max(pop, key=lambda x: x["fitness"])
    t0 = time.time()
    for g in range(gen_start, args.generations):
        # elitism: keep best 2
        pop.sort(key=lambda x: x["fitness"], reverse=True)
        elites = pop[:2]
        new_gen = [dict(e) for e in elites]
        while len(new_gen) < args.population:
            parent = tournament(pop, rng)
            child = dict(parent)
            child["config"] = mutate(parent["config"], rng)
            child.pop("fitness", None)
            new_gen.append(child)
        for m in new_gen[len(elites):]:
            m.update(evaluate_config(model, wc, cc, device, visible_prompts, m["config"]) or
                     {"fitness": -1.0, "agreement": 0.0, "ppl_best_mean": 999.0,
                      "distinct_first_mean": 0.0, "ok": 0, "fails": len(visible_prompts)})
        new_gen.sort(key=lambda x: x["fitness"], reverse=True)
        pop = new_gen
        best = max(pop, key=lambda x: x["fitness"])
        if best["fitness"] > best_seen["fitness"]:
            best_seen = best
        kept = {tuple(sorted(m["config"].items())) for m in pop}
        print(f"[gen {g}] best fitness {best['fitness']} (agree {best['agreement']}, "
              f"ppl {best['ppl_best_mean']}, distinct {best['distinct_first_mean']}, "
              f"fails {best['fails']}) | distinct configs in pop: {len(kept)} | "
              f"elapsed {time.time()-t0:.0f}s")
        if g % 2 == 0:
            Path(args.results).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results).write_text(json.dumps(
                {"generation": g, "population": pop, "best_seen": best_seen},
                indent=2))

    winner = max(pop, key=lambda x: x["fitness"])
    print("\n= Winner (visible test) =")
    print(json.dumps(winner, indent=2))

    # ---- SEALED HIDDEN GATE: touched exactly once ----
    print("\n= Sealed hidden gate (final exam, eval_hidden.jsonl) =")
    torch.manual_seed(777)
    base = evaluate_config(model, wc, cc, device, hidden_prompts, DEFAULT_CFG)
    win = evaluate_config(model, wc, cc, device, hidden_prompts, winner["config"])
    print(f"  default {json.dumps(DEFAULT_CFG)}: {json.dumps(base)}")
    print(f"  winner  {json.dumps(winner['config'])}: {json.dumps(win)}")
    if base and win:
        ok = (win["ppl_best_mean"] <= base["ppl_best_mean"] * 1.05
              and win["agreement"] >= base["agreement"] - 0.05)
        print(f"  GATE: {'PASS -> adopt evolved config' if ok else 'FAIL -> keep default'}")
        if not args.warning_only and not ok:
            print("  (exit 1 because gate failed)")
            sys.exit(1)
    else:
        print("  GATE: INCONCLUSIVE (one side failed)")

    out = {"generation_done": args.generations - 1, "winner": winner,
           "default": DEFAULT_CFG, "visible_prompts": len(visible_prompts),
           "hidden_gate": {"default": base, "winner": win},
           "checkpoint": str(Path(args.checkpoint).name)}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.results}")


if __name__ == "__main__":
    main()
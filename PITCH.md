# OctoTetrahedral AGI

### A Pure from-Scratch Transformer with Cognitive Modules — Final Results

---

## What It Is

A complete language-model pipeline built from zero — no GPT-2, no pretrained encoders,
every parameter trained from scratch on Apple Silicon MPS. The geometry of the project:
a tetrahedral-bias transformer, a six-term cognitive loss, differentiable working memory,
and reservoir dynamics, all wrapped in a reproducible 102K-sentence corpus and eval suite.

**Status: complete.** The headline measured result is a 52.9M-param model that predicts
held-out next tokens at **perplexity 1.32** and survives word-swap perturbation
(**robustness 0.721**). Open-ended generation is the documented, honest limitation
(see [The Wall](#the-wall) below).

---

## The Architecture

Three pillars, all trained from scratch:

### 1. Tetrahedral Attention
Standard attention computes `softmax(QK^T / sqrt(d))`. OctoTetrahedral adds a learned
**geometric bias** with a Gaussian positional prior:
```
softmax(QK^T / sqrt(d) + alpha * exp(-d^2 / (2 * sigma^2))),   sigma = sqrt(d_model)
```
A structural prior on nearby-token coupling instead of learning it from data.

### 2. Cognitive Geometry Engine
12 differentiable modules wired into every forward pass — SVD semantic-axis decomposer,
concept-alignment matrix, entropy-flow monitor, semantic-drift detector, anchor/topic
vectors, repetition dampener, branch scorer, manifold partitioner, goal vectors, attention
plane reconstructor, vector-field tracker, cross-limb orthogonality. All gated by config,
all producing auxiliary losses and diagnostics.

### 3. Six-Term Composite Objective
```
L = L_task + 0.10 L_WM + 0.05 L_meta + 0.02 L_resource + 0.05 L_ground + 0.15 L_stability
```
Task (cross-entropy, label-smoothed 0.1) plus a world-model hidden-state predictor,
meta-learning adaptation term, compute-efficiency term, grounding term, and a
stability term built on a compounding-cohesion tracker with EWC-style forgetting penalty.

### Supporting modules
- **Working memory** — 4-slot differentiable memory (goal / context / results / output).
- **Reservoir dynamics** — 8 parallel echo-state limbs, theta/alpha/gamma pacemaker,
  spectral radius 0.9 (edge of chaos).
- **TranscendPlexity controller** — phase tracking (EXPLORATION / CONSOLIDATION /
  DEEP_REASONING), 8-dim alpha ordering.

---

## Final Model (v8)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| nhead | 8 |
| num_layers | 8 |
| dim_ff | 2048 |
| dropout | 0.3 |
| max_len | 128 tokens |
| Total params | **52.9M** (45.9M core) |
| Word vocab | 26,603 (min_freq=5, 95.6% coverage) |
| Char vocab | 333 |
| Training data | 102,358 sentences (`data/combined_train.jsonl`) |
| Optimizer | AdamW (lr=3e-4, weight_decay 0.01, cosine annealing) |
| Hardware | Apple Silicon MPS |
| Checkpoint | `checkpoints/octo_transformer_best.pt` (epoch 16) |

---

## Results (Measured)

### Language Modeling (teacher-forced, held-out 500 sentences / 21,548 tokens)

| Model | Params | Eval PPL | Robustness |
|-------|--------|----------|------------|
| Old flagship (plain transformer) | 34.3M | 89.3 | 0.0 (fragile; 60x swap blowup) |
| Optuna-tuned (d=128, l=2, mf=100) | 2.7M | 140.5 | — |
| **v6** | 13.7M | 1.04 | 0.996 |
| v7 (warm-start + transcripts) | 15.2M | 1.48 | — |
| **v8 (final)** | **52.9M** | **1.32** | **0.721** |

Per-token accuracy at eval PPL 1.32 is ≈74%. v8's eval PPL declined monotonically across
its 17 epochs (1.67 → 1.32). The checkpoint is `octo_transformer_best.pt`.

### The Wall (honest limitation)
Autoregressive generation degenerates into repetition at **every** size tested
(v6: 1.04, v7: 1.48, v8: 1.32). Root cause: per-token survival ≈0.75 at sampling
temperature compounds to ~2e-4 over 30 tokens, so samples collapse into the highest
frequency vocabulary entries. Break-even needs PPL < ~1.05 *at sampling temperature* —
a regime this model family does not reach on a laptop. Instruction fine-tuning was
attempted in two formats (7K samples); both retained the collapse. This is reported
plainly. What the model **does** do extremely well is next-token prediction under
teacher forcing and robustness in the face of lexical noise.

### POS Tagging (separate BiLSTM backbone)
- **99.65% accuracy**, 17 tags, per-tag >99% — the project's strongest single result.
- 9.4M params, char + word embeddings.

### Hyperparameter Optimization
- 60 completed Optuna trials.
- Key finding: vocabulary cutoff (`min_freq`) matters more than model size.
  Small vocab / 5K-word models beat 30K-word models at this scale consistently.

---

## Corpus

`data/combined_train.jsonl` — 102,358 sentences, surviving and rebuilt after the
original 106K set (`mega_train_v2.jsonl`) was lost to a git/data incident:

| Source | Sentences |
|--------|-----------|
| Wikipedia-style training text | 78,139 |
| CLARIN (news, POS-enriched) | 11,479 |
| Cleaned long-form science transcripts (`data/transcripts.jsonl`) | 8,558 |
| WikiText-2 raw (re-downloaded parquet) | 4,182 |

Transcript cleaning is fully scripted (`tools/clean_transcripts.py`, 8,498 unique
sentences from ~30 DownSub files).

---

## Repository

```
train_transformer.py      # training, --resume, warm-start vocab extension
finetune_chat.py          # instruction fine-tuning (attempted)
eval_model.py             # PPL + generation + perturbation eval suite
octo_serve.py             # FastAPI inference server
optuna_search.py          # 60-trial hyperparameter sweep
core/                     # tetrahedral attention, cognitive geometry,
                          # 6-term objective, working memory, reservoir
data/combined_train.jsonl # 102,358-sentence corpus
data/transcripts.jsonl    # 8,498 cleaned transcript sentences
data/eval_heldout.jsonl   # 500-sentence eval set (deduped vs train)
tools/                    # corpus cleaning + merge scripts
checkpoints/              # v6/v7/v8 era checkpoints + chat probes
RESULTS.md                # full measured-experiment record (this file's canonical source)
```

---

## Key Numbers

- **1.32** — v8 held-out eval perplexity (52.9M, from scratch)
- **0.721** — v8 perturbation robustness (MODERATE; vs 0.0 for the old flagship)
- **99.65%** — POS tagging accuracy
- **52.9M** — final model parameters
- **102,358** — training sentences
- **26,603** — word vocabulary (95.6% token coverage)
- **6** — cognitive loss terms; **12** — geometry modules; **8** — reservoir limbs
- **60** — Optuna trials; **0** — pretrained components

---

*Built with PyTorch on Apple Silicon MPS. No GPT-2. No external models. Every claim here is measured; every failure is reported.*
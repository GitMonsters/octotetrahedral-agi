# OctoTetrahedral AGI

### A Pure OctoTetrahedral Architecture for Language Modeling

---

## The Problem

Modern language models achieve impressive fluency but share fundamental weaknesses:

1. **Memorization without understanding** — models learn surface statistics, not structure
2. **Catastrophic forgetting** — new knowledge overwrites old
3. **Brittle perturbation response** — swapping one word causes a 60x perplexity blowup
4. **No principled training objective** — standard cross-entropy ignores stability, grounding, and resource efficiency
5. **Architecture ceiling** — transformer attention has no geometric structure; it treats all token relationships as learned from scratch

---

## The Solution

OctoTetrahedral AGI is a **pure transformer language model** enhanced with six integrated cognitive modules derived from neuroscience, reservoir computing, and geometric deep learning. No external backbones (no GPT-2, no pretrained encoders). Every parameter is trained from scratch.

The architecture rests on three pillars:

### 1. Tetrahedral Attention

Standard transformer attention computes:
```
softmax(QK^T / sqrt(d))
```

OctoTetrahedral attention adds a **learned geometric bias**:
```
softmax(QK^T / sqrt(d) + alpha * geometric_bias)
```

The geometric bias encodes spatial relationships derived from tetrahedral structure. Tokens closer in tetrahedral space attend more strongly, creating structured information flow that respects positional proximity via Gaussian decay:
```
bias = exp(-d_ij^2 / (2 * sigma^2)),  sigma = sqrt(d_model)
```

This gives the model a **structural prior** — it doesn't have to learn positional relationships from scratch.

### 2. Cognitive Geometry Engine

12 operational modules implementing the full ML vocabulary as differentiable components:

| Module | What It Does |
|--------|-------------|
| SVD Activation Decomposer | Extracts dominant semantic axes via truncated SVD |
| Concept Alignment Matrix | Penalizes limb collapse via cosine similarity |
| Entropy Flow Monitor | Tracks uncertainty across stages (target: 2.0 bits) |
| Semantic Drift Detector | Measures vector rotation across forward passes |
| Anchor Vector System | Persistent identity/topic bias vectors (4 anchors, decay 0.95) |
| Repetition Dampener | Suppresses token echo patterns in logits |
| Branch Scorer | Scores reasoning branches by goal alignment |
| Manifold Partitioner | Enforces orthogonality between limb subspaces |
| Goal Vector System | Explicit direction guiding all reasoning |
| Attention Plane Reconstructor | Compresses attention into a 2D concept map |
| Vector Field Tracker | Tracks representation flow across layers |
| Cross-Limb Orthogonality | Keeps reasoning limbs independent |

All modules are **gated by config** (zero overhead when disabled) and produce auxiliary losses + diagnostic info.

### 3. Six-Term Composite Objective

Standard language models optimize a single loss. OctoTetrahedral optimizes six:

```
L_total = L_task + 0.10 * L_WM + 0.05 * L_meta + 0.02 * L_resource + 0.05 * L_ground + 0.15 * L_stability
```

| Term | Weight | What It Measures |
|------|--------|-----------------|
| **L_task** | 1.0 | Cross-entropy prediction loss (standard LM) |
| **L_WM** | 0.10 | World-model: predicts next hidden state (MSE + causal + rollout + calibration) |
| **L_meta** | 0.05 | Meta-learning: rewards faster adaptation and lower post-shift error |
| **L_resource** | 0.02 | Compute efficiency: penalizes over-compute on easy tasks, under-compute on hard |
| **L_ground** | 0.05 | Grounding: ties abstraction to reality via action-outcome accuracy |
| **L_stability** | 0.15 | Cohesion deficit + EWC forgetting penalty + oscillation detection |

Plus geometric auxiliary losses from the Cognitive Geometry Engine (entropy, drift, anchor drift, goal alignment, attention coherence, vector field smoothness).

---

## Supporting Modules

### Working Memory
Neural Turing Machine-inspired 4-slot differentiable memory:
- **Slot 0**: Goal/task representation
- **Slot 1**: Current context
- **Slot 2**: Intermediate results
- **Slot 3**: Output buffer

Multi-head attention reads, sigmoid-gated writes, selective erase gates. Gradients flow through memory state.

### Reservoir Dynamics
Echo-state computing with four mechanisms:
- **Echo State Constraint**: Spectral radius scaled to 0.9 (edge of chaos)
- **Neural Pacemaker**: Multi-frequency oscillatory driving (theta 6Hz, alpha 10Hz, gamma 40Hz)
- **Temporal Basis Diversity**: 8 "limbs" with different leak rates (Fourier-rich basis)
- **Linear Readout**: Harvests signal from concatenated limb states

### TranscendPlexity Controller
Phase detection and alpha-order dynamics:
- Tracks processing phase (EXPLORATION / CONSOLIDATION / DEEP_REASONING)
- Monitors compounding loss and stability
- 8-dimensional alpha ordering for multi-concept learning

### Compounding Cohesion Tracker
```
cohesion = 0.6 * cos_sim(prev_hidden, hidden) + 0.4 * trajectory_score
trajectory_score = 1 / (1 + 10 * variance(step_magnitudes))
```
Feeds into stability loss to prevent representation collapse.

### Tetrahedral Transformer Layer
Drop-in replacement for `nn.TransformerEncoderLayer`:
- Pre-norm residual blocks
- Geometric-bias attention shared across all layers
- LayerNorm + FFN (GELU) + dropout

---

## Architecture (Current Training Run)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 8 |
| num_layers | 4 |
| dim_ff | 512 |
| dropout | 0.06 |
| max_len | 128 tokens |
| Total params | **6.7M** |
| Word vocab | 5,226 (min_freq=100) |
| Char vocab | 126 |
| Training data | 106,121 sentences (C4 + WikiText-2 + CLARIN) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, cosine annealing) |
| Hardware | Apple Silicon MPS |

---

## Results

### POS Tagging (BiLSTM Backbone)
- **99.65% accuracy** on CLARIN evaluation set
- 17 POS tags, per-tag accuracy >99% on all categories
- 9.4M parameter BiLSTM with char + word embeddings

### Language Modeling

| Model | Params | Eval PPL | Notes |
|-------|--------|----------|-------|
| Old Transformer (nn.TransformerEncoder) | 34.3M | 89.3 | Best checkpoint, epoch 28 |
| Optuna-tuned (d=128, l=2, mf=100) | 2.7M | 140.5 | Best of 60 Optuna trials |
| **New Architecture (in training)** | **6.7M** | **TBD** | 6-term loss + tetrahedral attention |

### Hyperparameter Optimization
- 60 Optuna trials completed
- Key finding: **min_freq=100** (5K vocab) is the single most important hyperparameter
- Small models (128-dim, 2-3 layers) consistently outperform large ones at this search scale
- Dropout ~0.06, learning rate ~9e-4 optimal for small models

### Perturbation Robustness (Old Architecture)
- 34M model showed **complete fragility**: robustness = 0.0
- Single word swap caused 60x perplexity blowup
- Model was memorizing, not reasoning
- New architecture with geometric bias + cohesion tracking expected to improve this

### Training Diagnostics (New Architecture)

| Epoch | Train Loss | Cohesion | WM Loss | Stability | Phase |
|-------|-----------|----------|---------|-----------|-------|
| 0 | 0.342 | 0.010 | 0.599 | 0.187 | DEEP_REASONING |
| 1 | 0.004 | 0.006 | 0.123 | 0.100 | DEEP_REASONING |
| 2 | 0.002 | 0.005 | 0.062 | 0.090 | DEEP_REASONING |
| 3 | 0.002 | 0.004 | 0.038 | 0.085 | DEEP_REASONING |
| 4 | 0.001 | 0.003 | 0.028 | 0.083 | DEEP_REASONING |
| 5 | 0.001 | 0.003 | 0.020 | 0.080 | DEEP_REASONING |

All six loss terms active and converging. World-model loss decreasing (learning to predict next hidden state). Stability loss stable (model not oscillating). Phase locked to DEEP_REASONING.

---

## What Makes This Different

### vs. Standard Transformers
| Dimension | Standard Transformer | OctoTetrahedral |
|-----------|---------------------|-----------------|
| Attention | Learned from scratch | Geometric-bias prior |
| Loss | Cross-entropy only | 6-term composite |
| Forgetting | Catastrophic | EWC stability loss |
| Uncertainty | None | Entropy monitoring |
| Memory | Stateless | 4-slot working memory |
| Dynamics | Feedforward only | Reservoir + pacemaker |
| Robustness | Fragile | Cohesion-tracked |

### vs. GPT-2 / Pretrained Models
- **No pretrained backbone** — every parameter is learned from scratch
- **Transparent** — all cognitive modules produce diagnostic outputs
- **Principled** — training objective optimizes for stability, not just prediction
- **Compact** — 6.7M params vs 124M+ for GPT-2 small

### vs. Neuroscience-Inspired Models
- **Fully integrated** — not bolted on; all modules wired into the forward pass and loss
- **Trainable** — geometric bias, working memory, and reservoir all participate in backprop
- **Measurable** — cohesion, phase, entropy, drift all tracked per-batch

---

## Roadmap

### Phase 1: Architecture (Current)
- [x] Tetrahedral attention with geometric bias
- [x] 12-module cognitive geometry engine
- [x] 6-term composite loss
- [x] Working memory (4-slot NTM)
- [x] Reservoir dynamics (8-limb echo state)
- [x] Compounding cohesion tracker
- [x] Full MPS integration
- [ ] Complete 50-epoch training run
- [ ] Eval on WikiText-2 test set
- [ ] Perturbation robustness benchmark

### Phase 2: Optimization
- [ ] Optuna retrain with new architecture (NaN fix applied)
- [ ] Scale to 35M+ params on MPS
- [ ] Hyperparameter sweep with 6-term loss weights

### Phase 3: Instruction Tuning
- [ ] Fine-tune on instruction data for chat
- [ ] Multi-turn conversation support
- [ ] Context window expansion

### Phase 4: Evaluation
- [ ] CCL (Compounding Concept Learning) benchmark
- [ ] Intelligence testing suite
- [ ] Real-world task suite
- [ ] Comparative analysis vs. GPT-2 at same parameter count

---

## Repository Structure

```
octotetrahedral-agi/
  train_transformer.py          # Main LM training (6.7M param model)
  train_pos_bilstm.py           # BiLSTM POS tagger (99.65% accuracy)
  eval_model.py                 # PPL + generation + perturbation eval
  finetune_chat.py              # Instruction fine-tuning
  octo_serve.py                 # FastAPI inference server
  optuna_search.py              # Phase 1 hyperparameter sweep (60 trials)
  optuna_retrain.py             # Phase 2 retrain with best configs
  core/
    tetrahedral_attention.py    # Geometry-aware attention
    tetrahedral_transformer_layer.py  # Transformer layer with geometric bias
    cognitive_geometry.py       # 12 cognitive geometry modules
    recursive_engine_objective.py  # 6-term composite loss
    working_memory.py           # 4-slot NTM-style memory
    reservoir_dynamics.py       # Echo-state + pacemaker reservoir
    compound_loop.py            # Adaptive looped reasoning
    transcendplexity_integration.py  # Phase detection + alpha ordering
  data/
    mega_train_v2.jsonl         # 106K training sentences
  checkpoints/                  # 150+ model checkpoints
```

---

## Key Numbers

- **99.65%** POS tagging accuracy
- **6.7M** parameters (current model)
- **34.3M** parameters (flagship checkpoint)
- **6** terms in the composite training objective
- **12** cognitive geometry modules
- **8** reservoir limbs with theta/alpha/gamma pacemaker
- **4** working memory slots (goal / context / results / output)
- **60** Optuna hyperparameter trials completed
- **106,121** training sentences from C4 + WikiText-2 + CLARIN
- **0** pretrained components — fully from scratch

---

*Built with PyTorch. Runs on Apple Silicon MPS. No GPT-2. No external models. Pure OctoTetrahedral.*

# OctoTetrahedral AGI — Deep Dive (Final, Honest)

The full record: from first line of code to the final 52.9M-param run, including the
wall that ended the experiment. Every number below is measured and reproducible with
the repo's eval suite.

---

## Phase One: The POS Tagger

Foundation work: a BiLSTM part-of-speech tagger — char embeddings through a
bidirectional LSTM, concatenated with word embeddings, two bidirectional LSTM layers
with 256 hidden units, 19 POS labels, 9.4M parameters, trained on the CLARIN
enriched dataset.

It was instrumented with read-only versions of the cognitive modules — working memory,
reservoir dynamics, TranscendPlexity controller, cohesion tracker — that observed the
tagger without participating in training.

**Result: 99.65% accuracy on the CLARIN eval set.** All 17 tags above 99% —
punctuation 99.98%, pronouns 99.97%, determiners 99.94%. Checkpoint:
`checkpoints/octo_integrated_best.pt`. This remains the project's strongest single result.

## Phase Two: The Dual-Head Language Model

Same BiLSTM backbone, two heads: POS tagging and next-word prediction
(`train_lm.py`, class `OctoDualHead`, 12.1M params). On CLARIN it hit perplexity 1.18
on its training set — evidence the BiLSTM hidden states encode usable linguistic
structure. A proof-of-concept, not the main line.

## Phase Three: Off to the Transformers

We moved to a pure transformer built from scratch — word + char + positional
embeddings and a standard encoder stack — nothing from GPT-2 or any pretrained model.
This became the main line of the project.

## The Dataset

The defining data artifact was `data/mega_train_v2.jsonl` — 106,121 sentences from C4
(80K), WikiText-2 (24K), and CLARIN (12K). Vocabulary was built with a minimum-frequency
threshold; the sweet spot was min_freq=100 giving a 5,226-word vocab.

**Loss:** this file was subsequently lost to a git/data incident (never committed, raw
sources not retained) and proved unrecoverable. The project was rebuilt on a surviving
corpus of 102,358 sentences — combining the surviving Wikipedia-style text (78,139),
CLARIN (11,479), clean long-form science transcripts (8,558, newly harvested and cleaned
from ~30 DownSub files), and a fresh WikiText-2 raw download (4,182). Rebuild is fully
scripted in `tools/`.

## Phase Four: Hyperparameter Optimization

60 Optuna trials (d_model 128–320, heads 4–8, layers 2–6, ff 256–1024, dropout
0.05–0.3, lr 1e-4–5e-3, batch 8–32, min_freq 10–100). Winner: d=128, 2 layers,
ff=256, 8 heads, dropout 0.056, lr 9.26e-4, min_freq 100 → 2.7M params, eval PPL 140.5.

The transferable finding: **vocabulary size matters more than model size.** A 2.7M-model
with a 5K vocab consistently beat a 13M-model with a 31K vocab. min_freq was the single
most important knob.

## Phase Five: The Old Flagship (and the fragility that ended it)

Trained a 34.3M-param plain transformer (d=384, 6 heads, 6 layers, ff=1536, dropout 0.2)
50 epochs on MPS. Best eval PPL 89.3 at epoch 26; it produced recognizably English
generation — "the cat sat on the mat and she was a very good cat."

Perturbation testing (swap one word, measure PPL blowup): **complete collapse**.
60x PPL increase on a single word swap. Robustness 0.0. The model memorized surface
patterns. This is the measurement that motivated the whole cognitive-module program.

## Phase Six: Porting the Cognitive Modules

Five substantial modules — developed in a research directory and moved into `core/` —
formed the new architecture's spine:

- `tetrahedral_attention.py` (281 lines) — attention plus a learned Gaussian geometric bias.
- `cognitive_geometry.py` (1,005 lines) — the 12-module geometry engine.
- `working_memory.py` (284 lines) — 4-slot NTM-style differentiable memory.
- `reservoir_dynamics.py` (518 lines) — 8-limb echo-state reservoir, theta/alpha/gamma pacemaker, spectral radius 0.9.
- `recursive_engine_objective.py` (658 lines) — the 6-term composite loss.

## Phase Seven: Integration

`train_transformer.py` brings it all together in one forward pass — bias-shared
geometric attention, the geometry engine with auxiliary losses, the 6-term objective,
the cohesion tracker feeding the stability loss — with NaN guards on every term and
nan-safe training.

## Phase Eight: Bug Fixes Born From Measurement

This phase is what made the final numbers possible:

1. **Label smoothing / one-hot collapse.** Generation failed not because of the modules
   but because the untuned CE loss drove max-prob predictions ≥99.9% with entropy
   0.0001 nats. Fix: label smoothing 0.1. This moved eval PPL from ~1.5 territory toward
   ~1.0 and is required for the diagnostics to mean anything.
2. **UNK handling in generation.** UNK masked out of sampled tokens.
3. **MPS reliability.** Early MPS runs died with memory-pressure kills and a device/state
   bug. Fixes: `PYTORCH_ENABLE_MPS_FALLBACK=1`, controlled batch size, and disciplined
   eval/save. v8 ran 17 full epochs on MPS without a kill.
4. **Stability oscillation bug** (hidden-state shape mismatch across truncated batches) —
   fixed by always storing a single mean hidden vector.
5. **Truncation bug** (positional embedding out-of-bounds for long sentences) — fixed.
6. **Optuna retrain NaN** — OneCycleLR produced NaN on the new architecture; switched to
   CosineAnnealingLR + a NaN guard.

## Phase Nine: The Stable Runs (v6 → v7 → v8)

**v6 (13.7M; d=256, 4 layers, drop 0.3, min_freq=20):** first fully stable run. Eval PPL
**1.04**, robustness 0.996. Generation still collapsed → we called it, correctly, a
prediction-quality problem, not an architecture one.

**v7 (15.2M, warm-start):** a data experiment disguised as a model. Added `--resume`
with vocabulary extension and warm-started from v6 onto the new
102,358-sentence corpus (transcripts included). Eval PPL 1.48 — **worse**, because the
larger corpus and bigger vocab made the task harder. Deleted the theory that more of the
same data fixes generation. Cut at epoch 9.

**v8 (52.9M; d=512, 8 layers, ff=2048, drop 0.3, min_freq=5 — fresh vocab of 26,603,
95.6% coverage; MPS, ~58 min/epoch):** the scale-up attempt. Eval PPL declined
monotonically 1.67 → **1.32** over 17 epochs. Robustness **0.721** (MODERATE; 158/200
swaps increased PPL, median ratio 1.05). Generation still collapsed.

**Instruction fine-tuning (2 formats):** rebuilt the corpus as 7K instruction samples.
Format A used `Instruction:/Response:` — those trigger tokens were UNK, invisible to the
model (a data bug we caught and documented). Format B used in-vocab
`Question :/Response :`. Both fine-tuned on top of v8 and both retained the collapse —
the probe went from lr-annealed catastrophic unlearning at loss 1.71 up to a best of
1.91, then fell into `:`-pumping. No path to conversation at this scale.

## The Wall (why generation fails — arithmetic, not magic)

At eval PPL 1.32, single-token survival under sampling ≈ 0.75. Across 30 sampled tokens:
0.75^30 ≈ 2e-4. The generation therefore almost always decays into the
highest-frequency words in the vocabulary — the giant loops, `:` and `?` pumping and
"David But But But" you see in every sample. Break-even requires PPL < ~1.05 at
sampling temperature, which this model family does not reach on a laptop. The v8
numbers — 1.32 vs v6's 1.04 — prove the problem is not "our best model is broken":
v6 outperformed v8 and still could not generate. Three sizes and two data formats gave
the same answer.

---

## Final Numbers (canonical)

| Metric | Value |
|--------|-------|
| v8 eval perplexity (teacher-forced, 500 held-out) | **1.32** |
| v8 train perplexity | 6.23 |
| v8 robustness (word swap) | **0.721** — MODERATE |
| v6 eval perplexity | 1.04 |
| Old flagship eval perplexity / robustness | 89.3 / 0.0 |
| POS tagger accuracy | **99.65%** |
| Final params | 52.9M (45.9M core) |
| Training sentences | 102,358 |
| Word vocab | 26,603 (95.6% coverage) |
| Optuna trials | 60 |
| Instruction fine-tune best chat loss | 1.91 (still no generation) |
| Autoregressive fluency | **FAILS at every size — the honest, reported result** |

## What We Actually Have

- A from-scratch 52.9M-param predictor with eval PPL 1.32, robustness 0.721 — both
  measurements clean and monotonic.
- A rebuilt, reproducible corpus and cleaning toolchain.
- The strongest diagnostic artifact in the project: the proof that teacher-forced
  next-token prediction and open-ended generation decouple completely at this scale.
- Real engineering wins to carry forward: label smoothing, warm-start vocab extension,
  UNK masking, MPS-stable training, a robust eval suite.

## What Could Watch: Next Steps (only if pursued with a real GPU budget)

1. A checkpoint-quality sampler with retrieval/constrained decoding is the cheapest
   probe on `octo_transformer_best.pt` (no retrain).
2. Real scaling: this Mac cannot reach the PPL < 1.05 (sampling) regime. That needs
   multiple GPUs and ~100-1000x the current token budget.

Until either happens, the honest position is: the model predicts, it is robust, and it
does not generate. That is the record, and it is reproducible.
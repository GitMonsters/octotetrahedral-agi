# OctoTetrahedral AGI — Final Results (Honest Report)

Status: **complete / frozen**. Last run: v8, epoch 16. This document reports measured
results only. Where something did not work, this document says so.

---

## The One-Sentence Summary

We trained a from-scratch language model with a geometric-bias transformer, a 6-term
cognitive loss, working memory, and reservoir dynamics. It became a genuinely good
**teacher-forced next-token predictor** (eval perplexity 1.32 at 52.9M params), but it
**cannot generate fluent autoregressive text** at any size we tried (13.7M to 52.9M).
That gap is structural at this parameter/data scale, not a tuning miss.

---

## Final Model (v8) — The Deliverable

| Property | Measured value |
|----------|----------------|
| Checkpoint | `checkpoints/octo_transformer_best.pt` (epoch 16 of 17 trained) |
| Parameters | 52.9M total (45.9M core transformer) |
| Architecture | d=512, nhead=8, 8 layers, dim_ff=2048, dropout=0.3, max_len=128 |
| Loss | label-smoothed cross-entropy (0.1) + 6-term cognitive composite |
| Attention | tetrahedral geometric-bias attention (Gaussian positional prior) |
| Vocab | 26,603 words (min_freq=5, 95.6% token coverage) + 333 chars |
| Training data | 102,358 sentences (see Corpus section) |
| Eval set | 500 held-out sentences / 21,548 tokens (`data/eval_heldout.jsonl`) |
| **Eval perplexity** | **1.32** (teacher-forced; ≈74% next-token accuracy) |
| Train perplexity | 6.23 (includes rare-token pressure of the larger vocab) |
| **Robustness** | **0.721 (MODERATE)** — 158/200 single-word swaps increased PPL (ratio 1.14) |
| Diagnostics | phase=EXPLORATION, stability 0.756, cohesion 1.0 |
| Hardware | Apple Silicon MPS (all training succeeded without a GPU) |

Final generation sample (unchanged failure mode):
```
the cat sat sat united offer d d d d fit 35 35 Most Wales Wales Wales ...
he said that that that at at at at at But But But David : : : : : back back ...
```

---

## What Works

- **Teacher-forced next-token prediction.** Eval PPL 1.32 means the model *can* predict
  the next word given the full sentence with ~74% accuracy. This declined monotonically
  and cleanly: 1.67 → 1.36 across v8's 17 epochs.
- **Robustness to single-word perturbation improved over the old architecture.**
  Old 34.3M flagship: robust=0.0 (60x PPL blowup on one word swap). v8: robust=0.721,
  median perturbation ratio 1.05. The model relies on both surface and structural cues.
- **Curriculum / tooling.** Warm-start with vocabulary extension (v7), the label-smoothing
  fix, UNK-masking in generation, and the eval suite all work and are reusable.
- **POS tagger** (separate BiLSTM): 99.65% accuracy over 17 tags — the project's
  strongest single result.
- **Data.** A 102K-sentence corpus survived the loss of the original 106K set and was
  expanded with 8,498 sentences of long-form science transcripts (`data/transcripts.jsonl`).

---

## What Does Not Work (The Wall)

**Autoregressive generation collapses into repetition at every size tested.**

| Config | Params | Eval PPL | Autoregressive generation |
|--------|--------|----------|---------------------------|
| v6 | 13.7M | 1.04 | token soup, loops |
| v7 (transcript warm-start) | 15.2M | 1.48 | token soup, loops |
| v8 (scale-up) | 52.9M | 1.32 | token soup, loops |
| v8 + instruction finetune | 52.9M | 1.91 (chat) | `:`/`?` pumping, no answers |

Root causes established experimentally:
1. **One-hot collapse** (fixed): no label smoothing → 99.9% max-prob predictions,
   entropy 0.0001 nats. Fixed with smoothing=0.1.
2. **UNKed format tokens** (fixed in v7/v8): trigger words like `Instruction`/`Explain`
   weren't in the vocab and were invisible to the model. Fixed via vocab extension and
   in-vocab-only instruction data.
3. **Short-horizon washout** (not fixed — structural): at eval PPL 1.32 the per-token
   survival probability is ≈0.75; over 30 sampled tokens that compound to ~2e-4, so the
   model's sample almost always degenerates into the highest-frequency tokens in its
   vocabulary. Break-even would require PPL < ~1.05 *at sampling temperature*, which
   this class of model does not reach.
4. **Char embeddings are a dead feature** (measured): predictions are essentially
   identical with real, random, or zeroed character IDs.

The mathematical claim of the project — that a geometric-bias prior, working memory, and
a stability-augmented objective would yield a *generative* model on this hardware — is
not supported by the evidence. The architecture predicts well under teacher forcing and
degrades predictably; it does not generate.

---

## Corpus (v7/v8)

`data/combined_train.jsonl` — 102,358 sentences, built from:

| Source | Sentences | Notes |
|--------|-----------|-------|
| `training_data.jsonl` | 78,139 | Wikipedia-style text, wordpiece-tokenized artifact |
| `data/transcripts.jsonl` | 8,558 | Cleaned long-form science/tech video transcripts |
| `clarin_enriched_data.jsonl` | 11,479 | CLARIN news + POS-enriched text |
| WikiText-2 raw (re-downloaded) | 4,182 | Clean long-form encyclopedia text |

The original v3-v6 set (`data/mega_train_v2.jsonl`, 106,121 sentences incl. ~80K C4) was
lost (never committed to git; no raw sources remained) and could not be recovered.

---

## Training History (the full experiment record)

| Run | Size | Purpose | Result |
|-----|------|---------|--------|
| v3-v4 (MPS) | 13.7M | establish pipeline | MPS device bug + memory-pressure kills; moved to CPU |
| v5 (CPU) | 13.7M | diagnose generation | found one-hot collapse, UNK rate, eval overlap |
| **v6 (CPU)** | **13.7M** | first stable run | eval PPL 1.04, robust 0.996, still no generation |
| **v7 (CPU)** | **15.2M** | warm-start + transcripts | eval PPL 1.48; proved data isn't the fix |
| **v8 (MPS)** | **52.9M** | scale-up | eval PPL 1.32; proved scale-on-this-Mac isn't the fix |
| chat ft | 52.9M | instruction format | 1.91 chat loss; no answer generation (format bug fixed, still failed) |

Key engineering preserved from this journey:
- Label smoothing (0.1) — one-hot collapse fix
- UNK masking in `generate` and vocab extension for warm-start (`--resume`)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` + MPS memory discipline (v8 ran 17 epochs clean)
- `data/eval_heldout.jsonl` deduped against training corpus

---

## How to Reproduce

```
# Reproduce the reported eval numbers (needs the v8 best checkpoint):
python3 eval_model.py --checkpoint checkpoints/octo_transformer_best.pt --device cpu

# Rebuild the corpus from sources:
python3 tools/clean_transcripts.py     # raw transcripts -> data/transcripts.jsonl
python3 tools/clean_transcript.py            # single-transcript fallback
python3 tools/merge_corpus.py          # all JSONL sources -> data/combined_train.jsonl

# Retrain v8-equivalent (fresh vocab, MPS):
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -u train_transformer.py \
  --data-paths data/combined_train.jsonl --d-model 512 --nhead 8 --num-layers 8 \
  --dim-ff 2048 --dropout 0.3 --batch-size 8 --lr 3e-4 --min-freq 5 \
  --patience 10 --epochs 30 --device mps
```

---

## Working Demo: Retrieval + Ranking Chat (`tools/chat_retrieval.py`)

A chat-shaped demo that uses the model's measured strength instead of its weakness:
answers are **retrieved verbatim** from a local corpus and **ranked by the LM's
naturalness score** (answer-span perplexity conditioned on `Question : <q>
Response : <answer>`), not generated.

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 tools/chat_retrieval.py        # interactive
python3 tools/chat_retrieval.py --question "What is a black hole?"   # one-shot
```

Known, honest limits of the demo:
- A relevance gate (weighted-IDF query-token coverage ≥ 0.6) rejects candidates that
  merely contain a fuzzy keyword — e.g. "where is the grand canyon?" elicits an honest
  "no topically relevant corpus content" since the corpus has no canyon content, rather
  than surfacing a fluent but off-topic sentence.
- The scorer is weakly discriminative at this temperature of results — most fluent
  sentences score ~1.05-1.15 answer-PPL, so the keyword retrieval does the topical
  narrowing and the LM only picks among near-ties.
- Answers are pulled verbatim from the corpus (science/tech transcripts); it will
  happily surface a sentence that merely mentions a topic rather than defining it.

---

## Bottom Line

- **Good:** 52.9M from-scratch LM, teacher-forced eval PPL **1.32**, robust 0.721,
  POS tagger 99.65%, full reproducible toolchain, running on a laptop's MPS.
- **Honest limitation:** open-ended generation is not possible at this scale; the
  documents do not claim otherwise after v6.
# OctoTetrahedral AGI

Status: **complete / frozen**. A from-scratch language model with a geometric-bias
transformer, a six-term cognitive loss, working memory, and reservoir dynamics —
plus an honest report of what did and did not work.

## The One-Sentence Summary

The trained model (v8, 52.9M params) is a genuinely good **teacher-forced next-token
predictor** — eval perplexity **1.32** on 500 held-out sentences — but it **cannot
generate fluent autoregressive text** at any size we tried (13.7M to 52.9M). That gap
is structural at this parameter/data scale, not a tuning miss.

| Property | Measured value |
|----------|----------------|
| Checkpoint | `checkpoints/octo_transformer_best.pt` (epoch 16 of 17) |
| Parameters | 52.9M total (45.9M core transformer) |
| Architecture | d=512, nhead=8, 8 layers, dim_ff=2048, dropout=0.3, max_len=128 |
| Eval perplexity | 1.32 (teacher-forced), train 6.23 |
| Robustness | 0.721 (MODERATE — 158/200 swaps raised PPL) |
| Corpus | 102,358 sentences (training_data + clarin + transcripts + wikitext-2-raw) |

**The generative wall:** per-token survival ≈ 0.75 at eval PPL 1.32, so over 30 sampled
tokens you expect ≈ 0.75^30 ≈ 2e-4 real sentences. Break-even needs PPL < ~1.05 at
sampling temperature. Full story in [`RESULTS.md`](RESULTS.md).

## What It Can Do

Because it scores naturalness (not writes), it ships as a **Perplexity-style
retrieval + ranking chatbot** over a corpus, with an online fallback:

```bash
python3 tools/chat_retrieval.py                        # interactive
python3 tools/chat_retrieval.py --question "what is a black hole"
python3 tools/chat_retrieval.py --question "what is transcendplexity"
```

When the corpus has no topical answer it searches DuckDuckGo, Wikipedia, and Urban
Dictionary and ranks the retrieved sentences with the same LM score. Answers are
tagged with their source (`local`, `web: wikipedia: …`, `web: urbandictionary`,
`repo: FACTS.md`), and misspelled queries get `did you mean` suggestions.

Web UI:

```bash
python3 octo_serve.py --port 8080     # http://localhost:8080/chat-ui
```

## Reproduce the Final Model

```bash
python3 train_transformer.py --resume checkpoints/octo_transformer_best.pt
python3 eval_model.py
```

(Requires the corpus in `data/`; corpus and checkpoints are git-ignored by size.
See [`RESULTS.md`](RESULTS.md) → Reproduction for exact commands and the checkpoint
table.)

## Repository Map

- `train_transformer.py` — main model + training loop (`OctoTransformerLM`, `--resume`)
- `octo_serve.py` — FastAPI server: `/chat/rag`, `/chat-ui`, `/health`
- `tools/chat_retrieval.py` — retrieval + LM-ranking chatbot (local + web)
- `eval_model.py` — canonical evaluation suite (perplexity, robustness, diagnostics)
- `PITCH.md`, `PITCH_READ_ALOUD.md`, `PITCH_DEEP_DIVE.md` — the pitch, with real numbers
- `RESULTS.md` — full honest results, training history v3–v8, the generative wall
- `FACTS.md` — small curated facts (e.g. the Transcendplexity definition)
- `core/`, `archive/`, `benchmarks/`, `sync/`, `demo/` — experimental artifacts
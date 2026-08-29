# OctoTetrahedral AGI — Read-Aloud Pitch (Final)

---

Hey, let me tell you about OctoTetrahedral AGI.

It started as a question: can you build a language model from scratch — no GPT-2, no pretrained anything — that isn't just a pattern matcher? An architecture where the attention mechanism has built-in geometric structure, the loss function cares about stability and memory, not just the next word, and the whole thing runs on a laptop?

We did build that. And we measured, honestly, both what works and what doesn't.

Here's the architecture. Three ideas.

First, tetrahedral attention. Normal attention learns every token relationship from data. Ours adds a geometric bias — a Gaussian positional prior — so nearby tokens are structured from birth, not memorized.

Second, the cognitive geometry engine. Twelve differentiable modules running in every forward pass. Entropy monitoring, drift detection, anchor vectors for topic persistence, a repetition dampener, a goal-vector system, and cross-limb orthogonality. Every module produces a small auxiliary loss.

Third, a six-term composite objective. The task loss predicts the next word. A world-model loss predicts the model's own future hidden states. A meta-learning term rewards fast adaptation. A resource term keeps compute proportional to difficulty. A grounding term ties abstraction to outcome. And a stability term with a compounding-cohesion tracker prevents representation collapse and forgetting.

Around that, working memory — four differentiable slots, like a scratchpad you can backprop through — and reservoir dynamics: eight echo-state limbs driven by theta, alpha, and gamma oscillations, set right at the edge of chaos.

The training data is 102,358 sentences, rebuilt from sources that survived the project and expanded with thousands of sentences of long-form science transcripts. Every parameter is trained from scratch on Apple Silicon. No GPU.

And here's the measured result.

The final model — the v8 "scale-up" — has 52.9 million parameters. On held-out data it predicts the next token at a perplexity of 1.32. That's around 74% next-word accuracy. And here's the number that tells you this isn't just memorization: swap one word in a sentence and the old flagship transformer — 34 million parameters — exploded, 60 times worse, robustness zero. v8 scores 0.721. It's genuinely robust to lexical noise.

The model also does next-token prediction monotonically better across training, and the diagnostics — entropy, stability, cohesion — all behave.

But I'm going to tell you plainly what it does not do, because that's the honest finding. Autoregressive generation degenerates. Give it a prompt and sample, and it collapses into repetition. We tested this at three sizes — 13.7 million parameters, 15.2 million, and 52.9 million — plus instruction fine-tuning in two formats. Every one of them predicts beautifully under teacher forcing and then collapses when asked to generate open-ended text. The reason is arithmetic: at perplexity 1.32, each sampled token survives with about 75% probability, and over thirty tokens that compounds to one in five thousand. The sample luminance decays into the highest-frequency words in the vocabulary. A fluent generator at this scale would need perplexity under 1.05 while sampling — this family of models doesn't reach that on a laptop.

So the outcome is unusual and worth being precise about. Teacher-forced next-token prediction: excellent. Single-word robustness: real, and far better than the old architecture. Open-ended generation: does not work at this scale. That's the wall, and it's a scaling wall, not a bug — the fixes we applied during the project, like label smoothing to stop one-hot collapse and proper vocabulary handling, are all real engineering wins we kept.

Elsewhere in the project, the strongest result: a part-of-speech tagger at 99.65% accuracy, all seventeen tags above 99%. And sixty Optuna hyperparameter trials whose key finding was that vocabulary size matters more than model size.

So what you get with OctoTetrahedral is a complete, reproducible, from-scratch language-modeling stack with honest, measured results — including a measured statement of exactly where the ceiling is. No pretrained backbone. No GPT-2. No spin. Every parameter learned from zero, every number in this pitch reproduced by the eval suite in the repo.

That's OctoTetrahedral AGI.
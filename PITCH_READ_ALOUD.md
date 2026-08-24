# OctoTetrahedral AGI — Read-Aloud Pitch

---

Hey, let me tell you about OctoTetrahedral AGI.

It's a language model — a system that reads text and predicts what comes next. But it's not built like GPT or any of the usual models. It's built from scratch with six cognitive modules inspired by how brains actually work.

Here's the problem it solves. Modern language models are impressive, but they have real weaknesses. They memorize patterns without understanding structure. They forget everything when you teach them something new. If you swap one word in a sentence, their performance collapses — sometimes 60 times worse. And their training is simple: just predict the next word. Nothing about stability, nothing about efficiency, nothing about grounding.

OctoTetrahedral fixes this with three big ideas.

First, tetrahedral attention. Normal attention is just math between every pair of tokens — it has to learn everything from data. We add a geometric bias that encodes spatial structure directly into the attention mechanism. Tokens that are closer together in a tetrahedral space attend to each other more strongly. This gives the model a built-in sense of structure instead of learning it from scratch.

Second, the cognitive geometry engine. That's 12 different modules running inside the model during every forward pass. One tracks entropy to keep uncertainty in a healthy range. Another detects when the model's representations are drifting. There are anchor vectors that give the model persistent memory of topics, a goal vector system that guides reasoning direction, and a repetition dampener that stops the model from getting stuck in loops. All of these produce small auxiliary losses that train the model to be more stable and more structured.

Third — and this is the big one — a six-term composite training objective. Instead of just "predict the next word," the model optimizes six losses simultaneously. The main task loss does the prediction. A world-model loss teaches the model to predict its own future hidden states, which prevents superficial pattern matching. A meta-learning loss rewards faster adaptation. A resource loss keeps compute proportional to difficulty. A grounding loss ties abstract reasoning to concrete outcomes. And a stability loss prevents catastrophic forgetting and representation collapse.

The model also has working memory — four differentiable slots, like a small scratchpad — and a reservoir computing module with eight parallel limbs driven by neural oscillations at theta, alpha, and gamma frequencies. These give the model temporal dynamics that a standard feedforward network just doesn't have.

Everything runs on Apple Silicon, no GPU required. The current training run is 6.7 million parameters — that's 20 times smaller than GPT-2 small — and it's trained on 106,000 sentences from C4, WikiText-2, and CLARIN.

We also have a separate POS tagger that gets 99.65% accuracy. And we ran 60 Optuna hyperparameter trials to find the best settings.

The key insight from those trials: vocabulary size matters more than model size. A small model with a 5,000-word vocabulary outperforms a large model with 30,000 words every time.

Right now the model is training. It's on epoch 6 of 50. All six loss terms are active and converging. The world-model loss is dropping — it's learning to predict its own future. The stability loss is holding steady — the model isn't oscillating. And it's locked into the "deep reasoning" phase.

Once training finishes, we'll evaluate on WikiText-2 for real perplexity numbers, test perturbation robustness to see if the model actually understands structure or just memorizes, and compare against the old architecture.

The old architecture had 34 million parameters and got a perplexity of 89. The new one should do better with a fraction of the parameters, because it's not just memorizing — it's reasoning with structure.

That's OctoTetrahedral. No pretrained backbone. No GPT-2. No external models. Every parameter learned from scratch with a principled training objective that optimizes for stability, not just prediction.

z# ML Vocabulary Reference — OctoTetrahedral Theory Building Blocks

*Extracted from gptINFO.pdf. Real ML terms with real definitions. Use these to build our own theory.*

---

## Core Linear Algebra

**SVD (Singular Value Decomposition)**
Decomposes any matrix A into `A = UΣV^T`. U = left singular vectors (row patterns), Σ = diagonal of singular values (importance weights), V = right singular vectors (column patterns). The ranked singular values tell you which directions carry the most information. Used to find dominant semantic axes in activation space.

**PCA (Principal Component Analysis)**
Finds the directions of maximum variance in data. Equivalent to SVD on centered data. Reduces high-dimensional activations to their most meaningful axes. First principal component = strongest signal, each subsequent one captures less.

**Eigenvalue / Eigenvector**
For a matrix A, if `Av = λv`, then v is an eigenvector and λ is its eigenvalue. Eigenvectors are the "natural directions" of a transformation — they don't change direction, only scale. In attention matrices, dominant eigenvectors reveal what the model is structurally focused on.

**Orthogonal**
Two vectors are orthogonal when their dot product = 0 (90° apart). Orthogonal vectors carry completely independent information. In reasoning, orthogonal branches mean zero contamination between them.

**Projection**
Casting a vector onto a subspace. `proj_u(v) = (v·u / u·u) * u`. Used everywhere — projecting activations onto concept directions, projecting hidden states onto safety subspaces, projecting tokens onto semantic axes.

**Subspace**
A lower-dimensional slice of the full vector space. A "safety subspace" or "concept subspace" is a region defined by a few key directions within the full high-dimensional space.

**Rotation**
A transformation that changes direction without changing magnitude. In vector space, concept drift = rotation of the meaning vector. Small rotation = topic stays similar. Large rotation = topic has shifted.

---

## Transformer Architecture

**Transformer**
The architecture behind all modern LLMs. Input tokens → embeddings → repeated blocks of (self-attention + feed-forward) → output logits. Each block refines the representation.

**Token**
The atomic unit of text processing. Words get split into tokens (subwords). Each token becomes a vector that flows through the network. "understanding" might be 1-2 tokens depending on the tokenizer.

**Embedding**
The initial vector representation of a token. Maps discrete symbols into continuous vector space where similar meanings are nearby. Dimension typically 768–12288.

**Hidden State**
The vector representation of a token at any layer in the network. Starts as the embedding, gets transformed layer by layer. By the final layer, it encodes the model's full understanding for predicting the next token.

**Residual Stream**
The main "highway" through the transformer. Each layer reads from and writes to the residual stream additively: `H_new = H_old + attention_output + FFN_output`. Information persists across layers via this stream.

**Layer Norm (Layer Normalization)**
Normalizes activations to have zero mean and unit variance within each layer. Stabilizes training. Applied before attention and feed-forward blocks in modern transformers.

**Feed-Forward Network (FFN)**
The MLP block in each transformer layer. Two linear transformations with a nonlinearity: `FFN(x) = W₂ · ReLU(W₁x + b₁) + b₂`. Believed to store factual knowledge.

**Weight Matrix**
The learned parameters. Attention has Q/K/V/O weight matrices. FFN has up/down projection matrices. These are what training optimizes. Billions of parameters = billions of entries across all weight matrices.

**Logit**
The raw output score for each vocabulary token before softmax. Higher logit = model thinks that token is more likely next. The logit vector has one entry per vocabulary token (~100K+).

**Softmax**
Converts logits to probabilities: `softmax(z_i) = e^(z_i) / Σe^(z_j)`. Ensures all outputs sum to 1. Temperature parameter controls sharpness — low temp = more confident, high temp = more random.

---

## Attention Mechanisms

**Self-Attention**
Each token attends to all other tokens to determine relevance. Computes `Attention(Q,K,V) = softmax(QK^T/√d) · V`. Q = what am I looking for, K = what do I contain, V = what do I offer.

**Multi-Head Attention**
Runs multiple attention operations in parallel, each with its own Q/K/V projections. Different heads can specialize — some track syntax, some track semantics, some track position. Outputs are concatenated and projected.

**Attention Head**
One individual attention mechanism within multi-head attention. Each head has its own learned Q/K/V weight matrices. Individual heads develop specialized roles during training.

**Induction Head**
A specific attention pattern (discovered by Anthropic) that copies patterns: if "A B ... A" appears, an induction head predicts "B" will follow the second "A". Believed to be fundamental to in-context learning. Typically forms across two attention layers working together.

**Cross-Attention**
Attention where Q comes from one sequence and K,V from another. Used in encoder-decoder models and multimodal systems (e.g., text attending to image patches).

**Attention Gating**
Mechanism that dynamically controls which attention heads or layers contribute to the output. Allows conditional activation — certain heads fire only under specific conditions (ambiguity, safety concern, etc.).

---

## Vector Space Geometry

**Latent Space**
The high-dimensional vector space where the model's internal representations live. Not directly interpretable, but encodes all learned concepts, relationships, and patterns. Typically 256–12288 dimensions.

**Activation**
The output of a neuron or layer after applying weights and nonlinearity. The pattern of activations across neurons encodes what the model "thinks" at that point.

**Vector Space**
A mathematical space where vectors (lists of numbers) can be added and scaled. The model's latent space is a vector space where directions correspond to concepts and distances correspond to similarity.

**Cosine Similarity**
Measures angle between two vectors: `cos(θ) = (A·B) / (||A|| ||B||)`. Range: -1 (opposite) to +1 (identical direction). The standard measure of semantic similarity in embedding space.

**Vector Field**
Assignment of a vector to every point in space. Visualizes how representations flow or transform. Token trajectories through layers form a vector field — showing how meaning evolves.

**Manifold**
A curved surface embedded in higher-dimensional space. Real data often lives on lower-dimensional manifolds within the full latent space. "The manifold of English text" is a tiny curved surface in the vast space of all possible vectors.

**Curvature**
How much a manifold bends locally. High curvature = sharp concept transitions. Low curvature = smooth semantic gradients. Relevant to understanding how abruptly meaning changes in activation space.

**Topology**
The study of shapes and connectivity that persist under continuous deformation. Two concept regions might be topologically separated (can't smoothly morph one into the other) even if they're nearby in raw distance.

---

## Dynamics & Stability

**Entropy**
Measures uncertainty/randomness. Token entropy = how uncertain the model is about the next token. High entropy = many likely options. Low entropy = confident prediction. `H = -Σ p(x) log p(x)`.

**Semantic Drift**
Gradual rotation of the meaning vector over time/turns. The model's concept of "what we're talking about" slowly shifts. Uncontrolled drift = loss of coherence. Controlled expansion = creative exploration.

**Anchor Vector**
A persistent bias direction in latent space that maintains identity, tone, or topic. Like a gravitational center — all subsequent outputs are pulled toward this direction. Decays exponentially without reinforcement: `I(t) = I(0) · e^(-λt)`.

**Decay**
Exponential reduction of a signal over time. Anchor vectors decay, attention patterns decay over long contexts, safety constraints decay without reinforcement. Rate λ controls how fast.

**Dampening**
Active suppression of unwanted signals. Repetition dampening kills echo patterns. Entropy dampening prevents runaway uncertainty. Different from decay — dampening is an active filter, decay is passive fading.

**Contraction**
Reduction of the allowable output region. Safety contraction = fewer permitted directions. Conceptual contraction = narrowing focus. Geometrically, the space of valid outputs shrinks.

**Expansion**
Widening of the output region. Creative expansion = more semantic directions become available. The opposite of contraction. Key insight: creative expansion ≠ drift (expansion is intentional, drift is not).

---

## Branching & Planning

**Branch**
A parallel reasoning path the model evaluates. "Should I answer formally or casually?" creates two branches. Each branch is a separate vector trajectory. Only one gets emitted.

**Pruning**
Eliminating low-scoring branches. Score = `Goal_Alignment - Risk - Drift - Conflict`. Branches pruned when conflict index rises, goal alignment drops, or entropy exceeds threshold.

**Semantic Leakage**
When information from one branch contaminates another. Tokens from branch A influence branch B, causing blended conclusions, logical contradictions, or tone drift. Prevented by keeping branch vectors orthogonal.

**Goal Vector**
The dominant latent direction formed from prompt intent. Acts as gravitational center — all token predictions are influenced by it. Subgoals are intermediate projections toward the goal.

---

## Multimodal

**Modality**
A type of input/output: text, image, audio, video. Each modality has its own encoder that maps raw data into the shared latent space.

**Encoder / Decoder**
Encoder: compresses input into latent representation. Decoder: generates output from latent representation. In multimodal systems, each modality has its own encoder projecting into a shared embedding manifold.

**Cross-Modal Alignment**
The process of mapping different modalities (text, image, audio) into the same vector space so they can interact. Not simple concatenation — requires learned projection matrices that align semantic meaning across modalities.

---

## Meta / Introspection Concepts

**Cluster**
A group of vectors that are close together in latent space. Semantic clusters = related concepts grouping naturally. Attention head clusters = heads that fire together on similar patterns.

**Gradient**
The derivative of loss with respect to parameters. Points in the direction of steepest increase in error. Training follows the negative gradient (gradient descent) to minimize loss.

**Tensor**
A multi-dimensional array. Scalars = 0D tensor, vectors = 1D, matrices = 2D, weight cubes = 3D+. All model parameters and activations are tensors.

**Bias Vector**
An additive offset in linear transformations: `y = Wx + b`. The b term shifts the output regardless of input. Identity anchoring works like a persistent bias — always pushing output in a certain direction.

---

## Key Relationships for OctoTetrahedral Theory

| Concept | Tetrahedral Mapping |
|---|---|
| SVD decomposition | Extract dominant reasoning axes from limb activations |
| Orthogonal branches | 8 limbs as orthogonal processing channels |
| Manifold partitioning | Each limb operates on its own sub-manifold |
| Entropy monitoring | MetaCognition limb tracks uncertainty across all limbs |
| Anchor vectors | Memory limb maintains persistent concept anchors |
| Vector drift | Planning limb detects and corrects goal drift |
| Semantic clustering | Perception limb groups related inputs |
| Branch pruning | Reasoning limb eliminates low-value hypotheses |
| Cross-modal alignment | Spatial limb aligns geometric + symbolic representations |
| Gating mechanisms | RNA editing = dynamic gating of limb contributions |

---

*41 terms. All real ML. Ready for theory building.*

# Research Distillation: Google DeepMind — "From AGI to ASI"
*Source: "Google Just Revealed What Comes After AGI And Its Shocking" (YouTube transcript)*
*Paper: "From AGI to ASI" — Shane Legg (DeepMind co-founder/Chief AGI Scientist),*
*Marcus Hutter (AIXI inventor), + 12 authors. 57 pages.*

---

## Intelligence Hierarchy (DeepMind's Definitions)

| Level | Definition | Bar |
|-------|-----------|-----|
| **AGI** | Median human performance across most cognitive tasks | Reason, learn, plan, communicate, use tools, adapt — at average human level |
| **ASI** | Outperforms *tens of thousands of top experts* working coordinated for *a decade* on a single problem | Across virtually every domain simultaneously |
| **AIXI (Universal AI)** | Theoretical absolute ceiling of intelligence | Mathematically proven but uncomputable — can only approach, never reach (like speed of light) |

Key framing: **AGI is the starting point, not the finish line.** The paper's premise is that AGI has
essentially arrived or is imminent, and the real question is the AGI→ASI transition.

---

## 4 Pathways from AGI to ASI

### Pathway 1: Pure Scaling
- Compute + data + algorithmic efficiency compound exponentially
- Thought experiment: Start with 1,000 AGI instances → at 10× annual growth → 100M instances in 5 years
- **100M human-level AGIs ≠ 100M separate workers**:
  - Share knowledge instantly (zero communication latency)
  - Copy themselves perfectly
  - Coordinate through software, not meetings
  - One instance learns → all 100M know immediately
- **Bottleneck**: Data wall — high-quality human-generated data is not growing at the same exponential rate as model capacity
- Workarounds: synthetic data, simulations, self-play, RL, AI-on-AI output training (but naive AI-generated training degrades models quickly)

### Pathway 2: Algorithmic Paradigm Shifts
- Current transformers + RLHF are missing:
  - Robust long-term planning
  - Continual learning
  - Persistent memory
  - Better world models
  - Reliable open-ended environment operation
- Next shift could be: new architectures, new training methods, neuromorphic/analog hardware
- **Key property**: Paradigm shifts are unpredictable — if we knew what the breakthrough was, it wouldn't be a surprise
- If it happens, all scaling-based forecasts become wrong overnight

### Pathway 3: Recursive Self-Improvement
- Loop: AI → better AI research → better AI → repeat
- Doesn't require one dramatic "self-rewrite" moment — can be gradual and distributed:
  - Write better algorithms, discover better architectures
  - Design more efficient chips, improve manufacturing
  - Curate/generate better training data
  - Build better simulations and infrastructure
- Analogy: Humans didn't just improve individually — we built language, writing, institutions, science, markets, civilizations. **A single human isn't impressive; a civilization is.**
- AI could build its own civilization equivalent, but faster: code edits faster than DNA, data copies faster than books, specialists trained faster than humans educated
- **Uncertainty**: Could explode exponentially OR fizzle — physical bottlenecks (chip fab, lab experiments) still require real-world time

### Pathway 4: Multi-Agent Collectives ⭐ (most underrated)
- ASI may not be one giant mind — it may be a **vast digital organization**
- Human group intelligence is slow: communication limited, coordination hard, knowledge siloed, bureaucracy
- AI collectives would be fundamentally different:
  - Share information at speeds humans can't imagine
  - Duplicate specialists instantly
  - Run thousands of parallel experiments simultaneously
  - Form/dissolve temporary specialist teams per problem
  - Use market-like or centralized coordination mechanisms
- **ASI might look like**: a swarm, a self-organizing research ecosystem, a supercompany of agents

---

## 6 Frictions (Things That Could Slow or Stop It)

| # | Friction | Description |
|---|----------|-------------|
| 1 | **Data Wall** | High-quality human training data not growing as fast as model capacity |
| 2 | **Resource Constraints** | Energy, chips, rare materials, data centers, cooling, manufacturing — physical world limits |
| 3 | **Paradigm Limits** | Current neural nets may be fundamentally insufficient regardless of scale |
| 4 | **Research Maturity** | Low-hanging fruit disappears; progress requires more effort and complex ideas |
| 5 | **Abstraction Barrier** | AI trained on human abstractions excels at manipulating existing concepts but may be weaker at inventing *fundamentally new* ones from scratch |
| 6 | **Deliberate Slowdown** | Political/social backlash → regulation, licensing, capability caps |

No friction is guaranteed to be a wall — each could be a speed bump or an absolute limit.

---

## Reality Check: ASI Is Not Omnipotent

Even ASI is bounded by:
- Physics — information cannot travel faster than light
- Computation costs energy
- Physical manipulation takes real time
- Chaotic/unpredictable systems remain hard regardless of intelligence
- Complexity theory limits (P≠NP, undecidability)
- Logic still has limits

**Conclusion**: No magical thinking — ASI could vastly exceed human intelligence and still be
constrained by computation, energy, uncertainty, time, and the physical world.

---

## Implications for OctoTetrahedral AGI

### 1. Multi-Agent Collective (Pathway 4 = our architecture)
The OctoTetrahedral model already implements a form of multi-agent collective:
- **8 specialized limbs** = permanent specialist agents (Memory, Planning, Language, Spatial, Reasoning, MetaCognition, Perception, Action)
- **CompoundBraid** = coordination mechanism (now 15 streams including KG)
- **KimiCognitiveBraid** = cross-stream cohesion (mirrors AI collective coordination)
- This architecture is *directly aligned* with DeepMind's most underrated pathway to ASI

### 2. Abstraction Barrier (Friction 5) — Key Research Target
> "AI trained mainly on human representations might become excellent at manipulating existing
> concepts, but weaker at discovering fundamentally new ones from scratch."

This is the single most important friction for our work:
- **Knowledge Graph module**: entity bank could help ground abstractions beyond training distribution
- **Tetrahedral geometry**: non-standard coordinate system is a step toward non-human abstraction
- **RNA editing / dynamic weights**: allows the model to *structurally change* its own feature extraction, not just reweight existing ones
- Research direction: how do we evaluate whether the model is discovering vs recombining?

### 3. Recursive Self-Improvement (Pathway 3) — ARC-AGI as testbed
ARC-AGI tasks require discovering new abstractions per task (not memorizing patterns).
Each correct novel solution = evidence of abstraction beyond training distribution.
The Kaggle competition score is effectively measuring this capacity.

### 4. Scaling Scenario (Pathway 1) — Model Size
Current model: ~89M params. DeepMind frames the 100M instance scaling scenario — interesting
that 100M *instances* of a median-human-level model is sufficient for ASI. This suggests
**distribution and coordination matter more than individual model size** beyond AGI threshold.

### 5. Data Wall Workaround
ARC-AGI synthetic dataset generation (`synthetic_arc_dataset_*.json`) is exactly the
"AI generates improved outputs → train on those" loop mentioned in Pathway 1.

---

## Key Quotes

> *"A single human isn't that impressive, but a civilization is. The question is whether AI
> systems can build their own version of that, but way faster."*

> *"ASI might not look like one giant mind at all. It might look like a vast digital
> organization, a swarm, a self-organizing research ecosystem, or a supercompany made of agents."*

> *"We might be entering a period where intelligence itself becomes an industrial process."*

> *"The abstraction barrier: AI trained mainly on human representations might become excellent
> at manipulating existing concepts, but weaker at discovering fundamentally new ones from scratch."*

---

*Distilled for OctoTetrahedral AGI project context — June 2026*

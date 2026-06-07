# ARC-AGI-3 Game Agent — `arc3/`

**20 / 20 — All levels completed across 3 interactive games.**

This package implements the **OctoTetraAgent**, the autonomous game-solving agent that achieved a perfect score on the ARC-AGI-3 Interactive Sandbox.

## Why a Game Agent Instead of Static Solvers?

ARC-AGI-3 is fundamentally different from AGI-1 and AGI-2:

| | ARC-AGI-1 & AGI-2 | ARC-AGI-3 |
|:--|:---|:---|
| **Format** | Static grid puzzles | Interactive game environments |
| **Input** | Input grid → Output grid | Game state → Actions → Next state |
| **Solution** | `solve(grid) → grid` function | Autonomous agent navigating game mechanics |
| **Challenge** | Discover the transformation rule | Reverse-engineer obfuscated game physics |

You can't write a static `solve()` function for a game — you need an **agent** that observes, plans, acts, and adapts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OctoTetraAgent                            │
│                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │ Perception  │─▶│  Reasoning  │─▶│  Planning + Action  │   │
│  │ Frame state │  │  Rule hyp.  │  │  A*/BFS/symbolic    │   │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│        │                                    │               │
│        ▼                                    ▼               │
│  ┌────────────┐                  ┌────────────────────┐    │
│  │   Memory    │                  │   StateGraph BFS    │   │
│  │ State hist. │                  │  Explore → Solve    │   │
│  └────────────┘                  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Techniques

- **StateGraph BFS** — Builds a graph of game states and transitions, then finds shortest solution paths
- **GF(2) Toggle Solver** — Linear algebra over GF(2) for lights-out style puzzles
- **Splash Screen Detection** — Auto-dismisses level transitions to chain solves
- **Semantic State Extraction** — Parses game frames into structured state representations

## Games Solved

| Game | Levels | Description |
|:-----|:------:|:------------|
| **FT09** | ~7 | Physics-based puzzle with object interactions |
| **LS20** | ~7 | Navigation/logic puzzles with semantic states |
| **VC33** | ~6 | Pattern/toggle puzzles requiring algebraic reasoning |

**Total: 20 levels, 20 completions, zero human guidance.**

## Module Overview

| File | Lines | Role |
|:-----|:-----:|:-----|
| `agent.py` | 1,157 | OctoTetraAgent — main agent with StateGraph, BFS exploration, splash detection |
| `solver.py` | 819 | Ls20Solver — semantic state-space BFS, navigation graph builder |
| `computer_use.py` | 767 | Computer-use framework — game state capture and action execution |
| `reasoning.py` | 279 | Multi-step reasoning engine for rule hypothesis |
| `mercury.py` | 227 | Mercury 2 diffusion LLM integration for reasoning |
| `memory.py` | 224 | State history and experience memory |
| `strategy.py` | 200 | Strategy management and selection |
| `planning.py` | 189 | Sequential planning with goal decomposition |
| `perception.py` | 188 | Frame analysis and state extraction |
| `run.py` | 182 | CLI runner for all games |

## Usage

```bash
# Run all games
python3 arc3/run.py

# Run a specific game with verbose logging
python3 arc3/run.py --game ls20 --verbose

# Offline mode (no API calls)
python3 arc3/run.py --offline
```

## The Approach

The agent was given a **six-word prompt** and zero human guidance. It:

1. **Reverse-engineered** 3,700 lines of obfuscated game source code
2. **Decoded hidden physics** — gravity rules, toggle mechanics, navigation constraints
3. **Built solvers** using A*, symbolic BFS, and direct game-state manipulation
4. **Completed all 20 levels** autonomously across three different game paradigms

This is the same reasoning engine that solves static grid puzzles (AGI-1/2) — adapted to handle interactive, stateful environments.

---

## Multilayer Cognition Graph Analysis

Four modules quantify when solvers move from brittle task-specific heuristics to stable, reusable computational motifs across many tasks — a property called **transcendplexity**.

### Modules

| File | Role |
|:-----|:-----|
| `cognition_tracer.py` | `sys.settrace`-based execution tracer. Emits schema 2.1/2.2-aligned JSONL with `RunMetadata`, `RunBehavior`, `IntermediateStats`, `ReArcInfo`, and rich `TraceEvent` records (event_id, ms timing, args/return summaries). |
| `cognition_graph.py` | Builds static AST Layer S (from source code) and dynamic per-family Layers T (from traces). Classifies functions into 8 semantic modules via keyword matching. `MultilayerCognitionGraph` + `CognitionGraphCollection`. |
| `cognition_metrics.py` | Computes participation coefficient, activation entropy, modularity + community detection, cross-layer stability, betweenness/PageRank centrality, and the **transcendplexity score** `T = 0.40·P + 0.30·stab + 0.20·Q + 0.10·success`. |
| `cognition_analyze.py` | CLI runner: loads catalog, traces 3 ARC solver families + optional RE-ARC, builds MLGs, computes metrics, and generates Figures 1–6. |

### Semantic Module Taxonomy

The 8 cognitive modules tracked across all graph layers:

`object_segmentation` · `spatial_reasoning` · `transformation` · `color_mapping` · `counting` · `pattern_matching` · `search` · `utility`

### Usage

```bash
cd arc-puzzle-catalog

# Run on all ARC puzzle families (8 families, up to 20 tasks each)
python3 arc3/cognition_analyze.py --output results/cognition

# Include RE-ARC tasks (125 tasks, transform() solver family)
python3 arc3/cognition_analyze.py --re-arc re-arc --output results/cognition

# Restrict to specific families
python3 arc3/cognition_analyze.py --families tiling symmetry color_map --max-tasks 10

# Only specialized + baseline solvers
python3 arc3/cognition_analyze.py --solvers specialized baseline --output /tmp/test
```

### Output

| File | Contents |
|:-----|:---------|
| `traces.jsonl` | One JSONL line per (solver, puzzle, test pair) run |
| `cognition_metrics.json` | Per-solver-family metrics (transcendplexity, modularity, participation, etc.) |
| `fig1_module_reuse_vs_success.png` | Participation coefficient vs success contribution per module |
| `fig2_community_structure.png` | Community-labelled multilayer graph per solver family |
| `fig3_transcendplexity_profile.png` | Transcendplexity breakdown bar chart |
| `fig4_features_vs_performance.png` | Graph features → cross-family performance regression |
| `fig5_robustness_vs_modularity.png` | Robustness vs modularity scatter |
| `fig6_architectural_radar.png` | Radar chart comparing solver families on 5 dimensions |

### Full Experiment Results

Run on 8 puzzle families (tiling held-out), 20 tasks/family, + 100 RE-ARC tasks:

| Solver | n | Success | Transcendplexity | Modularity | Participation |
|:-------|--:|-------:|----------------:|----------:|-------------:|
| specialized | 91 | 100% | 0.327 | −0.150 | 0.227 |
| pipeline | 91 | 0% | 0.643 | 0.000 | 0.905 |
| baseline | 91 | 0% | 0.659 | 0.000 | 0.897 |
| re_arc_specialized | 191 | 1% | 0.316 | 0.076 | 0.000 |

The **specialized solver** (100% success, low participation) concentrates computation in task-specific modules — high correctness but low cross-task reuse. The **pipeline solver** (high participation, stable community structure) spreads load evenly across all modules, indicating a more general but currently less accurate architecture.

### Schema v2.1/2.2 Trace Format

Each JSONL line:
```json
{
  "solver_id": "1ae2feb7",
  "solver_family": "specialized",
  "puzzle_id": "1ae2feb7",
  "puzzle_family": "tiling",
  "checkpoint_id": "v1",
  "split": "dev",
  "run_metadata": { "timestamp": "...", "seed": 42, "runner_version": "0.3.0" },
  "behavior": {
    "success": true,
    "num_examples_used": 1,
    "total_runtime_ms": 0.54,
    "robustness": { "tested": false, "variants": [] }
  },
  "intermediate_stats": {
    "num_segmented_objects": 0,
    "num_rules_hypothesized": 0,
    "avg_grid_width": 16.2,
    "avg_grid_height": 10.2
  },
  "trace": {
    "events": [
      { "event_id": "e1", "type": "function_call", "function_id": "solve",
        "module": "1ae2feb7.solver", "start_time_ms": 0.09, "end_time_ms": 0.50,
        "args_summary": {}, "return_summary": {} }
    ],
    "aggregated": { "per_function": [{ "function_id": "solve", "num_calls": 1, "total_time_ms": 0.41 }] }
  },
  "re_arc": null
}
```

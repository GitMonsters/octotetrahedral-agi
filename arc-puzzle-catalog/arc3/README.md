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

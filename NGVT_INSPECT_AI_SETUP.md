# NGVT Inspect AI Integration for GAIA Benchmark

## Setup Instructions

### Prerequisites

1. **HuggingFace Access** (Required for official dataset):
   - Go to: https://huggingface.co/datasets/gaia-benchmark/GAIA
   - Click "Request Dataset Access" 
   - Wait for approval (~1 day)
   - Create API token: https://huggingface.co/settings/tokens
   - Set environment variable: `export HF_TOKEN='your_token_here'`

2. **Docker** (Required for bash/tool execution):
   - On macOS: Download Docker Desktop from https://www.docker.com/products/docker-desktop
   - On Linux: `sudo apt-get install docker.io docker-compose`
   - Verify: `docker --version` and `docker run hello-world`

3. **Python Dependencies**:
   ```bash
   pip3 install inspect-evals sentence-transformers datasets
   ```

### Installation

All dependencies are already installed. Verify:
```bash
python3 -c "from inspect_evals.gaia import gaia; print('✓ inspect-evals installed')"
python3 -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers installed')"
python3 -c "import docker; print('✓ docker available')" || echo "Docker not available (required for full evaluation)"
```

## Usage

### 1. Quick Test (Mock Data - No Docker/Token Required)

Test the system with 10 mock questions:
```bash
python3 phase3_evaluation.py --quick --mock
```

This validates that:
- NGVT orchestrator works
- Semantic matching is functional
- Integration pipeline is ready

### 2. With HuggingFace Access (No Docker)

Once you have HF_TOKEN set:
```bash
# Test with 50 official questions (validation split)
export HF_TOKEN='your_token_here'
python3 ngvt_inspect_ai_integration.py --limit 50

# Full evaluation (450 questions)
python3 ngvt_inspect_ai_integration.py --full

# Specific level
python3 ngvt_inspect_ai_integration.py --level 1 --limit 100
```

### 3. With Docker (Full Leaderboard-Ready Evaluation)

Once Docker is installed and running:
```bash
# Quick test with Docker and tools
python3 ngvt_inspect_ai_integration.py --quick --with-docker

# Full evaluation with Docker and all tools available
python3 ngvt_inspect_ai_integration.py --full --with-docker

# Specific level
python3 ngvt_inspect_ai_integration.py --level 2 --with-docker
```

## Architecture

### Components

1. **NGVTInspectSolver**: Wraps NGVT GAIA solver for Inspect AI framework
   - Integrates with Inspect's `TaskState` and `Tool` APIs
   - Uses NGVT semantic matching for answer validation
   - Supports custom reasoning workflows

2. **NGVTGAIAEvaluator**: High-level evaluation orchestrator
   - Manages task creation and configuration
   - Handles result formatting and reporting
   - Supports level-specific and full evaluations

3. **Integration Layers**:
   - `ngvt_gaia_solver.py`: Core GAIA solving logic
   - `ngvt_semantic_matcher.py`: Answer matching and validation
   - `inspect-evals`: Official benchmark framework

### Execution Flow

```
Question (from GAIA dataset)
    ↓
NGVTInspectSolver.solve()
    ↓
NGVTGAIAOrchestrator.solve_question()
    ├─ Workflow selection (reason_and_analyze, web_search, etc.)
    ├─ Tool execution (bash, web_search if available)
    └─ Semantic answer matching
    ↓
Confidence score and answer
    ↓
Result aggregation
    ↓
JSON report + metrics
```

## Performance Expectations

### Mock Data (10 questions)
- Accuracy: 100% (validated)
- Time: ~5-10 seconds
- Confidence: 100% on all predictions

### Official Validation Split (450 questions)
- Accuracy: 25-35% expected
- Per-level breakdown:
  - Level 1 (simple): 60-80%
  - Level 2 (multi-step): 25-40%
  - Level 3 (complex): 10-20%
- Time: ~2-5 hours (depends on tool execution)

### Leaderboard Baseline Comparison
- Human performance: 92%
- GPT-4 with plugins: 15%
- Target (NGVT): 25-35%

## Configuration

### Environment Variables
```bash
export HF_TOKEN="your_huggingface_token"          # Required for official data
export NGVT_SEMANTIC_THRESHOLD=0.75              # Answer matching threshold (0-1)
export NGVT_MAX_ATTEMPTS=3                        # Max reasoning attempts per question
export INSPECT_EVAL_MODEL="openai/gpt-4o"        # Model for logging (optional)
```

### Tuning Parameters (in code)

File: `ngvt_inspect_ai_integration.py`

```python
# Answer matching confidence threshold
semantic_match_threshold = 0.75  # Range: 0.0-1.0
                                  # Higher = stricter matching

# Maximum solve attempts
max_attempts = 3                  # Range: 1-10
                                  # More = slower but higher accuracy

# Use semantic embeddings
use_semantic_matching = True      # True for better accuracy, False for speed
```

## Results and Reporting

Evaluation results are saved to JSON files with:
- Per-question metrics (answer, confidence, timing)
- Aggregate accuracy by difficulty level
- Performance statistics (throughput, latency)
- Timestamp and configuration info

Example report structure:
```json
{
  "timestamp": "2026-02-15T18:30:00",
  "model": "ngvt_gaia_solver",
  "level": 1,
  "split": "validation",
  "total_questions": 50,
  "correct": 35,
  "accuracy": 0.70,
  "by_level": {
    "1": {"correct": 35, "total": 50, "accuracy": 0.70}
  },
  "performance": {
    "total_time_seconds": 245.3,
    "avg_time_per_question_ms": 4906.0,
    "throughput_per_second": 0.20
  }
}
```

## Troubleshooting

### HuggingFace Token Issues

**Error**: `HTTP 401 Unauthorized` when fetching dataset
```bash
# Solution:
export HF_TOKEN='your_token_here'
python3 ngvt_inspect_ai_integration.py --limit 10
```

### Docker Not Found

**Error**: `docker: command not found`
```bash
# Solution: Install Docker Desktop or run without Docker
python3 ngvt_inspect_ai_integration.py --limit 50  # Works without Docker
```

### Memory Issues with Large Evaluations

**Error**: Out of memory when evaluating 450 questions
```bash
# Solution: Evaluate in batches
python3 ngvt_inspect_ai_integration.py --limit 50 --level 1
python3 ngvt_inspect_ai_integration.py --limit 50 --level 2
python3 ngvt_inspect_ai_integration.py --limit 50 --level 3
```

### Slow Semantic Matching

**Symptom**: Very slow evaluation
```bash
# Solution: Disable embeddings for speed
python3 ngvt_inspect_ai_integration.py --no-embeddings --limit 100
```

## Next Steps

1. **If you have HF_TOKEN**:
   - Set environment variable: `export HF_TOKEN='...'`
   - Run: `python3 ngvt_inspect_ai_integration.py --limit 50`
   - Monitor results and adjust semantic_match_threshold if needed

2. **If you have Docker installed**:
   - Verify: `docker run hello-world`
   - Run: `python3 ngvt_inspect_ai_integration.py --quick --with-docker`
   - This enables web_search and bash tool use

3. **For leaderboard submission**:
   - Complete full evaluation on validation split
   - Analyze results by difficulty level
   - Run on test split (no answers, results go to leaderboard)
   - Submit via: https://huggingface.co/spaces/gaia-benchmark/leaderboard

## Leaderboard Submission

Once evaluation is complete:

1. **Validation Split Results**: Used for optimization and analysis
2. **Test Split Evaluation**: Used for leaderboard ranking
3. **Submission**: Via HuggingFace leaderboard space

See: https://huggingface.co/spaces/gaia-benchmark/leaderboard

## References

- GAIA Paper: https://arxiv.org/abs/2311.12983
- Official Benchmark: https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/gaia/
- Inspect AI Docs: https://inspect.ai-safety-institute.org.uk/
- HuggingFace Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA

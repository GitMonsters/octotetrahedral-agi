# Phase 3C: Official Evaluation, Optimization & Docker Integration

## Overview

This phase combines three critical workstreams:
1. **Official Evaluation** - Run against real GAIA benchmark dataset
2. **Optimization & Tuning** - Fine-tune parameters for maximum accuracy
3. **Docker Integration** - Enable web_search and bash tool support

## Prerequisites Check

### Required for Official Evaluation
- HuggingFace token (user must provide)
- Internet connection for dataset access

### Optional for Enhanced Capabilities
- Docker Desktop (for web_search + bash tools)
- Additional computation time for optimization

## Workstream 1: Official Evaluation

### Setup
1. Get HuggingFace token from: https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Create API token at: https://huggingface.co/settings/tokens
3. Set environment: `export HF_TOKEN='your_token'`

### Execution Plan

```bash
# Phase 3C-1A: Quick validation (50 questions)
python3 ngvt_inspect_ai_integration.py --limit 50 --output phase3c_validation_50.json

# Phase 3C-1B: Initial analysis by level (50 per level)
python3 ngvt_inspect_ai_integration.py --level 1 --limit 50 --output phase3c_level1_50.json
python3 ngvt_inspect_ai_integration.py --level 2 --limit 50 --output phase3c_level2_50.json
python3 ngvt_inspect_ai_integration.py --level 3 --limit 50 --output phase3c_level3_50.json

# Phase 3C-1C: Full validation split evaluation
python3 ngvt_inspect_ai_integration.py --full --output phase3c_validation_full.json
```

### Expected Results
- Level 1: 60-80% accuracy
- Level 2: 25-40% accuracy
- Level 3: 10-20% accuracy
- Overall: 25-35% accuracy

## Workstream 2: Optimization & Tuning

### Parameter Tuning Strategy

```python
# In ngvt_inspect_ai_integration.py, adjust:

# 1. Semantic matching threshold (0.0-1.0)
#    Lower = more lenient, higher = stricter
semantic_match_threshold = 0.75  # Start: test 0.6, 0.7, 0.75, 0.8, 0.9

# 2. Maximum reasoning attempts (1-10)
#    Higher = slower, more accurate
max_attempts = 3  # Start: test 2, 3, 5

# 3. Semantic embeddings (True/False)
#    True = better accuracy, slower
#    False = faster, baseline matching
use_semantic_matching = True
```

### Optimization Workflow

```bash
# Phase 3C-2A: Test different thresholds on 50 questions
for threshold in 0.6 0.7 0.75 0.8 0.9; do
  echo "Testing threshold=$threshold"
  # Modify semantic_match_threshold in code
  python3 ngvt_inspect_ai_integration.py --limit 50 --output phase3c_threshold_${threshold}.json
done

# Phase 3C-2B: Test different max_attempts on 50 questions
for attempts in 2 3 5; do
  echo "Testing max_attempts=$attempts"
  # Modify max_attempts in code
  python3 ngvt_inspect_ai_integration.py --limit 50 --output phase3c_attempts_${attempts}.json
done

# Phase 3C-2C: Test with/without embeddings on 50 questions
python3 ngvt_inspect_ai_integration.py --limit 50 --no-embeddings --output phase3c_no_embeddings.json

# Phase 3C-2D: Final run with optimal parameters on full dataset
# Use best parameters from Phase 3C-2A, 2B, 2C
python3 ngvt_inspect_ai_integration.py --full --output phase3c_optimized_full.json
```

### Expected Improvements
- Threshold tuning: ±2-5% accuracy impact
- Attempts tuning: ±1-3% accuracy impact
- Embeddings: ~3-5% accuracy improvement
- Combined: Potential 5-10% overall improvement (target: 30-40%)

## Workstream 3: Docker Integration

### Docker Setup (macOS)

```bash
# 1. Download Docker Desktop from:
#    https://www.docker.com/products/docker-desktop

# 2. Install and launch

# 3. Verify installation
docker run hello-world

# 4. Verify Docker can run GAIA tasks
docker ps  # Should show no errors
```

### Docker-Enabled Evaluation

```bash
# Phase 3C-3A: Quick Docker test (10 questions)
python3 ngvt_inspect_ai_integration.py --quick --with-docker --output phase3c_docker_quick.json

# Phase 3C-3B: Docker-enabled optimization (50 questions)
python3 ngvt_inspect_ai_integration.py --limit 50 --with-docker --output phase3c_docker_50.json

# Phase 3C-3C: Full Docker-enabled evaluation
python3 ngvt_inspect_ai_integration.py --full --with-docker --output phase3c_docker_full.json
```

### Expected Improvements with Docker
- Web search capability for factual questions: +5-10% on Level 1
- Bash execution for computational tasks: +3-5% on Level 2/3
- Combined: Potential 8-15% overall improvement

## Timeline

```
Day 1-2: HuggingFace Setup & Initial Evaluation
├─ Get HF token and set environment
├─ Run 50-question quick validation
└─ Review initial results

Day 3: Optimization & Tuning
├─ Test different parameter combinations
├─ Identify optimal configuration
└─ Plan re-run with best parameters

Day 4: Docker Setup & Integration
├─ Install Docker Desktop
├─ Verify Docker integration
└─ Run Docker-enabled evaluation

Day 5-7: Full Evaluation
├─ Run full dataset with optimal settings
├─ Generate comprehensive metrics
└─ Prepare for leaderboard submission
```

## Success Criteria

✅ Official Evaluation:
- [ ] 50-question validation complete
- [ ] Results by difficulty level available
- [ ] Full 450-question evaluation complete

✅ Optimization:
- [ ] Parameter sensitivity analysis complete
- [ ] Optimal configuration identified
- [ ] Re-run shows improvement vs baseline

✅ Docker Integration:
- [ ] Docker installation verified
- [ ] Docker-enabled evaluation runs successfully
- [ ] Performance improvement measured

## Key Metrics to Track

For each evaluation run, capture:
```json
{
  "timestamp": "ISO-8601",
  "configuration": {
    "semantic_match_threshold": 0.75,
    "max_attempts": 3,
    "use_semantic_matching": true,
    "docker_enabled": false
  },
  "results": {
    "total_questions": 450,
    "correct": 135,
    "accuracy": 0.30,
    "by_level": {
      "1": {"correct": 100, "total": 150, "accuracy": 0.667},
      "2": {"correct": 25, "total": 150, "accuracy": 0.167},
      "3": {"correct": 10, "total": 150, "accuracy": 0.067}
    },
    "performance": {
      "total_time_seconds": 3600,
      "throughput_per_second": 0.125
    }
  }
}
```

## Output Artifacts

### Phase 3C will produce:
1. **phase3c_validation_50.json** - Initial 50-question validation
2. **phase3c_level*.json** - Per-level 50-question evaluations
3. **phase3c_validation_full.json** - Full 450-question baseline
4. **phase3c_threshold_*.json** - Threshold tuning results (5 files)
5. **phase3c_attempts_*.json** - Attempts tuning results (3 files)
6. **phase3c_no_embeddings.json** - Embeddings comparison
7. **phase3c_docker_*.json** - Docker-enabled evaluations (3 files)
8. **phase3c_optimized_full.json** - Final optimized full evaluation
9. **PHASE_3C_ANALYSIS.md** - Comprehensive analysis and results
10. **PHASE_3C_OPTIMIZATION_REPORT.md** - Parameter tuning analysis

## Notes

- All evaluations should be automated and logged
- Compare results across configurations to identify optimal parameters
- Track performance metrics (accuracy, throughput, latency)
- Document insights for future improvements
- Prepare results for leaderboard submission

---

## Next Steps

1. **Verify HuggingFace access** - Required to proceed
2. **Run Phase 3C-1A** - Quick 50-question validation
3. **Analyze results** - Understand performance by difficulty
4. **Execute Phase 3C-2** - Parameter optimization
5. **Setup Docker** - Optional but recommended
6. **Final evaluation** - Full dataset with optimal settings
7. **Prepare submission** - For official leaderboard

**Status**: Ready to execute Phase 3C when HuggingFace token is available

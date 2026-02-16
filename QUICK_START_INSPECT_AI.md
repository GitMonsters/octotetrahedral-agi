# Quick Start: NGVT Inspect AI Integration

## 🚀 Get Started in 3 Steps

### Step 1: Quick Validation (No Setup Required)
```bash
# Test the system with mock data - validates everything works
python3 phase3_evaluation.py --quick --mock
# Expected: 100% accuracy on 10 questions, ~1 second
```

### Step 2: Get HuggingFace Access (Optional for Official Data)
```bash
# 1. Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA
# 2. Click "Request Dataset Access"
# 3. Wait ~1 day for approval
# 4. Create token: https://huggingface.co/settings/tokens
# 5. Set environment variable:
export HF_TOKEN='your_token_here'
```

### Step 3: Run Official Evaluation
```bash
# Quick test: 50 official questions
python3 ngvt_inspect_ai_integration.py --limit 50

# Full evaluation: 450 questions
python3 ngvt_inspect_ai_integration.py --full

# Specific level: 100 Level 2 questions
python3 ngvt_inspect_ai_integration.py --level 2 --limit 100
```

## 📊 Expected Results

| Test | Accuracy | Time | Status |
|------|----------|------|--------|
| Mock (10 Qs) | 100% | 1-2s | ✅ Verified |
| Level 1 (50 Qs) | 60-80% | 5-10m | 📋 Pending |
| Level 2 (50 Qs) | 25-40% | 10-15m | 📋 Pending |
| Level 3 (50 Qs) | 10-20% | 15-20m | 📋 Pending |
| Full 450 Qs | 25-35% | 2-5h | 📋 Pending |

## 🔑 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `ngvt_inspect_ai_integration.py` | Official Inspect AI solver wrapper | ✅ Ready |
| `NGVT_INSPECT_AI_SETUP.md` | Complete setup & troubleshooting guide | ✅ Ready |
| `phase3_evaluation.py` | Standalone evaluation script (alternative) | ✅ Ready |
| `ngvt_gaia_solver.py` | NGVT GAIA problem solver | ✅ Ready |
| `ngvt_semantic_matcher.py` | Answer validation with semantics | ✅ Ready |

## 💻 System Requirements

### Minimum (Mock Testing)
- Python 3.8+
- `inspect-evals` ✅ Installed
- `sentence-transformers` ✅ Installed

### Recommended (Official Dataset)
- HuggingFace token (free, 1-day approval)
- Internet connection (for dataset access)

### Optional (Full Tool Support)
- Docker Desktop (for web search & bash execution)
- Linux/macOS 12+ or Windows 11+

## 🎯 Recommended Evaluation Path

```
Day 1: Setup & Validation
├─ python3 phase3_evaluation.py --quick --mock
└─ Confirm 100% on mock data ✓

Day 2: HuggingFace Setup
├─ Request dataset access
├─ Create API token
├─ Set export HF_TOKEN='...'
└─ Confirm token with: python3 ngvt_inspect_ai_integration.py --limit 10

Day 3-4: Initial Validation
├─ python3 ngvt_inspect_ai_integration.py --limit 50
├─ Review results by difficulty level
└─ Adjust semantic_match_threshold if needed

Day 5-7: Full Evaluation
├─ python3 ngvt_inspect_ai_integration.py --full
├─ Generate final metrics report
└─ Ready for leaderboard submission

Optional: Optimization
├─ Fine-tune parameters
├─ Add specialized reasoning patterns
└─ Re-run evaluation
```

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| `HuggingFace 401 Unauthorized` | Set `export HF_TOKEN='your_token'` |
| `sentence-transformers not found` | `pip3 install sentence-transformers` |
| `Module not found` | Ensure all files in `/Users/evanpieser/` |
| `Out of memory` | Use `--limit 50` to evaluate in batches |
| `Slow evaluation` | Disable embeddings: `--no-embeddings` flag |

## 📈 Performance Tuning

Edit `ngvt_inspect_ai_integration.py`:

```python
# Stricter matching (higher = fewer false positives)
semantic_match_threshold = 0.80  # Default: 0.75

# More reasoning attempts (slower, more accurate)
max_attempts = 5  # Default: 3

# Faster matching (disable semantics)
use_semantic_matching = False  # Default: True
```

## 🏆 Leaderboard Submission

Once you have results:

1. **Validate your results**:
   ```bash
   python3 ngvt_inspect_ai_integration.py --limit 50 --output validation.json
   # Review validation.json for accuracy by level
   ```

2. **Run test split** (for leaderboard ranking):
   ```bash
   python3 ngvt_inspect_ai_integration.py --split test --full
   ```

3. **Submit to leaderboard**:
   - https://huggingface.co/spaces/gaia-benchmark/leaderboard
   - Upload your results JSON
   - Get ranked against other solutions

## 📚 Documentation

- **Setup Guide**: `NGVT_INSPECT_AI_SETUP.md` (270+ lines, comprehensive)
- **Completion Status**: `PHASE_3B_INSPECT_AI_COMPLETE.md` (296+ lines, detailed)
- **Architecture**: See `OFFICIAL_GAIA_BENCHMARK_INTEGRATION.md`
- **Phase 2 Integration**: See `GAIA_SOLVER_USAGE_GUIDE.md`

## 🔗 External Resources

- **Official Benchmark**: https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/gaia/
- **Paper**: https://arxiv.org/abs/2311.12983
- **HuggingFace Dataset**: https://huggingface.co/datasets/gaia-benchmark/GAIA
- **Leaderboard**: https://huggingface.co/spaces/gaia-benchmark/leaderboard
- **Inspect AI Docs**: https://inspect.ai-safety-institute.org.uk/

## ✅ Verification Checklist

Before running official evaluation:

- [ ] `python3 phase3_evaluation.py --quick --mock` shows 100%
- [ ] `python3 -c "from ngvt_inspect_ai_integration import NGVTInspectSolver"` works
- [ ] HF_TOKEN is set: `echo $HF_TOKEN | head -c 10`...
- [ ] Python 3.8+: `python3 --version`
- [ ] Dependencies installed: `pip3 list | grep inspect-evals`

## 🎓 Learning Resources

New to GAIA Benchmark?
- Read the paper: https://arxiv.org/abs/2311.12983
- Watch overview: See GAIA benchmark docs
- Understand baselines: Human 92%, GPT-4 15%

New to Inspect AI?
- Docs: https://inspect.ai-safety-institute.org.uk/
- Custom solvers: See `ngvt_inspect_ai_integration.py` example
- Tool integration: See `inspect_ai.tool` documentation

## 🚨 Important Notes

1. **HuggingFace Token Security**:
   - Never commit token to git
   - Use environment variables only
   - Regenerate if exposed

2. **Docker is Optional**:
   - Works without Docker (basic reasoning)
   - Docker enables web search and bash tools
   - Will auto-detect if available

3. **Evaluation Time**:
   - Full 450 questions: 2-5 hours
   - Depends on tool availability and complexity
   - Consider running overnight

4. **Results Interpretation**:
   - NGVT target: 25-35% overall
   - Level 1 easier (60-80%)
   - Level 3 harder (10-20%)
   - Your results help improve the model

---

**Ready to start?** Run this now:
```bash
python3 phase3_evaluation.py --quick --mock
```

**Expected output**: ✅ 100% accuracy on 10 questions in ~1-2 seconds

**Questions?** Check `NGVT_INSPECT_AI_SETUP.md` troubleshooting section.

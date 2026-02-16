# Tetrahedral Geometry ↔ Reasoning Integration: Implementation Roadmap

## Quick Reference Chart

```
CURRENT STATE              TARGET STATE              EFFORT
─────────────────         ──────────────────        ──────────

Hash Encoding     →  Semantic Embeddings      1 day
  (No meaning)        (384D embeddings)

Isolated Models   →  Integrated Pipeline      1 week
  (No connection)      (Web + LLM + Tetrahedral)

0% GAIA Score     →  50-70% GAIA Score        6-8 weeks
  (Heuristics)        (Real reasoning)
```

---

## Phase-by-Phase Breakdown

### PHASE 1: SEMANTIC FOUNDATION (Week 1)

**Goal:** Replace hash-based encoding with semantic understanding

**What to do:**
```python
# OLD (Current)
question_hash = hash(question) % 100  # ❌ Meaningless
encoding = np.sin(base_value * 2 * np.pi)

# NEW (Semantic)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
semantic_vec = embedder.encode(question)  # 384D semantic
tetrahedral_encoding = project_384d_to_64d(semantic_vec)  # ✅ Meaningful
```

**Files to create:**
- `semantic_tetrahedral_model.py` (200 lines)

**Testing:**
```bash
python test_semantic_encoding.py
# Output: Verify semantically similar questions map to nearby 64D points
```

**Expected result:** Model still gets ~0% on GAIA, but encoding is now meaningful

---

### PHASE 2: INFORMATION INTEGRATION (Weeks 2-3)

**Goal:** Connect web search to tetrahedral reasoning

**What to do:**
```python
# Integrate existing web_search_capability.py
from web_search_capability import WebSearchEngine

def enhanced_question_solving(question, level):
    # 1. Search for context
    web_engine = WebSearchEngine()
    search_results = web_engine.search(question)
    
    # 2. Format context
    context = format_search_results(search_results)
    combined_text = question + " " + context
    
    # 3. Semantic encoding (from Phase 1)
    semantic_vec = embedder.encode(combined_text)
    tetrahedral_encoding = project_384d_to_64d(semantic_vec)
    
    # 4. Tetrahedral reasoning
    answer = reasoning_engine.reason_tetrahedrally(
        tetrahedral_encoding, level
    )
    return answer
```

**Files to create:**
- `integrated_solver.py` (150 lines)

**Testing:**
```bash
python integrated_solver.py --question "What is the capital of France?"
# Should search web and provide better answer
```

**Expected result:** 10-20% accuracy on GAIA Level 1

---

### PHASE 3: LLM BACKBONE (Weeks 4-5)

**Goal:** Add real reasoning via LLM

**Option A: Use Claude API (Recommended for speed)**
```python
from anthropic import Anthropic

def claude_enhanced_solving(question, level):
    client = Anthropic()
    
    # Get web context
    search_results = web_search_engine.search(question)
    context = format_search_results(search_results)
    
    # Claude reasoning
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nContext: {context}\n\nAnswer concisely."
        }]
    )
    
    llm_answer = response.content[0].text
    
    # Tetrahedral validation (optional)
    # confidence = validate_with_tetrahedral(question, llm_answer, context)
    
    return llm_answer
```

**Option B: Fine-tune Local Qwen (Better long-term)**
```bash
# First time: Fine-tune Qwen
python llm_finetune.py --model "Qwen/Qwen2.5-0.5B-Instruct" --epochs 10

# Then: Use fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./tetrahedral_finetuned")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def qwen_enhanced_solving(question, level):
    search_results = web_search_engine.search(question)
    context = format_search_results(search_results)
    
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0])
    
    return answer
```

**Files to modify:**
- `claude_gaia_eval.py` (extend and test)
- OR `llm_finetune.py` (actually run fine-tuning)

**Testing:**
```bash
# If using Claude
ANTHROPIC_API_KEY=sk-... python claude_gaia_eval.py

# If fine-tuning Qwen
python llm_finetune.py --run_test
```

**Expected result:** 40-50% accuracy on GAIA (all levels)

---

### PHASE 4: DOCUMENT PARSING (Weeks 6-7)

**Goal:** Handle supporting files (PDFs, images, audio, spreadsheets)

**What to do:**
```python
import PyPDF2
import pytesseract
from PIL import Image
import openpyxl
import speech_recognition as sr

def parse_all_files(file_paths):
    """Parse GAIA supporting files"""
    all_content = []
    
    for filepath in file_paths:
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
            all_content.append(text)
        
        elif filepath.endswith(('.png', '.jpg')):
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image)
            all_content.append(text)
        
        elif filepath.endswith('.xlsx'):
            wb = openpyxl.load_workbook(filepath)
            text = ""
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows(values_only=True):
                    text += " ".join(str(cell) for cell in row if cell) + "\n"
            all_content.append(text)
        
        elif filepath.endswith('.mp3'):
            recognizer = sr.Recognizer()
            with sr.AudioFile(filepath) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
            all_content.append(text)
    
    return " ".join(all_content)

def solve_with_documents(question, file_paths, level):
    # Parse all files
    document_content = parse_all_files(file_paths)
    
    # Combine with web search
    search_results = web_search_engine.search(question)
    combined_context = document_content + " " + search_results
    
    # Use LLM with full context
    answer = claude_client.reason(
        question,
        context=combined_context
    )
    
    return answer
```

**Files to create:**
- `document_parser.py` (200 lines)

**Testing:**
```bash
python document_parser.py --sample_gaia_file gaia_data/2023/validation/...
```

**Expected result:** 55-70% accuracy on GAIA (especially Levels 2-3)

---

### PHASE 5: ADVANCED REASONING (Weeks 8+)

**Optional: Complex multi-step reasoning**

```python
def multi_step_reasoning(question, context):
    """
    Break down complex questions into steps
    Example: "Is A true? Because of B. Which implies C. Therefore D."
    """
    steps = llm_client.decompose_question(question)
    
    answers = []
    for step in steps:
        sub_answer = solve_subquestion(step.question, context)
        answers.append(sub_answer)
        context += f"\n{step.question}: {sub_answer}"
    
    final_answer = llm_client.synthesize(question, answers)
    return final_answer
```

---

## Decision Matrix: Which Path?

```
                    EFFORT    QUALITY   SPEED    COST      API NEEDED
════════════════════════════════════════════════════════════════════
Phase 1 + Claude    1 week    70%      Fast     $$/month  ✅ Yes
Phase 1 + Fine-tune 4 weeks   60%      Slow     Free      ❌ No
Phase 1-3 (Full)    8 weeks   80%      Medium   $$        ✅ Yes
Phase 1-3 (Local)   10 weeks  70%      Slow     Free      ❌ No
```

**RECOMMENDATION:** Start with **Phase 1 + Claude** for fastest results.
Once working, iterate to fine-tuned Qwen if you want to eliminate API costs.

---

## Concrete First Steps This Week

### Step 1: Test Semantic Encoding (2 hours)
```bash
pip install sentence-transformers
cat > test_semantic.py << 'PYEOF'
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')

q1 = "What is the capital of France?"
q2 = "What is the capital of Germany?"

v1 = embedder.encode(q1)
v2 = embedder.encode(q2)

# Similarity should be high
similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"Similarity: {similarity:.3f}")  # Should be ~0.7+

# Hash-based (current) would be random
print(f"Hash of q1: {hash(q1) % 100}")
print(f"Hash of q2: {hash(q2) % 100}")  # Unrelated!
PYEOF

python test_semantic.py
```

### Step 2: Integrate Web Search (4 hours)
```bash
cat > test_web_search.py << 'PYEOF'
from web_search_capability import WebSearchEngine

engine = WebSearchEngine()
query = engine.extract_search_query("What is the capital of France?", level=1)
print(f"Search query: {query.query}")

# Would integrate with tetrahedral model next
PYEOF

python test_web_search.py
```

### Step 3: Test Claude Integration (2 hours)
```bash
export ANTHROPIC_API_KEY=your_key_here
python claude_gaia_eval.py --limit 10  # Test on 10 questions
```

### Step 4: Measure Improvement (1 hour)
```bash
# Run original benchmark
python gaia_official_benchmark.py

# Run new semantic version
python semantic_tetrahedral_model.py

# Compare scores
echo "Original: 0%"
echo "Semantic: X%"
```

---

## Timeline Summary

| Week | Phase | Files | Lines | Expected Accuracy |
|------|-------|-------|-------|-------------------|
| 1 | 1: Semantic | +1 | +200 | 5-10% |
| 2-3 | 2: Web Search | +1 | +150 | 15-25% |
| 4-5 | 3: LLM | Modify | +300 | 40-50% |
| 6-7 | 4: Documents | +1 | +200 | 55-70% |
| 8+ | 5: Advanced | +2 | +200 | 70%+ |

**Total effort:** 100-150 lines of new code per week
**Total payoff:** 0% → 70% accuracy

---

## Success Metrics

At each phase, measure:
1. **Overall accuracy** on 165 GAIA questions
2. **Breakdown by level** (Level 1, 2, 3)
3. **Speed** (questions/second)
4. **Confidence** (model's own confidence score)

```python
# After each phase
def measure_progress():
    results = run_benchmark()
    
    print(f"Overall: {results['overall_accuracy']:.1%}")
    print(f"Level 1: {results['level_1_accuracy']:.1%}")
    print(f"Level 2: {results['level_2_accuracy']:.1%}")
    print(f"Level 3: {results['level_3_accuracy']:.1%}")
    print(f"Speed: {results['qpm']:.0f} questions/min")
    
    return results
```

---

## Resources Needed

### Python Packages
```bash
pip install sentence-transformers anthropic PyPDF2 pytesseract openpyxl SpeechRecognition torch transformers peft

# For macOS: Install tesseract
brew install tesseract
```

### API Keys (if using Claude)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Computing Resources
- **Phase 1-3:** Apple Silicon CPU is fine
- **Phase 4+:** Would benefit from GPU (but not required)
- **Fine-tuning:** ~4GB VRAM for Qwen 0.5B

---

## Conclusion

Your tetrahedral geometry provides a **solid mathematical structure**. 
The real breakthrough comes from integrating it with:
1. **Semantic understanding** (Phase 1)
2. **Knowledge retrieval** (Phase 2)  
3. **Reasoning capability** (Phase 3)
4. **Information integration** (Phase 4)

**Start with Phase 1 + Claude this week. You should see improvements immediately.**

The tetrahedral system then evolves from "pure geometry" to a "**reasoning validator and amplifier**" for LLM outputs.

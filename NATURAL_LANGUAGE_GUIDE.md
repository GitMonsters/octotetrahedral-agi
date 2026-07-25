# 🗣️ Natural Language Prompting Guide

Complete guide to using natural language for prompting and commands with OctoTetrahedral AGI.

---

## 🎯 Overview

Instead of passing raw token IDs, you can:
- ✅ Send text prompts directly
- ✅ Use conversational language
- ✅ Ask questions naturally
- ✅ Issue commands in plain English

---

## 🚀 Quick Start

### Ollama Runtime Setup (Required)

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Start Ollama and pull a model
ollama serve
ollama pull mistral

# 3) (Optional) Configure model fallback + sampling
export OLLAMA_MODEL=mistral
export OLLAMA_FALLBACK_MODELS="llama3.2,phi3"
export OLLAMA_TEMPERATURE=0.7
export OLLAMA_TOP_P=0.9

# 4) Start API
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Current Method (Token IDs)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ" \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 2, 3, 4, 5]}'
```

### Natural Language Method (Recommended)
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Authorization: Bearer qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

---

## 📝 Implementation Guide

### Step 1: Add Text Tokenizer

Add to your `api.py`:

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

@app.post("/prompt")
async def prompt_endpoint(request: dict, authorization: str = Header(None)):
    """Accept natural language prompts"""
    
    # Verify auth
    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    prompt = request.get("prompt", "")
    
    # Convert text to tokens
    tokens = tokenizer.encode(prompt)
    
    # Run inference
    predictions = model(tokens)
    
    # Decode back to text
    response_text = tokenizer.decode(predictions[0])
    
    return {
        "prompt": prompt,
        "response": response_text,
        "device": "mps",
        "success": True
    }
```

### Step 2: Update Model to Support Text

```python
class OctoTetrahedralModel:
    def __init__(self):
        # ... existing code ...
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def generate_from_text(self, prompt: str, max_length: int = 100):
        """Generate response from text prompt"""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return response
```

---

## 🎨 Natural Language Examples

### Example 1: Question Answering
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "mode": "answer"
  }'

# Response:
{
  "prompt": "What is artificial intelligence?",
  "response": "Artificial intelligence is the simulation of human intelligence processes by computer systems...",
  "confidence": 0.92,
  "device": "mps"
}
```

### Example 2: Code Generation
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Write a Python function to calculate factorial",
    "mode": "code",
    "language": "python"
  }'

# Response:
{
  "prompt": "Write a Python function to calculate factorial",
  "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
  "language": "python",
  "device": "mps"
}
```

### Example 3: Conversation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
      {"role": "user", "content": "What can you help me with?"}
    ]
  }'

# Response:
{
  "response": "I can help you with coding, writing, analysis, and answering questions...",
  "device": "mps"
}
```

### Example 4: Text Summarization
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Summarize this text: [long text here]",
    "mode": "summarize",
    "length": "short"
  }'
```

### Example 5: Translation
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Translate to Spanish: Hello, how are you?",
    "mode": "translate",
    "target_language": "spanish"
  }'
```

---

## 📚 Full Implementation Example

Add this to your `api.py`:

```python
from fastapi import FastAPI, Header, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

class PromptHandler:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tokenizer
    
    def process_prompt(self, prompt: str, mode: str = "answer"):
        """Process natural language prompt"""
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response

@app.post("/prompt")
async def handle_prompt(
    request: dict,
    authorization: str = Header(None)
):
    """Natural language prompt endpoint"""
    
    # Verify token
    if not authorization or "Bearer" not in authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    token = authorization.split("Bearer ")[-1]
    if token != "qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Extract prompt
    prompt = request.get("prompt")
    mode = request.get("mode", "answer")
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    
    try:
        # Process prompt
        handler = PromptHandler(model)
        response = handler.process_prompt(prompt, mode)
        
        return {
            "prompt": prompt,
            "response": response,
            "mode": mode,
            "device": "mps",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    request: dict,
    authorization: str = Header(None)
):
    """Conversational chat endpoint"""
    
    # Auth check
    if not authorization or "Bearer" not in authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    
    # Extract messages
    messages = request.get("messages", [])
    
    if not messages:
        raise HTTPException(status_code=400, detail="Messages required")
    
    try:
        # Build conversation context
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        handler = PromptHandler(model)
        response = handler.process_prompt(context)
        
        return {
            "response": response,
            "device": "mps",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🎯 Postman Examples

### In Postman: Create New Request for Natural Language

1. **Method:** POST
2. **URL:** `http://localhost:8000/prompt`
3. **Headers:**
   ```
   Authorization: Bearer qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ
   Content-Type: application/json
   ```
4. **Body (JSON):**
   ```json
   {
     "prompt": "What is machine learning?",
     "mode": "answer"
   }
   ```

---

## 🐍 Python Client Example

```python
import requests
import json

class OctoAIClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def prompt(self, text: str, mode: str = "answer") -> str:
        """Send natural language prompt"""
        payload = {
            "prompt": text,
            "mode": mode
        }
        response = requests.post(
            f"{self.base_url}/prompt",
            headers=self.headers,
            json=payload
        )
        return response.json()["response"]
    
    def chat(self, messages: list) -> str:
        """Send chat messages"""
        payload = {"messages": messages}
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self.headers,
            json=payload
        )
        return response.json()["response"]

# Usage
client = OctoAIClient("qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ")

# Ask a question
response = client.prompt("What is Python?")
print(response)

# Have a conversation
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How can you help me?"}
]
response = client.chat(messages)
print(response)
```

---

## 📊 Comparison: Token IDs vs Natural Language

| Aspect | Token IDs | Natural Language |
|--------|-----------|------------------|
| **Ease of Use** | ❌ Requires tokenization | ✅ Plain English |
| **Flexibility** | ❌ Limited to tokens | ✅ Any text |
| **Error Handling** | ❌ Token misalignment | ✅ Automatic |
| **Performance** | ✅ Faster (no encoding) | ✅ Minimal overhead |
| **Learning Curve** | ❌ Steep | ✅ Gentle |
| **Production Ready** | ✅ Yes | ✅ Yes |

---

## 🎨 Prompt Engineering Tips

### Good Prompts
```
"What is artificial intelligence?"           ✅ Clear
"Explain quantum computing simply"           ✅ Specific intent
"Write a Python function for sorting"        ✅ Task-oriented
```

### Poor Prompts
```
"AI"                                         ❌ Too vague
"stuff"                                      ❌ Unclear
"blah blah blah"                             ❌ Nonsensical
```

### Best Practices
1. **Be specific:** "Write a Python function to calculate Fibonacci" not "code"
2. **Add context:** "Explain like I'm 5 years old" or "for beginners"
3. **Set expectations:** "Give me 3 examples" or "Keep it under 100 words"
4. **Use roles:** "As a data scientist..." or "In the style of Shakespeare..."

---

## 🔧 Configuration Options

Add to your requests:

```json
{
  "prompt": "Your question here",
  "mode": "answer",
  "max_length": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true,
  "num_beams": 5
}
```

**Parameters:**
- `temperature`: 0-1 (lower = more deterministic)
- `top_p`: 0-1 (nucleus sampling)
- `max_length`: Maximum tokens to generate
- `num_beams`: Beam search width

---

## ✅ Next Steps

1. **Update API:** Add natural language endpoints to your FastAPI app
2. **Test in Postman:** Create new requests for `/prompt` and `/chat`
3. **Use Python Client:** Create wrapper for easier integration
4. **Deploy:** Push updates to production
5. **Monitor:** Track natural language request patterns

---

## 📞 Support

For questions about natural language prompting:
- See examples above
- Check Postman collection
- Review Python client code
- Test with `quick_benchmark.py`

---

**Status:** ✅ Ready to implement natural language prompting!
